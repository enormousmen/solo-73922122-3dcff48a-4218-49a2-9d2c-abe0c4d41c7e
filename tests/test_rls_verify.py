"""Integration tests for POST /rls/verify (row-level access verification).

Requires a running API (API_BASE) whose DATABASE_URL logs in as a superuser,
plus direct superuser access (TEST_DATABASE_URL) to build the fixtures:
schemas, plain tables with single-column integer PKs, RLS policies and roles.

Run:
    API_BASE=http://127.0.0.1:8000 \
    TEST_DATABASE_URL=postgresql://postgres@/diag?host=/tmp \
    pytest tests/test_rls_verify.py -v
"""
from __future__ import annotations

import os
import time

import httpx
import psycopg
import pytest

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
TEST_DSN = os.environ.get("TEST_DATABASE_URL")

SCHEMA = "rls_demo"

# Every fixture role/schema is created fresh and dropped in teardown.
ROLES = [
    "rls_reader",       # SELECT granted, sees only own tenant via policy
    "rls_other",        # SELECT granted, policy never matches fixture rows
    "rls_nogrants",     # role exists, no privileges at all
    "rls_owner",        # owns an RLS-enabled table (without FORCE)
    "rls_bypass",       # BYPASSRLS role, SELECT granted
]


def _api_up() -> bool:
    try:
        return httpx.get(f"{API_BASE}/health", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _api_up() or not TEST_DSN,
    reason="API not reachable or TEST_DATABASE_URL not set",
)


@pytest.fixture(scope="module", autouse=True)
def fixtures():
    admin = psycopg.connect(TEST_DSN, autocommit=True)
    try:
        admin.execute("SELECT current_setting('is_superuser')").fetchone()
    except Exception:
        pytest.skip("TEST_DATABASE_URL does not connect as superuser")
    admin.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    for r in ROLES:
        admin.execute(f'DROP ROLE IF EXISTS "{r}"')
    # Create every role first so later GRANTs/OWNER clauses resolve.
    admin.execute('CREATE ROLE "rls_reader" LOGIN')
    admin.execute('CREATE ROLE "rls_other" LOGIN')
    admin.execute('CREATE ROLE "rls_nogrants" LOGIN')
    admin.execute('CREATE ROLE "rls_owner" LOGIN')
    admin.execute('CREATE ROLE "rls_bypass" LOGIN BYPASSRLS')
    admin.execute(f"CREATE SCHEMA {SCHEMA}")

    # Main table: RLS enabled, policy isolates rows by tenant_id.
    admin.execute(
        f"CREATE TABLE {SCHEMA}.tenant_rows ("
        "id bigint PRIMARY KEY, tenant_id int NOT NULL, secret text)"
    )
    admin.execute(
        f"INSERT INTO {SCHEMA}.tenant_rows (id, tenant_id, secret) "
        "VALUES (1, 100, 's1'), (2, 200, 's2'), (3, 300, 's3')"
    )
    admin.execute(f"ALTER TABLE {SCHEMA}.tenant_rows ENABLE ROW LEVEL SECURITY")
    # Policy keys off the actual database identity (current_user), which the
    # verifier establishes through SET LOCAL ROLE on its own connection:
    # rls_reader sees tenant 100 only; every other non-bypassing role maps
    # to a tenant id that no fixture row has.
    admin.execute(
        f"CREATE POLICY own_tenant ON {SCHEMA}.tenant_rows "
        "FOR SELECT USING ("
        "tenant_id = CASE current_user WHEN 'rls_reader' THEN 100 ELSE -1 END)"
    )
    for r in ("rls_reader", "rls_other", "rls_bypass"):
        admin.execute(
            f'GRANT USAGE ON SCHEMA {SCHEMA} TO "{r}"; '
            f"GRANT SELECT ON {SCHEMA}.tenant_rows TO \"{r}\""
        )

    # Owner table: RLS enabled but NOT forced -> owner bypasses RLS by default.
    admin.execute(
        f"CREATE TABLE {SCHEMA}.owned_rows (id integer PRIMARY KEY, v text)"
    )
    admin.execute(f"INSERT INTO {SCHEMA}.owned_rows (id, v) VALUES (7, 'x')")
    admin.execute(f"ALTER TABLE {SCHEMA}.owned_rows OWNER TO rls_owner")
    # Ownership implies table-level SELECT; the owner still needs USAGE on
    # the containing schema to reach it.
    admin.execute(f'GRANT USAGE ON SCHEMA {SCHEMA} TO "rls_owner"')
    admin.execute(f"ALTER TABLE {SCHEMA}.owned_rows ENABLE ROW LEVEL SECURITY")
    admin.execute(
        f"CREATE POLICY deny_all ON {SCHEMA}.owned_rows FOR SELECT USING (false)"
    )

    # Table with no primary key.
    admin.execute(f"CREATE TABLE {SCHEMA}.no_pk (id int, v text)")
    # Table with a composite primary key.
    admin.execute(
        f"CREATE TABLE {SCHEMA}.composite_pk (a int, b int, PRIMARY KEY (a, b))"
    )
    # Table with a non-integer primary key.
    admin.execute(
        f"CREATE TABLE {SCHEMA}.text_pk (id text PRIMARY KEY, v text)"
    )
    # A view (not a plain table).
    admin.execute(f"CREATE VIEW {SCHEMA}.a_view AS SELECT 1 AS one")

    yield

    admin.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    for r in ROLES:
        admin.execute(f'DROP ROLE IF EXISTS "{r}"')
    admin.close()


@pytest.fixture()
def client():
    with httpx.Client(base_url=API_BASE, timeout=30) as c:
        yield c


def verify(client, **over):
    body = {
        "schema_name": SCHEMA,
        "table_name": "tenant_rows",
        "role_name": "rls_reader",
        "pk_value": 1,
    }
    body.update(over)
    r = client.post("/rls/verify", json=body)
    assert r.status_code == 200, r.text
    return r.json()


class TestConclusiveOutcomes:
    def test_row_not_found(self, client):
        out = verify(client, pk_value=999999)
        assert out["outcome"] == "row_not_found"
        assert out["reason"] is None
        assert out["target"]["primary_key_column"] == "id"
        assert out["target"]["is_plain_table"] is True
        assert out["table"] == {
            "row_security_enabled": True,
            "force_row_security": False,
            "owner": "postgres",
        }
        assert out["same_snapshot"] is True
        assert out["dedicated_connection"] is True

    def test_visible_when_policy_matches(self, client):
        # rls_reader maps to tenant 100 and must see its row (id=1).
        out = verify(client, role_name="rls_reader", pk_value=1)
        assert out["outcome"] == "visible", out
        assert out["role"] == {
            "name": "rls_reader",
            "exists": True,
            "is_superuser": False,
            "bypass_rls": False,
            "is_table_owner": False,
        }

    def test_visible_denied_for_other_tenant_of_same_role(self, client):
        # The same role's policy matches only tenant 100: row id=2 belongs
        # to tenant 200 and must be reported hidden, not visible.
        out = verify(client, role_name="rls_reader", pk_value=2)
        assert out["outcome"] == "hidden_by_rls", out

    def test_hidden_by_rls_when_policy_does_not_match(self, client):
        # rls_other never matches any fixture tenant: row exists, SELECT
        # granted, but the policy hides it.
        out = verify(client, role_name="rls_other", pk_value=1)
        assert out["outcome"] == "hidden_by_rls", out
        assert out["target"]["pk_value"] == 1

    def test_no_read_privilege(self, client):
        out = verify(client, role_name="rls_nogrants", pk_value=1)
        assert out["outcome"] == "no_read_privilege", out
        assert out["role"]["exists"] is True
        assert "SELECT" in out["detail"]

    def test_row_contents_never_leak(self, client):
        out = verify(client, role_name="rls_reader", pk_value=1)
        # Whatever the verdict, no column data may appear anywhere in the body.
        assert "s1" not in str(out)
        assert "secret" not in str(out)


class TestOwnerBypassAndForce:
    def test_table_owner_bypasses_rls_without_force(self, client):
        # Owner of an RLS-enabled (but not FORCE) table bypasses the policy:
        # the deny_all policy must NOT hide the row from the owner.
        out = verify(
            client,
            table_name="owned_rows",
            role_name="rls_owner",
            pk_value=7,
        )
        assert out["outcome"] == "visible", out
        assert out["role"]["is_table_owner"] is True
        assert out["table"]["row_security_enabled"] is True
        assert out["table"]["force_row_security"] is False

    def test_force_rls_makes_owner_subject_to_policy(self, client):
        admin = psycopg.connect(TEST_DSN, autocommit=True)
        try:
            admin.execute(f"ALTER TABLE {SCHEMA}.owned_rows FORCE ROW LEVEL SECURITY")
            out = verify(
                client,
                table_name="owned_rows",
                role_name="rls_owner",
                pk_value=7,
            )
            assert out["outcome"] == "hidden_by_rls", out
            assert out["table"]["force_row_security"] is True
        finally:
            admin.execute(
                f"ALTER TABLE {SCHEMA}.owned_rows NO FORCE ROW LEVEL SECURITY"
            )
            admin.close()

    def test_bypassrls_role_sees_hidden_row(self, client):
        # rls_bypass has BYPASSRLS and SELECT: the policy cannot hide the row.
        out = verify(client, role_name="rls_bypass", pk_value=2)
        assert out["outcome"] == "visible", out
        assert out["role"]["bypass_rls"] is True


class TestUnknownVerdicts:
    def test_missing_table(self, client):
        out = verify(client, table_name="does_not_exist")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "table_missing"

    def test_missing_schema(self, client):
        out = verify(client, schema_name="no_such_schema_xyz", table_name="t")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "table_missing"

    def test_view_is_not_plain_table(self, client):
        out = verify(client, table_name="a_view")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "not_plain_table"

    def test_missing_primary_key(self, client):
        out = verify(client, table_name="no_pk")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "no_primary_key"

    def test_composite_primary_key(self, client):
        out = verify(client, table_name="composite_pk")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "composite_primary_key"

    def test_non_integer_primary_key(self, client):
        out = verify(client, table_name="text_pk")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "pk_not_integer"

    def test_missing_role(self, client):
        out = verify(client, role_name="no_such_role_xyz")
        assert out["outcome"] == "unknown"
        assert out["reason"] == "role_missing"
        assert out["role"]["exists"] is False
        # table facts were still established before the role check failed
        assert out["target"]["is_plain_table"] is True
        assert out["table"]["row_security_enabled"] is True

    def test_request_validation(self, client):
        bad_bodies = [
            {"schema_name": "", "table_name": "t", "role_name": "r",
             "pk_value": 1},
            {"schema_name": "s", "table_name": "t", "role_name": "r",
             "pk_value": "not-an-int"},
            {"schema_name": "s", "table_name": "t", "role_name": "r"},
        ]
        for body in bad_bodies:
            r = client.post("/rls/verify", json=body)
            assert r.status_code == 422, (body, r.text)


class TestIsolationFromDiagnosticPool:
    def test_verify_uses_dedicated_connection_not_pool(self, client):
        before = client.get("/pool").json()
        assert before["size"] == 2
        out = verify(client, role_name="rls_other", pk_value=1)
        assert out["dedicated_connection"] is True
        after = client.get("/pool").json()
        # the two pooled connections are untouched and still the same
        # generation; no third long-lived connection appeared in the pool
        assert after["size"] == 2
        assert [c["generation"] for c in after["connections"]] == [
            c["generation"] for c in before["connections"]
        ]
        assert all(c["state"] == "idle" for c in after["connections"])

    def test_jobs_still_run_and_cancel_while_verifying(self, client):
        # occupy one pooled connection with a slow job made of short sleeps
        # so a cancel is honored cooperatively between statements even in
        # sandboxes where PostgreSQL signal delivery is unavailable.
        sleeper = client.post(
            "/jobs", json={"statements": ["SELECT pg_sleep(2)"] * 8}
        ).json()["job_id"]
        deadline = time.monotonic() + 10
        while True:
            j = client.get(f"/jobs/{sleeper}").json()
            if j["status"] == "running":
                break
            assert time.monotonic() < deadline
            time.sleep(0.1)
        try:
            # verification still completes promptly on its own connection
            t0 = time.monotonic()
            out = verify(client, role_name="rls_other", pk_value=2)
            assert time.monotonic() - t0 < 5.0
            assert out["outcome"] == "hidden_by_rls"
            # and the other pooled connection keeps serving jobs
            quick = client.post(
                "/jobs", json={"statements": ["SELECT 42 AS answer"]}
            ).json()["job_id"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                j = client.get(f"/jobs/{quick}").json()
                if j["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.15)
            assert j["status"] == "succeeded"
            assert j["statements"][0]["rows"] == [[42]]
            # cancel path still works
            r = client.post(f"/jobs/{sleeper}/cancel", json={})
            assert r.status_code == 202
        finally:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                j = client.get(f"/jobs/{sleeper}").json()
                if j["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.2)
            assert j["status"] == "cancelled"

    def test_no_session_state_left_in_pool(self, client):
        # A verification must not leak SET LOCAL ROLE state: the pooled
        # connections run as the login role before and after.
        check = client.post(
            "/jobs", json={"statements": ["SELECT current_user, current_setting('is_superuser')"]}
        ).json()["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            j = client.get(f"/jobs/{check}").json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.15)
        assert j["status"] == "succeeded"
        assert j["statements"][0]["rows"][0][0] == "postgres"
        assert j["statements"][0]["rows"][0][1] == "on"
