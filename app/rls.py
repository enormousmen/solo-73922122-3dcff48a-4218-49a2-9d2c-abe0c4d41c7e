"""Row-level access verification for a single table row and role.

The check answers one narrow question, without ever returning row contents:

* ``row_not_found``    -- the primary-key row does not exist;
* ``no_read_privilege``-- the row exists but the role lacks SELECT;
* ``hidden_by_rls``    -- the row exists, SELECT is granted, but a row-level
                          security policy filters the row out for the role;
* ``visible``          -- the role's own SELECT returns the row;
* ``unknown``          -- the target or the role identity could not be
                          established (see ``reason``); no misleading verdict
                          is ever emitted.

How correctness is achieved
---------------------------

* **Two reads, one snapshot, one table.**  A dedicated short-lived connection
  opens ``BEGIN ISOLATION LEVEL REPEATABLE READ``.  Every catalog lookup, the
  privileged existence read and the role's read run inside that single
  transaction, so all of them see one identical snapshot of the same table.
* **Real role identity.**  The role read is performed after
  ``SET LOCAL ROLE <role>``: privilege checks and RLS policies are evaluated
  by PostgreSQL exactly as if the role had connected itself.  Table owners,
  ``BYPASSRLS`` roles and ``FORCE ROW LEVEL SECURITY`` need no special
  handling -- the engine applies its normal rules and we report the observed
  result.
* **Dedicated connection.**  No pooled diagnostic connection is touched; the
  verification therefore never occupies one of the two long-lived
  connections and cannot interfere with job execution or cancellation.  The
  connection is committed/rolled back and closed before the request returns,
  so no role or transaction state is left behind, and no role is ever created
  or dropped.
"""
from __future__ import annotations

import logging
import threading

import psycopg
from psycopg import sql
from psycopg.errors import InsufficientPrivilege

from . import config

log = logging.getLogger("diag.rls")

# Public outcome vocabulary.
ROW_NOT_FOUND = "row_not_found"
NO_READ_PRIVILEGE = "no_read_privilege"
HIDDEN_BY_RLS = "hidden_by_rls"
VISIBLE = "visible"
UNKNOWN = "unknown"

# PostgreSQL type OIDs for the allowed single-column integer primary keys.
_INT_PK_TYPES = {
    21: "smallint",  # int2
    23: "integer",  # int4
    20: "bigint",  # int8
}

_RELKIND_LABELS = {
    "r": "ordinary table",
    "v": "view",
    "m": "materialized view",
    "f": "foreign table",
    "p": "partitioned table",
    "S": "sequence",
}


class _Unknown(Exception):
    """Internal control-flow signal: verification cannot be conclusive."""

    def __init__(self, reason: str, detail: str, fragments: dict | None = None):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        # Optional already-discovered target/role/table facts to surface
        # alongside the unknown verdict instead of blank fields.
        self.fragments = fragments or {}


class RLSVerifier:
    """Runs verifications on demand, each on its own short-lived connection."""

    def __init__(self, max_concurrency: int | None = None):
        self._slots = threading.BoundedSemaphore(
            max_concurrency or config.RLS_VERIFY_MAX_CONCURRENCY
        )

    def verify(
        self, schema_name: str, table_name: str, role_name: str, pk_value: int
    ) -> dict:
        if not self._slots.acquire(timeout=config.RLS_VERIFY_ACQUIRE_TIMEOUT_S):
            return _unknown_response(
                schema_name,
                table_name,
                role_name,
                pk_value,
                reason="busy",
                detail="too many concurrent verification requests; retry later",
            )
        try:
            return self._verify(schema_name, table_name, role_name, pk_value)
        finally:
            self._slots.release()

    # ------------------------------------------------------------------ core

    def _verify(
        self, schema_name: str, table_name: str, role_name: str, pk_value: int
    ) -> dict:
        base = {
            "schema_name": schema_name,
            "table_name": table_name,
            "primary_key_column": None,
            "pk_value": pk_value,
            "is_plain_table": False,
        }
        try:
            conn = psycopg.connect(
                config.DATABASE_URL,
                autocommit=False,
                connect_timeout=config.CONNECT_TIMEOUT_S,
                # Keep server-side prepared statements off, mirroring the
                # diagnostic pool and keeping the short-lived connection cheap.
                prepare_threshold=None,
            )
        except Exception as exc:
            log.warning("rls verify: cannot open dedicated connection", exc_info=True)
            return _unknown_response(
                schema_name,
                table_name,
                role_name,
                pk_value,
                reason="backend_unavailable",
                detail=f"cannot open verification connection: {_err(exc)}",
                target=base,
            )

        try:
            return self._run_on_connection(
                conn, schema_name, table_name, role_name, pk_value
            )
        except _Unknown as exc:
            _safe_rollback(conn)
            return _unknown_response(
                schema_name,
                table_name,
                role_name,
                pk_value,
                reason=exc.reason,
                detail=exc.detail,
                fragments=exc.fragments,
            )
        except Exception as exc:  # pragma: no cover - defensive catch-all
            log.exception("rls verify: unexpected backend error")
            _safe_rollback(conn)
            return _unknown_response(
                schema_name,
                table_name,
                role_name,
                pk_value,
                reason="backend_error",
                detail=f"verification failed: {_err(exc)}",
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _run_on_connection(
        self,
        conn: psycopg.Connection,
        schema_name: str,
        table_name: str,
        role_name: str,
        pk_value: int,
    ) -> dict:
        # Facts are accumulated into these envelopes so that even an 'unknown'
        # verdict can report everything that was actually established.
        target = {
            "schema_name": schema_name,
            "table_name": table_name,
            "primary_key_column": None,
            "pk_value": pk_value,
            "is_plain_table": False,
        }
        table_info = {
            "row_security_enabled": None,
            "force_row_security": None,
            "owner": None,
        }
        role_info = {
            "name": role_name,
            "exists": None,
            "is_superuser": None,
            "bypass_rls": None,
            "is_table_owner": None,
        }
        fragments = {"target": target, "table": table_info, "role": role_info}

        # One transaction == one snapshot for every read that follows.
        # It must be opened before any query (psycopg would otherwise have
        # started an implicit transaction, making BEGIN illegal).
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
        conn.execute(
            sql.SQL("SET LOCAL statement_timeout = {}").format(
                sql.Literal(f"{config.RLS_STATEMENT_TIMEOUT_MS}ms")
            )
        )
        # Fully-qualified identifiers are used everywhere; lock the search
        # path anyway so nothing can resolve unexpectedly.
        conn.execute("SET LOCAL search_path TO pg_catalog")

        # The login identity must be highly privileged: only then is the
        # "does the row really exist" read independent of policies/grants, so
        # 'row_not_found' means the row is truly absent rather than hidden
        # from us as well.
        login = conn.execute(
            "SELECT r.rolsuper, r.rolbypassrls, r.rolname "
            "FROM pg_roles r WHERE r.rolname = current_user"
        ).fetchone()
        if login is None or not login[0]:
            raise _Unknown(
                "insufficient_privileges",
                "the verification login role is not a superuser; an "
                "independent privileged existence read cannot be guaranteed",
            )

        table_oid, owner_oid = self._resolve_table(
            conn, schema_name, table_name, target, table_info
        )
        pk_column = self._resolve_integer_pk(conn, table_oid, target, fragments)
        # A plain relation with a valid single-column integer PK fully
        # identifies the target row's address.
        target["is_plain_table"] = True

        self._resolve_role(conn, role_name, owner_oid, role_info, fragments)

        # Read #1: privileged existence check (still the superuser identity,
        # RLS and grants do not apply to superusers).
        rel = sql.Identifier(schema_name, table_name)
        col = sql.Identifier(pk_column)
        exists_sql = sql.SQL("SELECT 1 FROM {} WHERE {} = %s LIMIT 1").format(rel, col)
        try:
            row = conn.execute(exists_sql, (pk_value,)).fetchone()
        except psycopg.errors.QueryCanceled as exc:
            # statement_timeout / lock wait cancelled: the existence of the
            # row simply could not be established -- never guess.
            raise _Unknown(
                "existence_unconfirmable",
                f"privileged existence read was cancelled "
                f"(timeout or lock wait): {_err(exc)}",
                fragments,
            )
        except Exception as exc:
            raise _Unknown(
                "existence_unconfirmable",
                f"privileged existence read failed: {_err(exc)}",
                fragments,
            )

        if row is None:
            result = self._conclusive(
                ROW_NOT_FOUND,
                f"no row with {pk_column} = {pk_value} exists in "
                f"{schema_name}.{table_name}",
                target,
                role_info,
                table_info,
            )
            conn.commit()  # end the transaction; no state left open
            return result

        # Read #2: the same row, same snapshot, but with the role's own
        # database identity. SET LOCAL undoes itself at transaction end, and
        # the dedicated connection is closed afterwards anyway.
        try:
            conn.execute(
                sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role_name))
            )
        except Exception as exc:
            raise _Unknown(
                "cannot_assume_role",
                f"SET ROLE to {role_name} failed: {_err(exc)}",
                fragments,
            )

        # Explicit grant check first so 'no SELECT' and 'filtered by RLS'
        # stay distinguishable even though both surface as an empty read.
        try:
            can_select = conn.execute(
                "SELECT has_table_privilege(current_user, %s, 'SELECT')",
                (table_oid,),
            ).fetchone()[0]
        except Exception as exc:
            raise _Unknown(
                "privilege_unconfirmable",
                f"privilege probe for {role_name} failed: {_err(exc)}",
                fragments,
            )

        if not can_select:
            result = self._conclusive(
                NO_READ_PRIVILEGE,
                f"role {role_name} has no SELECT privilege on "
                f"{schema_name}.{table_name}",
                target,
                role_info,
                table_info,
            )
            conn.commit()
            return result

        try:
            seen = conn.execute(exists_sql, (pk_value,)).fetchone() is not None
        except InsufficientPrivilege as exc:
            # Defensive: column-level grants or a revoked default can still
            # deny the read even when table-level SELECT shows granted.
            result = self._conclusive(
                NO_READ_PRIVILEGE,
                f"role {role_name} SELECT was denied: {_err(exc)}",
                target,
                role_info,
                table_info,
            )
            conn.commit()
            return result

        outcome = VISIBLE if seen else HIDDEN_BY_RLS
        if seen:
            detail = (
                f"role {role_name} can read the row "
                f"({pk_column} = {pk_value})"
            )
        else:
            rls = "enabled" if table_info["row_security_enabled"] else "not enabled"
            detail = (
                f"the row exists and {role_name} has SELECT, but the read "
                f"returns no row (row-level security {rls} on the table)"
            )
        result = self._conclusive(outcome, detail, target, role_info, table_info)
        conn.commit()
        return result

    # ------------------------------------------------------------- introspect

    def _resolve_table(self, conn: psycopg.Connection, schema_name: str,
                       table_name: str, target: dict, table_info: dict
                       ) -> tuple[int, int]:
        row = conn.execute(
            """
            SELECT c.oid,
                   c.relkind,
                   c.relrowsecurity,
                   c.relforcerowsecurity,
                   c.relowner,
                   pg_get_userbyid(c.relowner)
              FROM pg_class c
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = %s
               AND c.relname = %s
               AND substring(n.nspname for 9) <> 'pg_temp_'
               AND substring(n.nspname for 16) <> 'pg_toast_temp_'
            """,
            (schema_name, table_name),
        ).fetchone()
        if row is None:
            raise _Unknown(
                "table_missing",
                f"relation {schema_name}.{table_name} does not exist",
            )
        (oid, relkind, rls_enabled, force_rls, owner_oid, owner_name) = row
        table_info["row_security_enabled"] = bool(rls_enabled)
        table_info["force_row_security"] = bool(force_rls)
        table_info["owner"] = owner_name
        if relkind != "r":
            kind = _RELKIND_LABELS.get(relkind, f"relkind '{relkind}'")
            raise _Unknown(
                "not_plain_table",
                f"{schema_name}.{table_name} is a {kind}, not a plain table",
                {"target": target, "table": table_info},
            )
        return oid, owner_oid

    def _resolve_integer_pk(self, conn: psycopg.Connection, table_oid: int,
                            target: dict, fragments: dict) -> str:
        rows = conn.execute(
            """
            SELECT a.attname, a.atttypid
              FROM pg_index i
              JOIN pg_attribute a
                ON a.attrelid = i.indrelid
               AND a.attnum = ANY (i.indkey)
             WHERE i.indrelid = %s
               AND i.indisprimary
             ORDER BY array_position(i.indkey, a.attnum)
            """,
            (table_oid,),
        ).fetchall()
        if not rows:
            raise _Unknown(
                "no_primary_key",
                "the table has no primary key; verification requires a "
                "single-column integer primary key",
                fragments,
            )
        if len(rows) > 1:
            cols = ", ".join(r[0] for r in rows)
            raise _Unknown(
                "composite_primary_key",
                f"composite primary key ({cols}) is not supported; the "
                "primary key must be a single integer column",
                fragments,
            )
        col_name, type_oid = rows[0]
        if type_oid not in _INT_PK_TYPES:
            raise _Unknown(
                "pk_not_integer",
                f"primary key column {col_name} is not an integer type "
                "(smallint/integer/bigint)",
                fragments,
            )
        target["primary_key_column"] = col_name
        return col_name

    def _resolve_role(self, conn: psycopg.Connection, role_name: str,
                      owner_oid: int, role_info: dict, fragments: dict) -> None:
        row = conn.execute(
            "SELECT oid, rolname, rolsuper, rolbypassrls "
            "FROM pg_roles WHERE rolname = %s",
            (role_name,),
        ).fetchone()
        if row is None:
            role_info["exists"] = False
            raise _Unknown(
                "role_missing",
                f"database role {role_name} does not exist",
                fragments,
            )
        oid, name, is_super, bypass_rls = row
        role_info.update(
            name=name,
            exists=True,
            is_superuser=bool(is_super),
            bypass_rls=bool(bypass_rls),
            is_table_owner=(oid == owner_oid),
        )

    # -------------------------------------------------------------- packaging

    def _conclusive(self, outcome: str, detail: str, target: dict,
                    role_info: dict, table_info: dict) -> dict:
        return {
            "outcome": outcome,
            "reason": None,
            "detail": detail,
            "target": dict(target),
            "role": dict(role_info),
            "table": dict(table_info),
            "same_snapshot": True,
            "dedicated_connection": True,
        }


# --------------------------------------------------------------------- helpers


def _err(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _safe_rollback(conn: psycopg.Connection) -> None:
    try:
        conn.rollback()
    except Exception:
        pass


def _unknown_response(
    schema_name: str,
    table_name: str,
    role_name: str,
    pk_value: int,
    reason: str,
    detail: str,
    target: dict | None = None,
    fragments: dict | None = None,
) -> dict:
    default_target = {
        "schema_name": schema_name,
        "table_name": table_name,
        "primary_key_column": None,
        "pk_value": pk_value,
        "is_plain_table": False,
    }
    if fragments and fragments.get("target"):
        # Facts discovered during the failed verification override the blank
        # envelope wholesale (the fragment dict contains the full envelope).
        default_target.update(fragments["target"])
    elif target is not None:
        default_target.update(target)

    role_info = {
        "name": role_name,
        "exists": None,
        "is_superuser": None,
        "bypass_rls": None,
        "is_table_owner": None,
    }
    if fragments and fragments.get("role"):
        role_info.update(fragments["role"])

    table_info = {
        "row_security_enabled": None,
        "force_row_security": None,
        "owner": None,
    }
    if fragments and fragments.get("table"):
        table_info.update(fragments["table"])

    return {
        "outcome": UNKNOWN,
        "reason": reason,
        "detail": detail,
        "target": default_target,
        "role": role_info,
        "table": table_info,
        "same_snapshot": False,
        "dedicated_connection": True,
    }
