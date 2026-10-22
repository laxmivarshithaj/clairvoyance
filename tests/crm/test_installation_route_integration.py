"""The route-shape door selection against real Postgres — the SQL id branch
that unit tests, by construction, cannot see.

Skipped unless a database is offered:

    createdb crm_webhook_test
    for f in app/database/migrations/*.sql; do psql -q -d crm_webhook_test -f $f; done
    CRM_WEBHOOK_TEST_DSN=postgresql:///crm_webhook_test uv run pytest \\
        tests/crm/test_installation_route_integration.py -v

Set CRM_WEBHOOK_TEST_DSN and the DB-backed tests run (tests/crm/conftest.py).

The unit tests in test_connector_actions.py stub the accessor, so they can
only prove perform_action THREADS installation_id through. What they cannot
prove is the SQL itself: that ``$3::uuid`` pins exactly the named door across
Postgres's real unique index, that the merchant/connector clauses really
scope the pin, and that the status filter holds against the target's real
type. This file seeds real rows and reads them back through the REAL accessor
(get_installation_for_connector), which is the same door perform_action walks.

Only 'healthy' is INSTALLATION_USABLE (app/crm/connectivity/status.py), so a
merchant holding two healthy merchant_http base_urls is exactly the shape the
installation_id parameter exists for: the id pins the door, else newest wins.
"""

import asyncio
import uuid
from typing import AsyncIterator, Optional

import asyncpg
import pytest

from app.crm.connectivity.db.accessors.installation import (
    get_installation_for_connector,
)
from app.crm.connectivity.schemas.connector import ConnectorInstallation
from tests.crm.conftest import CRM_WEBHOOK_TEST_DSN as DSN

pytestmark = pytest.mark.skipif(
    not DSN, reason="set CRM_WEBHOOK_TEST_DSN to run the DB-backed route-shape tests"
)

MERCHANT = "itest-door-multi"
OTHER_MERCHANT = "itest-door-other"
# connector_key is vocabulary (never a CHECK — migration 060 rationale 1), so
# merchant_http inserts fine without a migration.
CONNECTOR = "merchant_http"


async def _seed(conn: asyncpg.Connection) -> dict:
    """Seed the multi-door merchant and the foreign merchant. Returns the uuids
    this file pins on."""
    await _purge(conn)

    # The OLDEST base_url. Both doors below share merchant + connector, so
    # absent an id the fallback must pick the NEWEST (second) one.
    first = uuid.uuid4()
    await conn.execute(
        "INSERT INTO crm_connector_installation"
        " (id, merchant_id, connector_key, external_account_id, status, installed_at)"
        " VALUES ($1, $2, $3, $4, 'healthy', now() - interval '2 days')",
        first,
        MERCHANT,
        CONNECTOR,
        "acct-first",
    )

    # The NEWEST base_url — the one an unpinned action must act through.
    second = uuid.uuid4()
    await conn.execute(
        "INSERT INTO crm_connector_installation"
        " (id, merchant_id, connector_key, external_account_id, status, installed_at)"
        " VALUES ($1, $2, $3, $4, 'healthy', now())",
        second,
        MERCHANT,
        CONNECTOR,
        "acct-second",
    )

    # A REVOKED door on the same connector: a pin pointing here must read as
    # None (fail closed), even though it IS this merchant's.
    revoked = uuid.uuid4()
    await conn.execute(
        "INSERT INTO crm_connector_installation"
        " (id, merchant_id, connector_key, external_account_id, status, installed_at)"
        " VALUES ($1, $2, $3, $4, 'revoked', now())",
        revoked,
        MERCHANT,
        CONNECTOR,
        "acct-revoked",
    )

    # Another merchant's HEALTHY door: neither the merchant clause nor the
    # connector clause may let a leaked id wander across.
    other = uuid.uuid4()
    await conn.execute(
        "INSERT INTO crm_connector_installation"
        " (id, merchant_id, connector_key, external_account_id, status, installed_at)"
        " VALUES ($1, $2, $3, $4, 'healthy', now())",
        other,
        OTHER_MERCHANT,
        CONNECTOR,
        "acct-other",
    )

    return {
        "first": str(first),
        "second": str(second),
        "revoked": str(revoked),
        "other": str(other),
    }


async def _purge(conn: asyncpg.Connection) -> None:
    """Only this file's merchants. The DSN may point at a shared dev box, so a
    crashed run's rows must not poison the next seeding (the account unique
    index would then refuse the re-insert)."""
    for merchant in (MERCHANT, OTHER_MERCHANT):
        await conn.execute(
            "DELETE FROM crm_channel_binding WHERE merchant_id = $1", merchant
        )
        await conn.execute(
            "DELETE FROM crm_connector_installation WHERE merchant_id = $1", merchant
        )


@pytest.fixture
async def doors() -> AsyncIterator[dict]:
    """Seed real rows and point the app's DB pool at this test database.

    The pool is assigned to app.database.pool directly: it must live on the
    loop this test runs on, and going through init_db_pool() would read the
    five POSTGRES_* variables that static config froze at import instead of
    the DSN this file was given (same reasoning as test_ingress_integration).
    """
    import app.database as database

    pool = await asyncpg.create_pool(dsn=DSN, min_size=1, max_size=2)
    previous, database.pool = database.pool, pool  # type: ignore[assignment]
    conn = await asyncpg.connect(DSN)
    try:
        ids = await _seed(conn)
        yield ids
    finally:
        await _purge(conn)
        await conn.close()
        database.pool = previous
        await pool.close()


def _get(
    merchant: str, connector: str, installation_id: str
) -> Optional[ConnectorInstallation]:
    """The accessor perform_action walks, with the pin SET — the real one."""
    return asyncio.run(
        get_installation_for_connector(
            merchant, connector, installation_id=installation_id
        )
    )


def test_pin_selects_the_named_door_not_the_newest(doors: dict) -> None:
    door = _get(MERCHANT, CONNECTOR, doors["first"])
    assert door is not None
    assert door.id == doors["first"]
    assert door.external_account_id == "acct-first", (
        "the OLDER door, pinned by id, wins over the newer one — the whole "
        "point of the installation_id parameter"
    )


def test_absent_id_falls_back_to_the_newest_usable_door(doors: dict) -> None:
    door = asyncio.run(get_installation_for_connector(MERCHANT, CONNECTOR))
    assert door is not None
    assert door.id == doors["second"]
    assert (
        door.external_account_id == "acct-second"
    ), "no installation_id: ORDER BY installed_at DESC picks the newest"


def test_pinned_revoked_door_fails_closed(doors: dict) -> None:
    door = _get(MERCHANT, CONNECTOR, doors["revoked"])
    assert door is None, (
        "a plan naming a door this merchant revoked must read the same as no "
        "connection — INSTALLATION_USABLE filters it out even by id"
    )


def test_other_merchants_id_selects_nothing(doors: dict) -> None:
    door = _get(MERCHANT, CONNECTOR, doors["other"])
    assert door is None, (
        "a leaked id from another tenant must select nothing, not the wrong "
        "door (the merchant clause scopes the pin)"
    )


def test_pinned_id_on_a_different_connector_selects_nothing(doors: dict) -> None:
    # The pin is only meaningful inside (merchant_id, connector_key): a
    # merchant_http node pointing its id at a whatsapp door must select nothing.
    door = _get(MERCHANT, "whatsapp", doors["first"])
    assert (
        door is None
    ), "the connector clause scopes the pin — a foreign-connector id picks no door"


def test_unrecognised_id_selects_nothing(doors: dict) -> None:
    door = _get(MERCHANT, CONNECTOR, str(uuid.uuid4()))
    assert door is None, "a random id is no doorway"
