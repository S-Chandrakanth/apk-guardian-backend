"""Registration, login, token authentication."""

import time

import jwt
import pytest
from sqlalchemy import select

from app.database.models import User
from app.security.tokens import ALGORITHM, ISSUER
from tests.api.conftest import PASSWORD, SECRET, login, register


# ---------------------------------------------------------------- register
def test_register_success_never_returns_the_password_or_its_hash(client):
    r = register(client)
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "alice@example.com" and body["name"] == "Alice" and body["id"] >= 1
    text = r.text.lower()
    assert PASSWORD.lower() not in text and "scrypt" not in text and "hash" not in text and "password" not in text


def test_register_stores_a_salted_hash_not_the_password(client):
    register(client)
    with client.app.state.session_factory() as db:
        user = db.scalar(select(User).where(User.email == "alice@example.com"))
        assert user.password_hash != PASSWORD and PASSWORD not in user.password_hash
        assert user.password_hash.startswith("scrypt$")


def test_register_duplicate_email_is_409_case_insensitive(client):
    assert register(client).status_code == 201
    assert register(client).status_code == 409
    assert register(client, email="ALICE@Example.com").status_code == 409


def test_register_normalises_email_to_lowercase(client):
    r = register(client, email="Mixed.Case@Example.COM")
    assert r.status_code == 201 and r.json()["email"] == "mixed.case@example.com"
    assert login(client, "mixed.case@example.com").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": "A", "email": "not-an-email", "password": PASSWORD},
        {"name": "A", "email": "a@example.com", "password": "short"},
        {"name": "A", "email": "a@example.com", "password": "x" * 129},
        {"name": "   ", "email": "a@example.com", "password": PASSWORD},
        {"name": "A", "email": "a@example.com"},
    ],
)
def test_register_validation_errors_are_422(client, payload):
    assert client.post("/register", json=payload).status_code == 422


def test_validation_error_does_not_echo_the_password(client):
    pw = "tiny-pw-9"
    r = client.post("/register", json={"name": "A", "email": "a@example.com", "password": pw})
    assert r.status_code == 422
    assert pw not in r.text
    assert set(r.json()["detail"][0]) <= {"loc", "msg", "type"}


def test_password_equal_to_email_is_refused(client):
    r = client.post("/register", json={"name": "A", "email": "same.value.here@example.com", "password": "same.value.here@example.com"})
    assert r.status_code == 422


def test_oversized_json_body_is_413(client):
    r = client.post("/register", content=b"{" + b" " * (70 * 1024) + b"}", headers={"Content-Type": "application/json"})
    assert r.status_code == 413


# ---------------------------------------------------------------- login
def test_login_success_returns_bearer_token(client):
    register(client)
    r = login(client)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600 and len(body["access_token"]) > 40
    assert PASSWORD not in r.text


def test_wrong_password_and_unknown_email_look_identical(client):
    register(client)
    wrong = login(client, password="not the right password")
    unknown = login(client, email="nobody@example.com")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert wrong.headers["www-authenticate"] == "Bearer"


def test_login_is_rate_limited_per_ip_and_email(client):
    register(client)
    for _ in range(3):  # login_max_failures=3 in the test settings
        assert login(client, password="wrong password!").status_code == 401
    blocked = login(client, password="wrong password!")
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0
    # even the correct password is refused while locked out
    assert login(client).status_code == 429
    # a different account is unaffected
    register(client, "bob@example.com", name="Bob")
    assert login(client, "bob@example.com").status_code == 200


def test_successful_login_resets_the_failure_counter(client):
    register(client)
    for _ in range(2):
        login(client, password="wrong password!")
    assert login(client).status_code == 200
    for _ in range(2):
        assert login(client, password="wrong password!").status_code == 401


def test_login_upgrades_an_outdated_hash(client, monkeypatch):
    from app.security import passwords

    register(client)
    monkeypatch.setattr(passwords, "SCRYPT_N", 2 ** 11)  # policy raised after registration
    assert login(client).status_code == 200
    with client.app.state.session_factory() as db:
        assert db.scalar(select(User.password_hash)).startswith("scrypt$2048$")


# ---------------------------------------------------------------- /me and token checks
def test_me_returns_only_the_caller(client, alice, bob):
    me = client.get("/me", headers=alice)
    assert me.status_code == 200 and me.json()["email"] == "alice@example.com"
    assert client.get("/me", headers=bob).json()["email"] == "bob@example.com"


def _token(sub="1", **over):
    now = int(time.time())
    claims = {"sub": sub, "iat": now, "exp": now + 600, "iss": ISSUER, "jti": "x"}
    claims.update(over)
    return jwt.encode(claims, SECRET, algorithm=ALGORITHM)


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer not.a.token"},
    {"Authorization": "Basic YWxpY2U6cGFzcw=="},
    {"Authorization": "Bearer "},
])
def test_me_rejects_missing_or_malformed_credentials(client, headers):
    r = client.get("/me", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_me_rejects_expired_wrong_secret_and_unsigned_tokens(client):
    register(client)
    now = int(time.time())
    assert client.get("/me", headers={"Authorization": f"Bearer {_token()}"}).status_code == 200  # control
    expired = _token(exp=now - 10)
    wrong_key = jwt.encode({"sub": "1", "iat": now, "exp": now + 600, "iss": ISSUER}, "w" * 48, algorithm=ALGORITHM)
    unsigned = jwt.encode({"sub": "1", "iat": now, "exp": now + 600, "iss": ISSUER}, key="", algorithm="none")
    for token in (expired, wrong_key, unsigned):
        assert client.get("/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_for_a_deleted_user_is_rejected(client):
    register(client)
    headers = {"Authorization": f"Bearer {login(client).json()['access_token']}"}
    assert client.get("/me", headers=headers).status_code == 200
    with client.app.state.session_factory() as db:
        db.delete(db.scalar(select(User)))
        db.commit()
    assert client.get("/me", headers=headers).status_code == 401


@pytest.mark.parametrize("path", ["/scans", "/reports/1", "/me"])
def test_protected_endpoints_need_a_token(client, path):
    assert client.get(path).status_code == 401
