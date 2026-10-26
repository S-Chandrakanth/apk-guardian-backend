"""Unit tests for passwords, tokens, the login limiter, settings, and the CLI."""

import time

import jwt
import pytest

from app.config import MIN_SECRET_KEY_LENGTH, PROJECT_ROOT, Settings
from app.security import passwords
from app.security.rate_limit import LoginRateLimiter
from app.security.tokens import ALGORITHM, ISSUER, create_access_token, decode_access_token

SECRET = "s" * 48


# ---------------- passwords ----------------
def test_hash_and_verify_roundtrip():
    stored = passwords.hash_password("a long passphrase 1")
    assert stored != "a long passphrase 1"
    assert passwords.verify_password("a long passphrase 1", stored)
    assert not passwords.verify_password("a long passphrase 2", stored)
    assert not passwords.verify_password("", stored)


def test_hashes_are_salted():
    assert passwords.hash_password("same password!") != passwords.hash_password("same password!")


def test_production_cost_parameters_are_the_documented_ones(monkeypatch):
    monkeypatch.undo()  # drop the cheap test settings from the autouse fixture
    assert (passwords.SCRYPT_N, passwords.SCRYPT_R, passwords.SCRYPT_P) == (2 ** 15, 8, 3)
    stored = passwords.hash_password("production cost check")
    assert stored.startswith("scrypt$32768$8$3$")
    assert passwords.verify_password("production cost check", stored)
    assert not passwords.needs_rehash(stored)


def test_old_cost_hashes_still_verify_and_are_flagged_for_rehash(monkeypatch):
    stored = passwords.hash_password("migrate me please")  # cheap parameters from the test fixture
    assert not passwords.needs_rehash(stored)  # matches the current (cheap) settings
    monkeypatch.setattr(passwords, "SCRYPT_N", 2 ** 11)  # the "policy" is raised later
    assert passwords.verify_password("migrate me please", stored)  # parameters are read from the hash itself
    assert passwords.needs_rehash(stored)


@pytest.mark.parametrize("bad", ["", "plain", "scrypt$x$y", "scrypt$1$1$1$!!!$!!!", "bcrypt$1$2$3", "$$$$$"])
def test_malformed_hashes_return_false_not_exceptions(bad):
    assert passwords.verify_password("anything", bad) is False


def test_absurdly_long_passwords_are_refused():
    with pytest.raises(ValueError):
        passwords.hash_password("x" * (passwords.MAX_PASSWORD_BYTES + 1))
    assert passwords.verify_password("x" * (passwords.MAX_PASSWORD_BYTES + 1), passwords.hash_password("y" * 12)) is False


def test_pbkdf2_fallback_hashes_verify(monkeypatch):
    monkeypatch.setattr(passwords, "_HAS_SCRYPT", False)
    monkeypatch.setattr(passwords, "PBKDF2_ITERATIONS", 1000)
    stored = passwords.hash_password("fallback password")
    assert stored.startswith("pbkdf2_sha256$1000$")
    assert passwords.verify_password("fallback password", stored)
    assert not passwords.verify_password("other password!", stored)


# ---------------- tokens ----------------
def test_token_roundtrip():
    token, lifetime = create_access_token(7, SECRET, 30)
    assert lifetime == 1800
    data = decode_access_token(token, SECRET)
    assert data is not None and data.user_id == 7


def test_token_rejected_with_wrong_key_expired_tampered_none_alg_and_garbage():
    token, _ = create_access_token(7, SECRET, 30)
    assert decode_access_token(token, "t" * 48) is None
    expired, _ = create_access_token(7, SECRET, -1)
    assert decode_access_token(expired, SECRET) is None
    head, body, sig = token.split(".")
    assert decode_access_token(f"{head}.{body}.{sig[:-2]}AA", SECRET) is None
    unsigned = jwt.encode({"sub": "7", "iat": int(time.time()), "exp": int(time.time()) + 600, "iss": ISSUER}, key="", algorithm="none")
    assert decode_access_token(unsigned, SECRET) is None
    for junk in ("", "abc", "a.b.c", "Bearer x"):
        assert decode_access_token(junk, SECRET) is None


def test_token_with_wrong_issuer_or_missing_claims_is_rejected():
    now = int(time.time())
    wrong_iss = jwt.encode({"sub": "1", "iat": now, "exp": now + 60, "iss": "someone-else"}, SECRET, algorithm=ALGORITHM)
    no_sub = jwt.encode({"iat": now, "exp": now + 60, "iss": ISSUER}, SECRET, algorithm=ALGORITHM)
    non_numeric = jwt.encode({"sub": "abc", "iat": now, "exp": now + 60, "iss": ISSUER}, SECRET, algorithm=ALGORITHM)
    for token in (wrong_iss, no_sub, non_numeric):
        assert decode_access_token(token, SECRET) is None


# ---------------- login limiter ----------------
def test_limiter_blocks_after_max_failures_and_expires():
    now = [0.0]
    limiter = LoginRateLimiter(3, 60, clock=lambda: now[0])
    for _ in range(3):
        assert limiter.retry_after("k") == 0
        limiter.record_failure("k")
    assert limiter.retry_after("k") > 0
    assert limiter.retry_after("other") == 0
    now[0] = 61
    assert limiter.retry_after("k") == 0


def test_limiter_reset_clears_failures():
    limiter = LoginRateLimiter(2, 60)
    limiter.record_failure("k")
    limiter.record_failure("k")
    assert limiter.retry_after("k") > 0
    limiter.reset("k")
    assert limiter.retry_after("k") == 0


# ---------------- settings ----------------
def test_settings_refuse_missing_or_weak_secret():
    with pytest.raises(Exception):
        Settings(_env_file=None)  # no SECRET_KEY at all
    for weak in ("short", "change-me", "x" * (MIN_SECRET_KEY_LENGTH - 1)):
        with pytest.raises(Exception):
            Settings(_env_file=None, secret_key=weak)
    assert Settings(_env_file=None, secret_key="x" * MIN_SECRET_KEY_LENGTH)


def test_settings_resolve_relative_sqlite_path_against_project_root():
    s = Settings(_env_file=None, secret_key="x" * 40, database_url="sqlite:///./data/my.db")
    assert s.resolved_database_url == "sqlite:///" + (PROJECT_ROOT / "data" / "my.db").resolve().as_posix()
    absolute = Settings(_env_file=None, secret_key="x" * 40, database_url="sqlite:///C:/Users/me/x.db")
    assert absolute.resolved_database_url == "sqlite:///C:/Users/me/x.db"


def test_settings_cors_list_ignores_wildcard_and_blanks():
    s = Settings(_env_file=None, secret_key="x" * 40, cors_origins="http://a.test, *, ,http://b.test")
    assert s.cors_origin_list == ["http://a.test", "http://b.test"]


# ---------------- CLI ----------------
def test_cli_generate_secret_prints_a_strong_key(capsys):
    from app import cli

    assert cli.main(["generate-secret"]) == 0
    assert len(capsys.readouterr().out.strip()) >= MIN_SECRET_KEY_LENGTH


def test_cli_write_env_creates_file_and_never_overwrites_without_force(tmp_path, monkeypatch, capsys):
    from app import cli

    env = tmp_path / ".env"
    monkeypatch.setattr(cli, "ENV_PATH", env)
    monkeypatch.setattr(cli, "ENV_EXAMPLE", tmp_path / "missing.example")
    assert cli.main(["generate-secret", "--write-env"]) == 0
    first = env.read_text()
    assert first.startswith("SECRET_KEY=") and len(first.split("=", 1)[1].strip()) >= MIN_SECRET_KEY_LENGTH
    assert cli.main(["generate-secret", "--write-env"]) == 1
    assert env.read_text() == first
    assert cli.main(["generate-secret", "--write-env", "--force"]) == 0
    assert env.read_text() != first


def test_cli_write_env_keeps_other_settings(tmp_path, monkeypatch):
    from app import cli

    env = tmp_path / ".env"
    env.write_text("MAX_APK_SIZE_MB=50\nSECRET_KEY=\n")
    monkeypatch.setattr(cli, "ENV_PATH", env)
    assert cli.main(["generate-secret", "--write-env"]) == 0
    text = env.read_text()
    assert "MAX_APK_SIZE_MB=50" in text and len(text.split("SECRET_KEY=")[1].strip()) >= MIN_SECRET_KEY_LENGTH


def test_cli_init_db_and_check(tmp_path, monkeypatch, capsys):
    from app import cli

    monkeypatch.setenv("SECRET_KEY", "k" * 48)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'cli.db').as_posix()}")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "nomodels"))
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    assert cli.main(["init-db"]) == 0
    assert (tmp_path / "cli.db").exists()
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "Database status:   ok" in out and "NOT_FOUND" in out and "k" * 20 not in out  # secret never printed


def test_cli_write_env_replaces_the_placeholder_key_from_env_example(tmp_path, monkeypatch):
    from app import cli

    env = tmp_path / ".env"
    env.write_text("SECRET_KEY=change-me\nMAX_APK_SIZE_MB=50\n")
    monkeypatch.setattr(cli, "ENV_PATH", env)
    assert cli.main(["generate-secret", "--write-env"]) == 0
    text = env.read_text()
    assert "change-me" not in text and "MAX_APK_SIZE_MB=50" in text


def test_settings_resolve_relative_model_and_upload_dirs_against_project_root():
    s = Settings(_env_file=None, secret_key="x" * 40, model_dir="./models", upload_tmp_dir="data/up")
    assert s.model_dir == (PROJECT_ROOT / "models").resolve()
    assert s.upload_tmp_dir == (PROJECT_ROOT / "data" / "up").resolve()
