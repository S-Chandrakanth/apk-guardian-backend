"""Command-line helpers.

    python -m app.cli generate-secret --write-env   create/update .env with a strong SECRET_KEY
    python -m app.cli init-db                       create the SQLite tables (safe to repeat)
    python -m app.cli check                         show configuration, database, model, and engine status

Run these from the backend folder.
"""

from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path

from pydantic import ValidationError

from app.config import MIN_SECRET_KEY_LENGTH, PROJECT_ROOT, Settings

ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
_SECRET_LINE = re.compile(r"^\s*SECRET_KEY\s*=\s*(.*)$", re.MULTILINE)


def _load_settings() -> Settings | None:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e.get("loc"))
        print(f"Configuration problem with: {fields}")
        print("Create a strong SECRET_KEY with:  python -m app.cli generate-secret --write-env")
        return None


def cmd_generate_secret(write_env: bool, force: bool) -> int:
    key = secrets.token_urlsafe(48)
    if not write_env:
        print(key)
        return 0
    if ENV_PATH.exists():
        text = ENV_PATH.read_text(encoding="utf-8")
        match = _SECRET_LINE.search(text)
        current = match.group(1).strip().strip('"\'') if match else ""
        has_real_key = len(current) >= MIN_SECRET_KEY_LENGTH and current.lower() != "change-me"
        if match and has_real_key and not force:
            print(f"{ENV_PATH} already has a SECRET_KEY. Nothing changed (use --force to replace it;")
            print("replacing it logs every user out).")
            return 1
        text = _SECRET_LINE.sub(f"SECRET_KEY={key}", text, count=1) if match else text.rstrip("\n") + f"\nSECRET_KEY={key}\n"
    else:
        base = ENV_EXAMPLE.read_text(encoding="utf-8") if ENV_EXAMPLE.exists() else ""
        text = _SECRET_LINE.sub(f"SECRET_KEY={key}", base, count=1) if _SECRET_LINE.search(base) else base + f"SECRET_KEY={key}\n"
    ENV_PATH.write_text(text, encoding="utf-8")
    print(f"SECRET_KEY written to {ENV_PATH}. Keep this file private; it is git-ignored.")
    return 0


def cmd_init_db() -> int:
    settings = _load_settings()
    if settings is None:
        return 2
    from app.database.session import database_status, init_db, make_engine, seed_default_users

    engine = make_engine(settings)
    init_db(engine)
    seed_default_users(engine)
    print(f"Database ready ({database_status(engine)}): {settings.sqlite_path or 'configured database'}")
    return 0


def cmd_check() -> int:
    settings = _load_settings()
    if settings is None:
        return 2
    from app.database.session import database_status, make_engine
    from app.services.apk_features import ExtractorEnvironmentError, load_androguard
    from app.services.model_service import ModelService

    print(f"Database file:     {settings.sqlite_path or settings.resolved_database_url}")
    print(f"Database status:   {database_status(make_engine(settings))}")
    try:
        print(f"Androguard:        {load_androguard()[2]}")
    except ExtractorEnvironmentError as exc:
        print(f"Androguard:        UNAVAILABLE ({exc})")
    status = ModelService(settings.model_dir, settings.allow_synthetic_model).status()
    print(f"Model status:      {status.state} - {status.detail}")
    print(f"Upload limit:      {settings.max_apk_size_mb} MB, analysis timeout {settings.analysis_timeout_seconds:g} s")
    print(f"CORS origins:      {', '.join(settings.cors_origin_list) or '(none)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-secret", help="print a new strong SECRET_KEY, or write it to .env")
    gen.add_argument("--write-env", action="store_true")
    gen.add_argument("--force", action="store_true", help="replace an existing SECRET_KEY")
    sub.add_parser("init-db", help="create the database tables")
    sub.add_parser("check", help="show configuration and component status")
    args = parser.parse_args(argv)

    if args.command == "generate-secret":
        return cmd_generate_secret(args.write_env, args.force)
    if args.command == "init-db":
        return cmd_init_db()
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
