"""Context: opt-in Colab-only PostgreSQL setup; importing this module changes nothing.

System installation executes only from its explicit notebook cell. Reinitializing
preserves database records and credentials. The data lives in the temporary runtime.
"""

import json
import os
from pathlib import Path
import secrets
import subprocess
import sys


def run(command, **kwargs):
    """Context: use argument lists, not shell interpolation, for system operations."""
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Colab setup command failed: {command[0]}; inspect runtime package availability.")
    return result.stdout.strip()


def initialize_postgres():
    """Context: install/start local Postgres, then commit extension setup as administrator."""
    if sys.platform != "linux" or not Path("/content").exists():
        raise RuntimeError("This installer is for a disposable Google Colab runtime only.")
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "postgresql", "postgresql-server-dev-all", "build-essential", "git"])
    run(["service", "postgresql", "start"])
    clusters = run(["pg_lsclusters", "--no-header"]).splitlines()
    if not clusters:
        raise RuntimeError("No PostgreSQL cluster was created.")
    version, cluster, port, *_ = clusters[0].split()
    pg_config = f"/usr/lib/postgresql/{version}/bin/pg_config"
    shared = Path(run([pg_config, "--sharedir"]))
    if not (shared / "extension" / "vector.control").exists():
        source = Path("/content/financial_rag_pgvector_src")
        tag = "v0.8.7"
        if not source.exists():
            run(["git", "clone", "--depth", "1", "--branch", tag, "https://github.com/pgvector/pgvector.git", str(source)])
        if run(["git", "-C", str(source), "describe", "--tags", "--exact-match"]) != tag:
            raise RuntimeError("Unexpected pgvector source version; refusing to build it.")
        run(["make", "-C", str(source), f"PG_CONFIG={pg_config}"])
        run(["make", "-C", str(source), "install", f"PG_CONFIG={pg_config}"])
    config_path = Path("/content/financial_rag_database.json")
    settings = json.loads(config_path.read_text()) if config_path.exists() else {
        "user": "financial_rag_demo", "database": "financial_rag_demo", "password": secrets.token_urlsafe(24)}
    config_path.write_text(json.dumps(settings))
    config_path.chmod(0o600)

    def admin(sql, database="postgres"):
        return run(["runuser", "-u", "postgres", "--", "psql", "-p", port, "-d", database,
                    "-v", "ON_ERROR_STOP=1", "-tA"], input=sql)

    # Context: names are fixed, password is generated locally, and literals are
    # escaped before entering stdin. Credentials never appear in command arguments.
    user, database = settings["user"], settings["database"]
    password = settings["password"].replace("'", "''")
    if user != "financial_rag_demo" or database != "financial_rag_demo":
        raise ValueError("Unexpected demo database configuration.")
    if admin("SELECT 1 FROM pg_roles WHERE rolname='financial_rag_demo';") != "1":
        admin(f"CREATE ROLE financial_rag_demo LOGIN PASSWORD '{password}';")
    if admin("SELECT 1 FROM pg_database WHERE datname='financial_rag_demo';") != "1":
        admin("CREATE DATABASE financial_rag_demo OWNER financial_rag_demo;")
    # Context: psql autocommit finishes CREATE EXTENSION before the Python driver
    # registers vector types. Extension ownership remains with the administrator.
    admin("CREATE EXTENSION IF NOT EXISTS vector;", database)
    from psycopg.conninfo import make_conninfo
    os.environ["DATABASE_URL"] = make_conninfo(host="127.0.0.1", port=int(port), dbname=database,
                                             user=user, password=settings["password"])
    return {"database": database, "host": "127.0.0.1", "port": int(port),
            "extension": "vector", "lifecycle": "temporary Colab runtime"}
