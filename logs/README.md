# Logs

This top-level directory is the default runtime location for background search logs.

The background automation stores durable run state in SQLite, with the default database path:

```text
logs/background_search.sqlite3
```

Generated SQLite databases are runtime artifacts and should not be committed. Tests should create temporary SQLite databases under test-managed temporary directories.
