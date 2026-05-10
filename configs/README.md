# Configs

Top-level configuration files hold scientific and operational thresholds that should not be hidden in chat context or hardcoded into pathway logic.

`background_search_v0.json` is the initial config for fixture-only background automation. Every SQLite run ledger entry records its config version and fingerprint so later agents can reproduce which weights and thresholds shaped a run.
