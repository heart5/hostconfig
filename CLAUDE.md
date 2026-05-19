# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Host configuration collection and comparison tool. Collects system info, Python environment, library versions, and project configuration from hosts (Linux/Windows/macOS), saves to local JSON, and syncs comparison tables to Joplin notes via the Joplin Web Clipper API.

## Commands

- **Run**: `python hostconfig.py` (from project root)
- **Run in Jupyter**: Open `hostconfig.ipynb` and run all cells (`.py` ↔ `.ipynb` synced via jupytext)

No formal build system, linter, or test suite exists in this repo.

## Git setup

The `func/` directory is a git submodule. Clone with:

```bash
git clone --recurse-submodules <repo-url>
# or for an already-cloned repo:
git submodule update --init --recursive
```

## Architecture

### Project structure

```
hostconfig/
├── hostconfig.py              # Thin entry point (~55 lines)
├── pathmagic.py               # sys.path extension context manager
├── hcm/                       # Functional modules
│   ├── imports.py             # Centralized func imports (no jpfuncs)
│   ├── models.py              # Pure data classes (ConfigSnapshot, UpdateRecord, LibsConfig)
│   ├── collector.py           # HostConfigCollector
│   ├── storage.py             # LocalStorage: JSON persist + merge + smart save
│   ├── markdown.py            # Markdown parse/generate (pure functions)
│   ├── joplin_sync.py         # JoplinClient + update_joplin_note
│   ├── sync_service.py        # SyncService: orchestrator
│   └── utils.py               # Helpers + jpfuncs import guard
├── func/                      # Git submodule: utility library
│   ├── first.py               # Project root discovery (finds `rootfile`)
│   ├── getid.py               # Device ID/name/user
│   ├── jpfuncs.py             # Joplin API (module-level: jpapi = getapi())
│   ├── configpr.py            # INI config reader
│   ├── logme.py               # Unified logging
│   ├── sysfunc.py             # Shell command execution, IPython detection
│   └── wrapfuncs.py           # @timethis decorator
├── data/                      # Config files + host snapshots
├── docs/                      # Documentation
└── rootfile                   # Project root marker
```

### Key classes

| Class | Module | Role |
|:---|:---|:---|
| `HostConfigCollector` | hcm/collector.py | Collects system/Python/library/project info → ConfigSnapshot |
| `LocalStorage` | hcm/storage.py | JSON read/write, merge, smart save, deep comparison |
| `JoplinClient` | hcm/joplin_sync.py | Joplin Web Clipper API thin wrapper |
| `SyncService` | hcm/sync_service.py | Orchestrator: collect → compare → save → sync |

### Key data classes (hcm/models.py)

- `ConfigSnapshot` — `system / python / libraries / project / collection_time` + `to_dict()` / `from_dict()`
- `UpdateRecord` — `timestamp / device_id / device_name / has_changes / summary`
- `LibsConfig` — `required_libs / optional_libs / ai_libs`

### Main flow (SyncService.run)

1. `collector.collect_all()` → ConfigSnapshot
2. `_compare_with_previous()` — compare against `data/hostconfig/{device_id}.json`
3. If changed: `storage.save(snapshot)`
4. `update_joplin_note()` — read existing Joplin configs, merge, push

### Import pattern

- `pathmagic.py` context manager temporarily adds `.` to `sys.path`
- `hcm/imports.py` centralizes func imports (NO jpfuncs)
- `func/jpfuncs.py` 使用 `_LazyJoplinAPI` 惰性代理推迟 `getapi()` 到首次调用，导入期安全。`getapi()` 失败抛 `JoplinUnreachableError`
- `hcm/` 各模块直接 `from func.jpfuncs import ...` 按需导入，不设 mock/guard 中间层。网络错误由调用方的 try/except 统一处理

### Configuration files

All in `data/`:
- `happyjp.ini` — Joplin API cloud INI ID
- `happyjphard.ini` — Hardware params (device_id → device_name)
- `happyjpinifromcloud.ini` — Device name → device_id mapping (synced from cloud)
- `happyjpsys.ini` — System flags (e.g., `FORCE_UPDATE=true`)
- `joplinai.ini` — Remote Joplin config (optional, for remote-first mode)

The file `rootfile` at repo root is an empty marker used by `func/first.py` to discover the project root.

### Jupytext

`.py` and `.ipynb` files synced via jupytext (configured in `jupytext.toml`). Format: `percent` (cells separated by `# %%`).
