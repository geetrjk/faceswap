#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.setdefault("FACE_SWAP_SHARED_ROOT", str(repo_root))
    host = os.environ.get("DEPLOY_APP_HOST", "0.0.0.0")
    port = int(os.environ.get("DEPLOY_APP_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
