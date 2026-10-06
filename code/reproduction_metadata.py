"""Portable provenance for current-manuscript calculations."""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(sources, settings):
    paths = [Path(p).resolve() for p in sources]
    paths.append(Path(__file__).resolve())
    config = ROOT / 'configs/current_manuscript_checks.json'
    return {
        'schema_version': 1,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'platform': platform.system()},
        'settings': settings,
        'config_sha256': sha256(config),
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
    }


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def settings():
    return json.loads((ROOT / 'configs/current_manuscript_checks.json').read_text(encoding='utf-8'))
