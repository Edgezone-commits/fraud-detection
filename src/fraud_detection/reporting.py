"""Write results to reports/metrics.json.

Each script updates only its own sections, so the experiment results and the final
test result can live in the same file without overwriting each other.
"""

import json

from fraud_detection.config import REPORTS_DIR


def read_report():
    """Return the current contents of reports/metrics.json, or {} if there is none."""
    path = REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_report(sections):
    """Merge `sections` (a dict of top-level keys) into reports/metrics.json."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "metrics.json"
    current = read_report()
    current.update(sections)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return path
