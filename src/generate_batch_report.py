from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable
import json

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generate_report(output_root, config_path=None, report_path=None, title="Crack Detection Report"):
    """Generate a lightweight HTML index for batch detection outputs."""
    output_root = Path(output_root)
    config_path = Path(config_path) if config_path else None
    report_path = Path(report_path) if report_path else output_root / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    images = list(_iter_images(output_root))
    config = _load_config(config_path) if config_path else {}
    rows = []
    for img in images:
        rel = img.relative_to(report_path.parent).as_posix()
        rows.append(
            f"<figure><img src='{escape(rel)}' alt='{escape(img.name)}'>"
            f"<figcaption>{escape(str(img.relative_to(output_root)))}</figcaption></figure>"
        )

    cfg_items = "".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
        for k, v in sorted(config.items())
    )
    if not cfg_items:
        cfg_items = "<tr><td colspan='2'>No config file was loaded.</td></tr>"

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #17202a; background: #f8faf7; }}
    header {{ max-width: 960px; margin-bottom: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #57606a; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; padding: 12px; background: white; border: 1px solid #dde5dc; border-radius: 12px; box-shadow: 0 8px 24px rgba(23, 32, 42, 0.06); }}
    img {{ width: 100%; height: auto; border-radius: 8px; background: #eef2ee; }}
    figcaption {{ margin-top: 8px; font-size: 12px; color: #57606a; word-break: break-all; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; margin: 24px 0; background: white; }}
    th, td {{ border: 1px solid #dde5dc; padding: 8px 10px; text-align: left; }}
    th {{ width: 280px; background: #eef4ee; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="meta">Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {len(images)} image files indexed</p>
  </header>
  <h2>Configuration</h2>
  <table>{cfg_items}</table>
  <h2>Output Images</h2>
  <section class="grid">{''.join(rows) if rows else '<p>No output images found.</p>'}</section>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)
