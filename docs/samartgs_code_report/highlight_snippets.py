#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, JsonLexer, PythonLexer, TextLexer, YamlLexer


REPORT_ROOT = Path(__file__).resolve().parent
SNIPPET_DIR = REPORT_ROOT / "assets" / "snippets"
OUT_DIR = REPORT_ROOT / "assets" / "highlighted_snippets"
LINE_RE = re.compile(r"^\s*(\d+)\s\|\s?(.*)$")


EXTRA_METADATA = {
    "branch_dispatch": {
        "language": "python",
        "highlight_lines": [1158, 1163, 1188, 1189, 1190, 1191],
    },
    "direct_trainable": {
        "language": "python",
        "highlight_lines": [394, 395],
    },
    "direct_forward_grad": {
        "language": "python",
        "highlight_lines": [415, 416, 425, 431, 452, 453, 459, 460],
    },
    "direct_commit_short": {
        "language": "python",
        "highlight_lines": [507, 511, 521, 543],
    },
    "mlp_model_short": {
        "language": "python",
        "highlight_lines": [6, 46, 49],
    },
    "mlp_initialize_short": {
        "language": "python",
        "highlight_lines": [1036, 1037, 1046, 1047],
    },
    "mlp_forward_regularized": {
        "language": "python",
        "highlight_lines": [652, 653, 654, 655, 675, 682, 701, 702, 707],
    },
    "mlp_commit_short": {
        "language": "python",
        "highlight_lines": [694, 770, 771, 772, 773, 829, 831],
    },
    "shared_outputs_short": {
        "language": "python",
        "highlight_lines": [1212, 1239, 1247, 1248, 1249, 1271, 1337, 1443, 1444, 1448],
    },
}


def load_index() -> Dict[str, Dict[str, object]]:
    index_path = SNIPPET_DIR / "snippets_index.json"
    if not index_path.exists():
        return {}
    items = json.loads(index_path.read_text())
    return {item["slug"]: item for item in items}


def lexer_for(language: str):
    if language == "python":
        return PythonLexer()
    if language in {"yaml", "yml"}:
        return YamlLexer()
    if language == "json":
        return JsonLexer()
    if language in {"bash", "shell", "sh"}:
        return BashLexer()
    return TextLexer()


def parse_numbered_snippet(text: str) -> Tuple[str, int, Dict[int, int]]:
    code_lines = []  # type: List[str]
    line_map = {}  # type: Dict[int, int]
    first_line = 1
    for idx, line in enumerate(text.splitlines(), start=1):
        match = LINE_RE.match(line)
        if match:
            original_line = int(match.group(1))
            if not line_map:
                first_line = original_line
            line_map[original_line] = idx
            code_lines.append(match.group(2))
        else:
            code_lines.append(line)
    return "\n".join(code_lines) + "\n", first_line, line_map


def relative_highlights(original_highlights: List[int], line_map: Dict[int, int]) -> List[int]:
    return [line_map[line] for line in original_highlights if line in line_map]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_index()
    generated = []  # type: List[Dict[str, object]]

    for path in sorted(SNIPPET_DIR.glob("*.txt")):
        slug = path.stem
        info = {**metadata.get(slug, {}), **EXTRA_METADATA.get(slug, {})}
        language = str(info.get("language", "text"))
        raw = path.read_text()
        code, start_line, line_map = parse_numbered_snippet(raw)
        hl_lines = relative_highlights([int(line) for line in info.get("highlight_lines", [])], line_map)
        formatter = HtmlFormatter(
            style="friendly",
            linenos="table",
            linenostart=start_line,
            cssclass="codehilite",
            wrapcode=True,
            hl_lines=hl_lines,
        )
        html = highlight(code, lexer_for(language), formatter)
        (OUT_DIR / f"{slug}.html").write_text(html)
        generated.append(
            {
                "slug": slug,
                "language": language,
                "source": str(path.relative_to(REPORT_ROOT)),
                "output": f"assets/highlighted_snippets/{slug}.html",
                "line_start": start_line,
                "highlight_lines": info.get("highlight_lines", []),
            }
        )

    css_formatter = HtmlFormatter(style="friendly", cssclass="codehilite")
    css = css_formatter.get_style_defs(".codehilite")
    css += "\n.codehilite .hll { background-color: #fff7cc; display: block; }\n"
    (OUT_DIR / "pygments.css").write_text(css)
    (OUT_DIR / "highlighted_index.json").write_text(json.dumps(generated, indent=2))
    print(f"Generati {len(generated)} snippet evidenziati in {OUT_DIR}")


if __name__ == "__main__":
    main()
