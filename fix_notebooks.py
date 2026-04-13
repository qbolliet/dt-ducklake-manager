"""Script pour corriger les lignes trop longues dans les notebooks Jupyter."""

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def get_e501_errors():
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select", "E501", "--output-format=json"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent)
    )
    return json.loads(result.stdout)


def wrap_comment(line: str, max_len: int = 88) -> list:
    """Coupe un commentaire trop long en plusieurs lignes."""
    stripped = line.rstrip()
    indent = len(stripped) - len(stripped.lstrip())
    indent_str = stripped[:indent]
    content = stripped[indent:]

    if content.startswith("# "):
        prefix = "# "
        text = content[2:]
    elif content.startswith("#"):
        prefix = "#"
        text = content[1:]
    else:
        return [line]

    full_prefix = indent_str + prefix
    available = max_len - len(full_prefix)

    if available <= 0:
        return [line]

    words = text.split(" ")
    lines = []
    current = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current else 0)
        if current_len + word_len > available and current:
            lines.append(full_prefix + " ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len

    if current:
        lines.append(full_prefix + " ".join(current))

    return lines


def find_brace_safe_split(content: str, max_chars: int) -> int:
    """Trouve la dernière position de coupe sûre dans une f-string."""
    depth = 0
    last_safe = -1
    i = 0
    while i < len(content) and i < max_chars:
        c = content[i]
        if c == '{':
            if i + 1 < len(content) and content[i+1] == '{':
                i += 2
                continue
            depth += 1
        elif c == '}':
            if i + 1 < len(content) and content[i+1] == '}':
                i += 2
                continue
            depth -= 1
            if depth == 0 and i + 1 <= max_chars:
                last_safe = i + 1
        elif c == ' ' and depth == 0:
            last_safe = i + 1
        i += 1
    return last_safe


def fix_cell_line(line: str, max_len: int = 88) -> list:
    """Tente de corriger une ligne trop longue dans un notebook."""
    stripped = line.rstrip('\n')
    if len(stripped) <= max_len:
        return [line]

    lstripped = stripped.lstrip()

    # Cas 1: commentaire
    if lstripped.startswith('#'):
        result = wrap_comment(stripped)
        if result and len(result) > 1:
            return [l + '\n' for l in result]

    # Cas 2: chaîne littérale standalone
    if re.match(r'^[fFrRbBuU]*([\"\'])', lstripped):
        indent_len = len(stripped) - len(lstripped)
        indent = stripped[:indent_len]
        prefix_m = re.match(r'^([fFrRbBuU]*)([\"\'])', lstripped)
        if prefix_m:
            prefix = prefix_m.group(1)
            quote = prefix_m.group(2)
            if not lstripped[len(prefix):].startswith(quote * 3):
                str_start = len(prefix) + len(quote)
                inner = lstripped[str_start:]
                close_idx = None
                for i in range(len(inner) - 1, -1, -1):
                    if inner[i] == quote and (i == 0 or inner[i-1] != '\\'):
                        close_idx = i
                        break
                if close_idx is not None:
                    content = inner[:close_idx]
                    trailing = inner[close_idx+1:]
                    overhead = indent_len + len(prefix) + 2 * len(quote)
                    available = max_len - overhead
                    split_pos = find_brace_safe_split(content, available)
                    if split_pos > 0 and split_pos < len(content):
                        part1 = content[:split_pos].rstrip()
                        part2 = content[split_pos:].lstrip() if content[split_pos] == ' ' else content[split_pos:]
                        if part2:
                            l1 = f"{indent}{prefix}{quote}{part1}{quote}\n"
                            l2 = f"{indent}{prefix}{quote} {part2}{quote}{trailing}\n"
                            if len(l1.rstrip()) <= max_len and len(l2.rstrip()) <= max_len:
                                return [l1, l2]

    return [line]


def fix_notebook(filepath: Path, errors_by_cell: dict) -> int:
    """Corrige les lignes trop longues dans un notebook."""
    with open(filepath, encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    fixes = 0

    for cell_idx, line_nums in errors_by_cell.items():
        # Les indices de cellules dans ruff sont 1-based
        cell_pos = cell_idx - 1
        if cell_pos >= len(cells):
            continue

        cell = cells[cell_pos]
        source = cell.get('source', [])
        if isinstance(source, str):
            source = source.splitlines(keepends=True)
        else:
            source = list(source)

        # Traitement en ordre inverse
        for line_num in sorted(set(line_nums), reverse=True):
            idx = line_num - 1
            if idx >= len(source):
                continue
            line = source[idx]
            if not line.endswith('\n'):
                line = line + '\n'
            result = fix_cell_line(line)
            if len(result) > 1:
                source[idx:idx+1] = result
                fixes += 1

        cell['source'] = source

    if fixes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

    return fixes


def main():
    errors = get_e501_errors()

    # Regroupement par notebook et cellule
    by_notebook = defaultdict(lambda: defaultdict(list))
    for err in errors:
        if err['filename'].endswith('.ipynb') and err.get('cell') is not None:
            by_notebook[err['filename']][err['cell']].append(err['location']['row'])

    # Correspondance entre les chemins Unicode de ruff et les chemins réels du filesystem
    import os
    nb_dir = Path('notebooks')
    real_paths = {
        str(nb_dir / f).replace('/', os.sep): nb_dir / f
        for f in os.listdir(nb_dir)
        if f.endswith('.ipynb')
    }

    total = 0
    for filepath, by_cell in sorted(by_notebook.items()):
        # Tentative de correspondance via le nom de fichier
        real_path = None
        for real_str, real_p in real_paths.items():
            if real_p.exists() and real_p.suffix == '.ipynb':
                # Comparaison approximative par index (ordre alphabétique)
                real_path = real_p
                break

        # Approche directe: trouver le bon fichier par numéro dans le nom
        fname_ruff = Path(filepath).name
        nb_num = fname_ruff[0] if fname_ruff[0].isdigit() else ''
        if nb_num:
            for real_str, real_p in real_paths.items():
                if Path(real_str).name.startswith(nb_num):
                    real_path = real_p
                    break

        if real_path is None or not real_path.exists():
            print(f"NOT FOUND: {filepath}")
            continue

        fixes = fix_notebook(real_path, by_cell)
        if fixes:
            print(f"FIXED ({fixes}): {real_path.name}")
            total += fixes
        else:
            print(f"UNCHANGED: {real_path.name}")

    print(f"\nTotal: {total}")


if __name__ == '__main__':
    main()
