"""
List the functions and classes added, modified or removed on this branch, file by file, from the
git diff against the upstream branch (default origin/develop). Used to keep the "Code changes"
tables of the experiment docs (WeatherGenerator-configs/docs/seaice/xp_0*.md, §3) complete:
the docs give the why, this script gives the what.

    uv run --offline python scripts/code_changes.py --base 5680e9c         # src/ and packages/
    uv run --offline python scripts/code_changes.py --base 5680e9c --all   # + scripts/ and tests/
    uv run --offline python scripts/code_changes.py src/weathergen/model   # against origin/develop

5680e9c is the last commit before the sea-ice work: the earlier changes of the GREP reader are
Pierluigi Cosi's work, on which the branch is built.

A name is "modified" when its source text changed (comments and formatting included);
"module level" covers imports, constants and other code outside functions and classes.
"""

import argparse
import ast
import subprocess
import sys


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def definitions(src):
    """{qualified name: source text} of the functions and classes (methods as Class.method),
    and the module-level text (everything outside the top-level definitions)."""
    if src is None:
        return {}, ""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out, covered = {}, set()

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                text = "".join(lines[start - 1 : child.end_lineno])
                if isinstance(child, ast.ClassDef):
                    # the class itself: its text without the methods (they are listed apart)
                    body = [n for n in child.body if not isinstance(n, ast.FunctionDef)]
                    text = "".join("".join(lines[n.lineno - 1 : n.end_lineno]) for n in body)
                    out[name + " (class body)"] = text
                    visit(child, name + ".")
                else:
                    out[name] = text  # nested functions stay inside their parent
                if prefix == "":
                    covered.update(range(start, child.end_lineno + 1))

    visit(tree, "")
    module = "".join(line for i, line in enumerate(lines, 1) if i not in covered)
    return out, module


def file_changes(base, path):
    old = git("show", f"{base}:{path}") or None
    try:
        with open(path) as f:
            new = f.read()
    except FileNotFoundError:
        new = None
    d_old, m_old = definitions(old)
    d_new, m_new = definitions(new)
    rows = []
    if old is None:
        return [(name, "new") for name in d_new], "new file"
    if new is None:
        return [(name, "removed") for name in d_old], "removed file"
    for name in d_new:
        if name not in d_old:
            rows.append((name, "new"))
        elif d_new[name] != d_old[name]:
            rows.append((name, "modified"))
    rows += [(name, "removed") for name in d_old if name not in d_new]
    if m_new != m_old:
        rows.insert(0, ("module level", "modified"))
    return rows, "modified file"


def out(text):
    sys.stdout.write(text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", default=["src", "packages"], help="paths to look at")
    ap.add_argument("--base", default="origin/develop", help="upstream branch (merge base used)")
    ap.add_argument("--all", action="store_true", help="also scripts/ and tests/")
    a = ap.parse_args()
    paths = a.paths + (["scripts", "tests"] if a.all else [])
    merge_base = git("merge-base", "HEAD", a.base).strip()
    if not merge_base:
        sys.exit(f"no merge base with {a.base} (git fetch?)")
    files = git("diff", "--name-only", f"{merge_base}", "--", *paths).split()
    files = [f for f in files if f.endswith(".py")]
    out(f"# Code changes against {a.base} (merge base {merge_base[:8]})\n\n")
    for path in sorted(files):
        rows, kind = file_changes(merge_base, path)
        out(f"## `{path}` ({kind})\n\n")
        if not rows:
            out("only comments or formatting outside the definitions\n\n")
            continue
        out("| Name | Status |\n|---|---|\n")
        for name, status in rows:
            out(f"| `{name}` | {status} |\n")
        out("\n")


if __name__ == "__main__":
    main()
