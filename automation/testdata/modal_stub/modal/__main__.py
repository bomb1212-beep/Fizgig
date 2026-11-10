"""`python -m modal ...` for the fake SDK: only `deploy` is supported; it records nothing but exits 0."""
import sys

args = sys.argv[1:]
if args and args[0] == "deploy":
    target = args[1] if len(args) > 1 else "?"
    print(f"[fake modal] deployed {target}")
    sys.exit(0)
if args and args[0] == "token":
    print("[fake modal] token ok")
    sys.exit(0)
print(f"[fake modal] unsupported command: {' '.join(args)}", file=sys.stderr)
sys.exit(2)
