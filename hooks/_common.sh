#!/bin/sh
# hooks/_common.sh — shared helpers for TENKA's git hooks.
#
# Sourced, never executed directly. Keep POSIX sh: Git for Windows runs hooks
# through its bundled sh, not bash.

# ─── Output ──────────────────────────────────────────────────────────────────

if [ -t 2 ]; then
    C_RED=$(printf '\033[31m'); C_YEL=$(printf '\033[33m')
    C_DIM=$(printf '\033[2m');  C_OFF=$(printf '\033[0m')
else
    C_RED=''; C_YEL=''; C_DIM=''; C_OFF=''
fi

die() {
    printf '%s\n' "${C_RED}BLOCKED${C_OFF} $1" >&2
    shift
    for line in "$@"; do printf '  %s\n' "$line" >&2; done
    printf '\n  %sSee CLAUDE.md > Git workflow. Do not bypass with --no-verify.%s\n' \
        "$C_DIM" "$C_OFF" >&2
    exit 1
}

warn() { printf '%s\n' "${C_YEL}warning${C_OFF} $*" >&2; }

# ─── Python resolution ───────────────────────────────────────────────────────

# Bare `python` / `python3` on Windows are often the Microsoft Store stub: they
# print an install advert and exit non-zero. Probe for a real interpreter and
# echo the working command, or echo nothing if none is usable.
find_python() {
    for _cmd in "py -3.11" "py -3" "python3" "python"; do
        if $_cmd -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" \
             >/dev/null 2>&1; then
            printf '%s' "$_cmd"
            return 0
        fi
    done
    return 1
}
