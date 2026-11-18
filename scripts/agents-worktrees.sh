#!/bin/bash
# List every git worktree of a repository with its branch, whether its HEAD
# is an ancestor of origin/main, whether it has uncommitted or untracked
# non-build files, and its disk size. With --remove-merged, remove linked
# worktrees whose HEAD is in origin/main and whose tree is clean of
# everything except ignored or known build output (R6.97).
#
# Usage:
#   scripts/agents-worktrees.sh [--remove-merged] [--dry-run] [--main <ref>] [REPO]
set -euo pipefail

REMOVE_MERGED=0
DRY_RUN=0
MAIN_REF="origin/main"
REPO=""

usage() {
    cat <<'EOF'
usage: scripts/agents-worktrees.sh [--remove-merged] [--dry-run] [--main <ref>] [REPO]

List every git worktree of REPO (default: .) with size, branch, whether HEAD
is in origin/main, and whether it is clean of non-build files.

  --remove-merged  remove linked worktrees that are merged, unlocked, and clean
                   of everything except ignored or known build output
  --dry-run        list and print what --remove-merged would remove, without removing
  --main <ref>     reference commit/branch for merged check (default: origin/main)
  -h, --help       show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --remove-merged)
            REMOVE_MERGED=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --main)
            [ $# -ge 2 ] || { echo "agents-worktrees: --main requires an argument" >&2; exit 2; }
            MAIN_REF="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "agents-worktrees: unknown option '$1'" >&2
            exit 2
            ;;
        *)
            if [ -n "$REPO" ]; then
                echo "agents-worktrees: unexpected argument '$1'" >&2
                exit 2
            fi
            REPO="$1"
            shift
            ;;
    esac
done

if [ $# -gt 0 ]; then
    if [ -n "$REPO" ] || [ $# -gt 1 ]; then
        echo "agents-worktrees: unexpected extra arguments" >&2
        exit 2
    fi
    REPO="$1"
fi

REPO=${REPO:-.}

if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    echo "agents-worktrees: '$REPO' is not a git repository" >&2
    exit 1
fi

if ! git -C "$REPO" rev-parse --verify --quiet "${MAIN_REF}^{commit}" >/dev/null 2>&1; then
    echo "agents-worktrees: reference '$MAIN_REF' not found in '$REPO'" >&2
    exit 1
fi

is_build_artifact() {
    case "$1" in
        .build/*|build/*|dist/*|__pycache__/*|*/__pycache__/*|.pytest_cache/*|*/.pytest_cache/*|node_modules/*|*/node_modules/*|*.dSYM/*)
            return 0
            ;;
        *.o|*.lo|*.a|*.so|*.dylib|*.dll|*.lib|*.exe|*.obj|*.pdb|*.exp|*.ilk|*.pyc|*.pyo|*.out|*.err|*.profraw|*.profdata|*.gcda|*.gcno|*.gcov)
            return 0
            ;;
        runner|runner.exe|runner-debug|runner-debug.exe|difftok|difftok.exe)
            return 0
            ;;
        runner-*|test-*)
            case "$1" in
                */*) return 1 ;;
                *.c|*.h|*.m|*.mm|*.cu|*.cuh|*.cpp|*.cc|*.hpp|*.py|*.sh|*.md|*.txt|*.json|*.yml|*.yaml|*.toml)
                    return 1
                    ;;
                *) return 0 ;;
            esac
            ;;
    esac
    return 1
}

format_kib() {
    LC_ALL=C awk -v k="$1" 'BEGIN {
        if (k >= 1048576) { printf "%.1fG\n", k / 1048576.0 }
        else if (k >= 10240) { printf "%.0fM\n", k / 1024.0 }
        else if (k >= 1024) { printf "%.1fM\n", k / 1024.0 }
        else { printf "%dK\n", k }
    }'
}

HERE_REAL=$(pwd -P)
REMOVED_PATHS=""

printf "%-8s %-7s %-8s %-32s %s\n" "SIZE" "IN_MAIN" "STATUS" "BRANCH" "PATH"

process_entry() {
    local wt="$1" head="$2" branch="$3" is_bare="$4" is_locked="$5"
    [ -n "$wt" ] || return 0

    local role="linked"
    if [ "$is_bare" -eq 1 ] || [ -d "$wt/.git" ]; then
        role="main"
    fi

    local in_main="no"
    local status="clean"
    local size="-"
    local has_unignored_build=0

    if [ "$is_bare" -eq 1 ]; then
        branch="(bare)"
    elif [ -z "$branch" ]; then
        branch="(detached)"
    fi

    if [ ! -d "$wt" ]; then
        status="missing"
    else
        local kib
        kib=$(du -sk "$wt" | awk '{print $1}')
        size=$(format_kib "$kib")

        if [ "$is_bare" -eq 0 ] && [ -n "$head" ] && ! echo "$head" | grep -q '^0000000'; then
            if git -C "$wt" merge-base --is-ancestor "$head" "$MAIN_REF" >/dev/null 2>&1; then
                in_main="yes"
            fi
        fi

        if [ "$is_bare" -eq 0 ]; then
            local entry xy path
            while IFS= read -r -d '' entry; do
                xy=${entry:0:2}
                path=${entry:3}
                if [ "$xy" != "??" ]; then
                    status="dirty"
                    break
                fi
                if is_build_artifact "$path"; then
                    has_unignored_build=1
                else
                    status="dirty"
                    break
                fi
            done < <(git -C "$wt" status --porcelain=v1 -z -uall)
        fi
    fi

    if [ "$is_locked" -eq 1 ] && [ "$status" = "clean" ]; then
        status="locked"
    fi

    printf "%-8s %-7s %-8s %-32s %s\n" "$size" "$in_main" "$status" "$branch" "$wt"

    if { [ "$REMOVE_MERGED" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; } && [ "$role" = "linked" ] && [ "$is_locked" -eq 0 ] && [ "$in_main" = "yes" ] && [ "$status" = "clean" ]; then
        local wt_real
        wt_real=$(cd "$wt" 2>/dev/null && pwd -P) || wt_real="$wt"
        case "$HERE_REAL/" in
            "$wt_real/"*)
                echo "skip: $wt (current working directory is inside this worktree)" >&2
                return 0
                ;;
        esac
        if [ "$DRY_RUN" -eq 1 ]; then
            REMOVED_PATHS="${REMOVED_PATHS}${wt}"$'\n'
            return 0
        fi
        if [ "$has_unignored_build" -eq 1 ]; then
            git -C "$REPO" worktree remove --force "$wt"
        else
            git -C "$REPO" worktree remove "$wt"
        fi
        REMOVED_PATHS="${REMOVED_PATHS}${wt}"$'\n'
    fi
}

wt=""
head=""
branch=""
is_bare=0
is_locked=0

while IFS= read -r line || [ -n "$wt" ]; do
    if [ -z "$line" ]; then
        process_entry "$wt" "$head" "$branch" "$is_bare" "$is_locked"
        wt=""
        head=""
        branch=""
        is_bare=0
        is_locked=0
        continue
    fi
    case "$line" in
        "worktree "*)
            wt=${line#worktree }
            ;;
        "HEAD "*)
            head=${line#HEAD }
            ;;
        "branch refs/heads/"*)
            branch=${line#branch refs/heads/}
            ;;
        "branch "*)
            branch=${line#branch }
            ;;
        "detached")
            branch="(detached)"
            ;;
        "bare")
            is_bare=1
            ;;
        "locked"|"locked "*)
            is_locked=1
            ;;
    esac
done < <(git -C "$REPO" worktree list --porcelain)

if [ -n "$REMOVED_PATHS" ]; then
    label="removed"
    [ "$DRY_RUN" -eq 1 ] && label="would remove"
    printf "%s" "$REMOVED_PATHS" | while IFS= read -r p; do
        [ -n "$p" ] && echo "${label}: $p"
    done
fi
