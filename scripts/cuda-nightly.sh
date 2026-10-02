#!/bin/bash
# The nightly CUDA job for a lab box that hosted CI cannot reach (R6.9):
# pull main, build, run scripts/cuda-gate.sh, and make a red night visible.
#
#   scripts/cuda-nightly.sh BASE_DIR
#
# BASE_DIR holds everything the job writes (history.log, logs/, out/) and two
# files a person puts there once:
#   cuda-nightly.env  VAR=value lines for this script (below)
#   cuda-gate.env     VAR=value lines for scripts/cuda-gate.sh
# The checkout it updates and builds is the one this script sits in, or the
# one NIGHTLY_SRC names. Run a COPY of the script with NIGHTLY_SRC set: the
# update replaces the file under a shell that is still reading it otherwise.
#   cp <checkout>/scripts/cuda-nightly.sh BASE/run.sh && bash BASE/run.sh BASE
#
# It is a guest on a shared machine. Before it touches the CPU it checks the
# conditions the box's other users named, and when one holds it logs a skip
# and exits 0. A skipped night is not a green night: history.log says which.
#
#   NIGHTLY_SKIP_LOCKS  space-separated lock files; a lock whose first number
#                       is a live pid means somebody else is running
#   NIGHTLY_SKIP_IF     a shell command; exit 0 means skip (e.g. a container
#                       name filter)
#   NIGHTLY_LOCK        the lock this job holds while it runs (its own pid,
#                       the Windows pid under MSYS2)
#   NIGHTLY_MAKE_ARGS   make arguments for the build (e.g. OS=Windows_NT -j4)
#   NIGHTLY_REMOTE      git remote and branch to build (default origin main)
#   NIGHTLY_SRC         the checkout to update and build
#   NIGHTLY_ISSUE_REPO  owner/name; when set and the GitHub CLI is logged in,
#                       a failed night opens one issue there, or comments on
#                       the open one, so a person sees it. Unset: local only.
set -u
BASE=${1:?usage: cuda-nightly.sh BASE_DIR}
mkdir -p "$BASE/logs" "$BASE/out" || exit 2
# shellcheck disable=SC1091
[ -f "$BASE/cuda-nightly.env" ] && { set -a; . "$BASE/cuda-nightly.env"; set +a; }
SRC=${NIGHTLY_SRC:-$(cd "$(dirname "$0")/.." && pwd)}
log() { echo "$(date '+%F %T') $*" >> "$BASE/history.log"; }

pid_alive() {
  if command -v tasklist > /dev/null 2>&1; then
    tasklist //FI "PID eq $1" 2> /dev/null | grep -q "[[:space:]]$1[[:space:]]"
  else
    kill -0 "$1" 2> /dev/null
  fi
}
lock_held() {  # file -> 0 when it names a live pid
  [ -f "$1" ] || return 1
  local p
  p=$(grep -o '[0-9][0-9]*' "$1" 2> /dev/null | head -n 1)
  [ -n "$p" ] && pid_alive "$p"
}

for l in ${NIGHTLY_SKIP_LOCKS:-}; do
  if lock_held "$l"; then log "skip: $l is held"; exit 0; fi
done
if [ -n "${NIGHTLY_SKIP_IF:-}" ] && bash -c "$NIGHTLY_SKIP_IF" > /dev/null 2>&1; then
  log "skip: NIGHTLY_SKIP_IF held"; exit 0
fi
if [ -n "${NIGHTLY_LOCK:-}" ]; then
  if lock_held "$NIGHTLY_LOCK"; then log "skip: $NIGHTLY_LOCK is held"; exit 0; fi
  if [ -r "/proc/$$/winpid" ]; then cat "/proc/$$/winpid" > "$NIGHTLY_LOCK"
  else echo $$ > "$NIGHTLY_LOCK"; fi
  trap 'rm -f "$NIGHTLY_LOCK"' EXIT
fi

report_failure() {  # what logfile
  log "FAIL: $1 (see $2)"
  [ -n "${NIGHTLY_ISSUE_REPO:-}" ] || return 0
  command -v gh > /dev/null 2>&1 || { log "no gh: the failure is recorded here only"; return 0; }
  local title="CUDA nightly is failing on $(hostname)"
  # The public report names the commit and the step, nothing from the logs:
  # a log tail carries paths and model names nobody reviewed for publication.
  local body
  body=$(printf 'The scheduled CUDA gate failed.\n\n- when: %s\n- commit: %s\n- what: %s\n\nThe logs are on the box, under the nightly directory.\n' \
    "$(date -u '+%F %T UTC')" "$(git -C "$SRC" rev-parse --short HEAD 2> /dev/null)" "$1")
  local open
  open=$(gh issue list -R "$NIGHTLY_ISSUE_REPO" --state open --search "\"$title\" in:title" \
           --json number --jq '.[0].number' 2> /dev/null)
  if [ -n "$open" ]; then
    gh issue comment "$open" -R "$NIGHTLY_ISSUE_REPO" --body "$body" > /dev/null 2>&1 \
      && log "commented on issue #$open"
  else
    gh issue create -R "$NIGHTLY_ISSUE_REPO" --title "$title" --body "$body" > /dev/null 2>&1 \
      && log "opened an issue"
  fi
}

cd "$SRC" || exit 2
# shellcheck disable=SC2086
set -- ${NIGHTLY_REMOTE:-origin main}
if ! git fetch -q "$1" "$2" > "$BASE/logs/fetch.log" 2>&1 \
   || ! git checkout -q --detach FETCH_HEAD >> "$BASE/logs/fetch.log" 2>&1; then
  report_failure "could not update the checkout" "$BASE/logs/fetch.log"; exit 1
fi
COMMIT=$(git rev-parse --short HEAD)
# shellcheck disable=SC2086
if ! make ${NIGHTLY_MAKE_ARGS:-} runner > "$BASE/logs/build.log" 2>&1; then
  report_failure "the build failed at $COMMIT" "$BASE/logs/build.log"; exit 1
fi
GATE_ENV="$BASE/cuda-gate.env" bash scripts/cuda-gate.sh . "$BASE/out" > "$BASE/logs/gate.log" 2>&1
rc=$?
case $rc in
  0) log "pass at $COMMIT" ;;
  *) report_failure "cuda-gate.sh exit $rc at $COMMIT" "$BASE/logs/gate.log" ;;
esac
exit $rc
