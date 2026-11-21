#!/usr/bin/env bash
# Keep the two-node lane up: start it, watch it, and restart it as a pair.
#
# Runs on the host that runs launch_cluster.sh, as the main process of a systemd unit
# (ansible/role-fragment/templates/gb300-pair.service.j2 in the bring-up bundle). The two
# ranks are one engine: a container that comes back alone after a reboot of its box cannot
# rejoin, and the survivor hangs. So nothing here restarts one rank. Every cycle is
#
#   wait until the peer answers ssh and its docker daemon answers   (no time limit)
#   remove the container on both nodes
#   ./launch_cluster.sh                                              (returns at READY, or fails)
#   watch both containers and /health until something is wrong, then start over
#
# A launch that fails is retried with a backoff (60 s doubling to 15 min, reset after a
# lane has stayed up for STABLE_S). launch_cluster.sh exit status 3 means the lane came up
# without its steering hooks: both ranks are removed at once and the lane stays down until
# the next attempt, because serving every request stock while configured to steer is
# worse than not serving. SIGTERM (systemctl stop) removes both ranks and exits 0.
#
# Everything is logged to stdout. Knobs, all optional:
#   PEER_POLL_S=15  WATCH_INTERVAL_S=30  WATCH_FAILS=4  BACKOFF_MIN_S=60  BACKOFF_MAX_S=900
#   STABLE_S=600    HEALTH_TIMEOUT_S=5
# Test hooks: SUPERVISOR_SLEEP and SUPERVISOR_CLOCK (commands used instead of sleep and of
# the shell's clock), SUPERVISOR_MAX_CYCLES. tools/supervisor-dryrun in the bundle uses them.
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
# shellcheck disable=SC1091
source ./config.env

: "${PEER_POLL_S:=15}" "${WATCH_INTERVAL_S:=30}" "${WATCH_FAILS:=4}" "${BACKOFF_MIN_S:=60}"
: "${BACKOFF_MAX_S:=900}" "${STABLE_S:=600}" "${HEALTH_TIMEOUT_S:=5}"
: "${SUPERVISOR_SLEEP:=sleep}" "${SUPERVISOR_CLOCK:=}" "${SUPERVISOR_MAX_CYCLES:=0}"
peer=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_SSH")

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') pair-supervisor: $*"; }
now() { if [[ -n "$SUPERVISOR_CLOCK" ]]; then $SUPERVISOR_CLOCK; else echo "$SECONDS"; fi; }
# Long waits run in the background and are waited for, so a stop request is handled at
# once instead of after the sleep or the launch (which can take an hour) returns.
child=""
hold() { "$@" & child=$!; wait "$child"; local rc=$?; child=""; return "$rc"; }
nap() { hold $SUPERVISOR_SLEEP "$1"; }

peer_ready() { "${peer[@]}" docker info >/dev/null 2>&1; }

remove_both() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  "${peer[@]}" docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

# running | <docker state> | missing | unreachable
local_state() { docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo missing; }
peer_state() {
  local out
  if out="$("${peer[@]}" docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)"; then echo "$out"
  elif peer_ready; then echo missing
  else echo unreachable; fi
}
healthy() { curl -fsS --max-time "$HEALTH_TIMEOUT_S" "$API_URL/health" >/dev/null 2>&1; }

stopping=0
on_term() {
  stopping=1
  [[ -n "$child" ]] && kill "$child" 2>/dev/null
  log "stop requested: removing both ranks"
  remove_both
  exit 0
}
trap on_term TERM INT

backoff="$BACKOFF_MIN_S"
cycle=0
log "supervising $CONTAINER_NAME (peer $RANK1_SSH, health $API_URL/health)"
while :; do
  cycle=$((cycle + 1))
  if (( SUPERVISOR_MAX_CYCLES > 0 && cycle > SUPERVISOR_MAX_CYCLES )); then
    log "cycle limit reached (test hook): leaving the lane as it is"
    exit 0
  fi

  waited=0
  until peer_ready; do
    (( waited == 0 )) && log "peer not ready (ssh or docker): waiting"
    waited=$((waited + 1))
    nap "$PEER_POLL_S"
  done
  (( waited > 0 )) && log "peer ready after $waited poll(s)"

  log "cycle $cycle: removing both ranks, then launching"
  remove_both
  hold ./launch_cluster.sh
  rc=$?
  if (( rc != 0 )); then
    if (( rc == 3 )); then
      log "launch reported READY without steering hooks on a stage (exit 3): removing both ranks, the lane stays down"
    else
      log "launch failed (exit $rc); each rank's full output stays in logs/live-rank<r>-<time>.log on its own node after the containers are removed"
    fi
    remove_both
    log "next attempt in ${backoff}s"
    nap "$backoff"
    backoff=$(( backoff * 2 > BACKOFF_MAX_S ? BACKOFF_MAX_S : backoff * 2 ))
    continue
  fi

  log "READY: watching both ranks every ${WATCH_INTERVAL_S}s"
  up_since="$(now)"
  misses=0
  reason=""
  while [[ -z "$reason" ]]; do
    nap "$WATCH_INTERVAL_S"
    here="$(local_state)"
    there="$(peer_state)"
    if [[ "$here" != running ]]; then
      reason="this node's container is '$here'"
    elif [[ "$there" != running ]]; then
      # One recheck: a single failed ssh must not take a healthy lane down.
      nap "$PEER_POLL_S"
      there="$(peer_state)"
      [[ "$there" != running ]] && reason="the peer's container is '$there'"
    elif healthy; then
      misses=0
    else
      misses=$((misses + 1))
      log "health check missed ($misses of $WATCH_FAILS)"
      (( misses >= WATCH_FAILS )) && reason="$misses consecutive health checks missed"
    fi
  done
  up_for=$(( $(now) - up_since ))
  log "lane lost after ${up_for}s: $reason"
  if (( up_for >= STABLE_S )); then
    backoff="$BACKOFF_MIN_S"
  else
    log "it did not stay up for ${STABLE_S}s: next attempt in ${backoff}s"
    nap "$backoff"
    backoff=$(( backoff * 2 > BACKOFF_MAX_S ? BACKOFF_MAX_S : backoff * 2 ))
  fi
done
