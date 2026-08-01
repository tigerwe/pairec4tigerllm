#!/usr/bin/env bash
set -euo pipefail

LOCAL_INTERFACE="${LOCAL_INTERFACE:-enp41s0f1}"
LOCAL_CIDR="${LOCAL_CIDR:-192.168.100.12/24}"
REMOTE_HOST="${REMOTE_HOST:-root@141.61.91.188}"
REMOTE_INTERFACE="${REMOTE_INTERFACE:-enp41s0f1}"
REMOTE_CIDR="${REMOTE_CIDR:-192.168.100.11/24}"
APPLY="${APPLY:-0}"
PING_COUNT="${PING_COUNT:-3}"

log() {
  printf '\n== %s ==\n' "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

case "$APPLY" in
  0|1) ;;
  *) die "APPLY must be 0 or 1" ;;
esac
[[ "$LOCAL_INTERFACE" =~ ^[a-zA-Z0-9_.:-]+$ ]] \
  || die "invalid LOCAL_INTERFACE"
[[ "$REMOTE_INTERFACE" =~ ^[a-zA-Z0-9_.:-]+$ ]] \
  || die "invalid REMOTE_INTERFACE"
[[ "$LOCAL_CIDR" =~ ^[0-9.]+/[0-9]+$ ]] || die "invalid LOCAL_CIDR"
[[ "$REMOTE_CIDR" =~ ^[0-9.]+/[0-9]+$ ]] || die "invalid REMOTE_CIDR"
[[ "$PING_COUNT" =~ ^[1-9][0-9]*$ ]] || die "PING_COUNT must be positive"

LOCAL_IP="${LOCAL_CIDR%/*}"
REMOTE_IP="${REMOTE_CIDR%/*}"
command -v ip >/dev/null 2>&1 || die "ip is required"
command -v ssh >/dev/null 2>&1 || die "ssh is required"

show_local_state() {
  echo "hostname=$(hostname)"
  echo "expected_interface=${LOCAL_INTERFACE} expected_cidr=${LOCAL_CIDR}"
  ip -br link show
  ip -br -4 address show
  if [ -e "/sys/class/net/${LOCAL_INTERFACE}" ]; then
    ip -d link show dev "$LOCAL_INTERFACE"
    printf 'operstate='; cat "/sys/class/net/${LOCAL_INTERFACE}/operstate"
    printf 'carrier='; cat "/sys/class/net/${LOCAL_INTERFACE}/carrier" 2>/dev/null \
      || echo unavailable
    if command -v ethtool >/dev/null 2>&1; then
      ethtool "$LOCAL_INTERFACE" 2>/dev/null \
        | grep -E 'Speed:|Duplex:|Link detected:' || true
    else
      echo "ethtool=unavailable"
    fi
  else
    echo "interface_status=MISSING"
  fi
}

show_remote_state() {
  ssh "$REMOTE_HOST" \
    "env IFACE='$REMOTE_INTERFACE' CIDR='$REMOTE_CIDR' bash -s" <<'REMOTE'
set -euo pipefail
echo "hostname=$(hostname)"
echo "expected_interface=${IFACE} expected_cidr=${CIDR}"
ip -br link show
ip -br -4 address show
if [ -e "/sys/class/net/${IFACE}" ]; then
  ip -d link show dev "$IFACE"
  printf 'operstate='; cat "/sys/class/net/${IFACE}/operstate"
  printf 'carrier='; cat "/sys/class/net/${IFACE}/carrier" 2>/dev/null \
    || echo unavailable
  if command -v ethtool >/dev/null 2>&1; then
    ethtool "$IFACE" 2>/dev/null \
      | grep -E 'Speed:|Duplex:|Link detected:' || true
  else
    echo "ethtool=unavailable"
  fi
else
  echo "interface_status=MISSING"
fi
REMOTE
}

has_local_address() {
  ip -o -4 address show dev "$LOCAL_INTERFACE" \
    | awk '{print $4}' | grep -Fxq "$LOCAL_CIDR"
}

has_remote_address() {
  ssh "$REMOTE_HOST" \
    "ip -o -4 address show dev '$REMOTE_INTERFACE' | awk '{print \$4}' | grep -Fxq '$REMOTE_CIDR'"
}

check_connectivity() {
  local status=0
  log "Master to worker1 25G ping"
  ping -I "$LOCAL_INTERFACE" -c "$PING_COUNT" -W 2 "$REMOTE_IP" || status=1
  log "Worker1 to master 25G ping"
  ssh "$REMOTE_HOST" \
    "ping -I '$REMOTE_INTERFACE' -c '$PING_COUNT' -W 2 '$LOCAL_IP'" \
    || status=1
  return "$status"
}

log "Master 25G state"
show_local_state
log "Worker1 25G state"
show_remote_state

[ -e "/sys/class/net/${LOCAL_INTERFACE}" ] \
  || die "master interface does not exist: ${LOCAL_INTERFACE}"
ssh "$REMOTE_HOST" "test -e '/sys/class/net/${REMOTE_INTERFACE}'" \
  || die "worker1 interface does not exist: ${REMOTE_INTERFACE}"

if [ "$APPLY" = "0" ]; then
  if has_local_address && has_remote_address && check_connectivity; then
    echo "STRICT_25G_LINK_READY"
    exit 0
  fi
  echo "classification=STRICT_25G_LINK_NOT_CONFIGURED"
  echo "next=APPLY=1 bash scripts/restore_strict_25g_link.sh"
  exit 2
fi

[ "$(id -u)" -eq 0 ] || die "APPLY=1 must run as root on master"

log "Restore fixed 25G addresses"
ip link set dev "$LOCAL_INTERFACE" up
ip address replace "$LOCAL_CIDR" dev "$LOCAL_INTERFACE"
ssh "$REMOTE_HOST" \
  "ip link set dev '$REMOTE_INTERFACE' up && ip address replace '$REMOTE_CIDR' dev '$REMOTE_INTERFACE'"
sleep 2

has_local_address || die "master address was not applied: ${LOCAL_CIDR}"
has_remote_address || die "worker1 address was not applied: ${REMOTE_CIDR}"
check_connectivity || die "25G addresses exist but bidirectional ping failed"

echo "STRICT_25G_LINK_RESTORED"
echo "master=${LOCAL_INTERFACE}:${LOCAL_CIDR}"
echo "worker1=${REMOTE_INTERFACE}:${REMOTE_CIDR}"
echo "next=bash scripts/deploy_datasystem_25g_master.sh"
