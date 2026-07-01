#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
SERVICE="${SERVICE:-inference-brpc-kvc-probe}"
POD_LABEL="${POD_LABEL:-app=${SERVICE}}"
PORT="${PORT:-18100}"
PAIREC_DEPLOYMENT="${PAIREC_DEPLOYMENT:-pairec}"
MASTER_NODE="${MASTER_NODE:-master}"
WORKER_NODE="${WORKER_NODE:-worker1}"
SSH_WORKER="${SSH_WORKER:-0}"

section() {
  printf '\n== %s ==\n' "$1"
}

run_allow_fail() {
  printf '+ %s\n' "$*"
  "$@" || true
}

run_shell_allow_fail() {
  printf '+ %s\n' "$*"
  bash -lc "$*" || true
}

tcp_check() {
  local host="$1"
  local port="$2"
  if [ -z "$host" ] || [ -z "$port" ]; then
    echo "skip tcp check: host or port is empty"
    return 0
  fi
  echo "-- tcp ${host}:${port} --"
  if command -v nc >/dev/null 2>&1; then
    nc -vz -w 3 "$host" "$port" || true
  else
    timeout 3 bash -lc "cat < /dev/null > /dev/tcp/${host}/${port}" \
      && echo "tcp open" || echo "tcp failed"
  fi
}

jsonpath_or_empty() {
  kubectl "$@" 2>/dev/null || true
}

section "target discovery"
run_allow_fail kubectl -n "$NAMESPACE" get svc "$SERVICE" -o wide
run_allow_fail kubectl -n "$NAMESPACE" get endpoints "$SERVICE" -o wide
run_allow_fail kubectl -n "$NAMESPACE" get pods -l "$POD_LABEL" -o wide

SVC_IP="$(jsonpath_or_empty -n "$NAMESPACE" get svc "$SERVICE" -o jsonpath='{.spec.clusterIP}')"
POD="$(jsonpath_or_empty -n "$NAMESPACE" get pod -l "$POD_LABEL" -o jsonpath='{.items[0].metadata.name}')"
POD_IP=""
POD_NODE=""
if [ -n "$POD" ]; then
  POD_IP="$(jsonpath_or_empty -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.podIP}')"
  POD_NODE="$(jsonpath_or_empty -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
fi

MASTER_IP="$(jsonpath_or_empty get node "$MASTER_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')"
WORKER_IP="$(jsonpath_or_empty get node "$WORKER_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')"

cat <<EOF
service=${NAMESPACE}/${SERVICE}
service_ip=${SVC_IP}
pod=${POD}
pod_ip=${POD_IP}
pod_node=${POD_NODE}
master=${MASTER_NODE} ${MASTER_IP}
worker=${WORKER_NODE} ${WORKER_IP}
current_host=$(hostname 2>/dev/null || true)
EOF

section "nodes and system pods"
run_allow_fail kubectl get nodes -o wide
run_shell_allow_fail "kubectl -n kube-system get pods -o wide | grep -E 'calico|kube-proxy|coredns|nvidia' || true"

section "host tcp checks from current shell"
tcp_check "$POD_IP" "$PORT"
tcp_check "$SVC_IP" "$PORT"

section "host route and sysctl checks"
if [ -n "$POD_IP" ]; then
  run_allow_fail ip route get "$POD_IP"
fi
if [ -n "$SVC_IP" ]; then
  run_allow_fail ip route get "$SVC_IP"
fi
run_shell_allow_fail "ip route | grep -E '172\\.16\\.|cali|tunl|vxlan|bird|proto bird' || true"
run_shell_allow_fail "ip -d link show 2>/dev/null | grep -E 'cali|tunl|vxlan' || true"
run_shell_allow_fail "sysctl net.ipv4.ip_forward net.ipv4.conf.all.rp_filter net.ipv4.conf.default.rp_filter 2>/dev/null || true"

section "service dataplane checks on current host"
if [ -n "$SVC_IP" ]; then
  run_shell_allow_fail "sudo -n iptables-save -t nat 2>/dev/null | grep -F '${SVC_IP}' || true"
  run_shell_allow_fail "sudo -n iptables-save -t nat 2>/dev/null | grep -E 'KUBE-SVC|KUBE-SEP' | head -80 || true"
  run_shell_allow_fail "sudo -n ipvsadm -Ln 2>/dev/null | grep -A3 -F '${SVC_IP}:${PORT}' || true"
fi

section "target pod self checks"
if [ -n "$POD" ]; then
  run_allow_fail kubectl -n "$NAMESPACE" exec "pod/${POD}" -- sh -lc "hostname; cat /etc/resolv.conf; ss -lntp 2>/dev/null | grep ':${PORT}' || true"
  run_allow_fail kubectl -n "$NAMESPACE" exec "pod/${POD}" -- sh -lc "if [ -x /opt/pairec-brpc/bin/brpc_recommend_client ]; then /opt/pairec-brpc/bin/brpc_recommend_client --server=127.0.0.1:${PORT} --method=health --requests=1; fi"
  if [ -n "$POD_IP" ]; then
    run_allow_fail kubectl -n "$NAMESPACE" exec "pod/${POD}" -- sh -lc "if [ -x /opt/pairec-brpc/bin/brpc_recommend_client ]; then /opt/pairec-brpc/bin/brpc_recommend_client --server=${POD_IP}:${PORT} --method=health --requests=1; fi"
  fi
  if [ -n "$SVC_IP" ]; then
    run_allow_fail kubectl -n "$NAMESPACE" exec "pod/${POD}" -- sh -lc "if [ -x /opt/pairec-brpc/bin/brpc_recommend_client ]; then /opt/pairec-brpc/bin/brpc_recommend_client --server=${SVC_IP}:${PORT} --method=health --requests=1; fi"
  fi
fi

section "pairec pod to target tcp checks"
run_allow_fail kubectl -n "$NAMESPACE" get deploy "$PAIREC_DEPLOYMENT" -o wide
if [ -n "$POD_IP" ]; then
  run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- sh -lc "if command -v nc >/dev/null 2>&1; then nc -vz -w 3 ${POD_IP} ${PORT}; else echo 'nc not found in pairec pod'; fi"
fi
if [ -n "$SVC_IP" ]; then
  run_allow_fail kubectl -n "$NAMESPACE" exec "deploy/${PAIREC_DEPLOYMENT}" -- sh -lc "if command -v nc >/dev/null 2>&1; then nc -vz -w 3 ${SVC_IP} ${PORT}; else echo 'nc not found in pairec pod'; fi"
fi

if [ "$SSH_WORKER" = "1" ] && [ -n "$WORKER_IP" ]; then
  section "worker host reverse-path checks via ssh"
  run_shell_allow_fail "ssh ${WORKER_IP} 'hostname; ip route get ${MASTER_IP:-127.0.0.1} || true; ip route | grep -E \"172\\\\.16\\\\.|cali|tunl|vxlan|proto bird\" || true; sysctl net.ipv4.ip_forward net.ipv4.conf.all.rp_filter net.ipv4.conf.default.rp_filter 2>/dev/null || true'"
fi

section "interpretation and next actions"
cat <<EOF
Read the checks in this order:

1. If target pod self checks pass but current-host tcp checks fail:
   The service itself is healthy. The break is current node host networking.

2. If current-host -> PodIP fails and 'ip route get <PodIP>' has no Calico route:
   Refresh Calico on the current node first, then worker if needed:
     kubectl -n kube-system delete pod -l k8s-app=calico-node --field-selector spec.nodeName=${MASTER_NODE}
     kubectl -n kube-system delete pod -l k8s-app=calico-node --field-selector spec.nodeName=${WORKER_NODE}

3. If current-host -> PodIP works but current-host -> ClusterIP fails:
   Refresh kube-proxy on the current node:
     kubectl -n kube-system delete pod -l k8s-app=kube-proxy --field-selector spec.nodeName=${MASTER_NODE}
   If the kube-proxy label is different, inspect:
     kubectl -n kube-system get pods -o wide | grep kube-proxy

4. If PaiRec pod -> target works but host -> target fails:
   The benchmark path inside Kubernetes is valid. Host-only probes should be run via kubectl exec,
   port-forward, or after fixing node host dataplane.

5. If both host and PaiRec pod fail cross-node:
   Focus on Calico node health, route tables, and node-to-node firewall/rp_filter.
EOF
