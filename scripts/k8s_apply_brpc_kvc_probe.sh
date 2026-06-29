#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-pairec}"
APP="${APP:-inference-brpc-kvc-probe}"
IMAGE="${IMAGE:-docker.io/library/pairec-brpc-inference:k8s-arm64-trtllm-v1}"
NODE_NAME="${NODE_NAME:-worker1}"
DATASYSTEM_HOST="${DATASYSTEM_HOST:-141.61.91.189}"
DATASYSTEM_PORT="${DATASYSTEM_PORT:-18481}"
KVC_PROBE_OBJECT_COUNT="${KVC_PROBE_OBJECT_COUNT:-4}"
KVC_PROBE_OBJECT_BYTES="${KVC_PROBE_OBJECT_BYTES:-3670016}"
KVC_PROBE_TIMEOUT_MS="${KVC_PROBE_TIMEOUT_MS:-5000}"
KVC_PROBE_TTL_SEC="${KVC_PROBE_TTL_SEC:-120}"

echo "Applying brpc DataSystem KVC MSet/MGet probe"
echo "  namespace:       ${NAMESPACE}"
echo "  app:             ${APP}"
echo "  image:           ${IMAGE}"
echo "  node:            ${NODE_NAME}"
echo "  datasystem:      ${DATASYSTEM_HOST}:${DATASYSTEM_PORT}"
echo "  object_count:    ${KVC_PROBE_OBJECT_COUNT}"
echo "  object_bytes:    ${KVC_PROBE_OBJECT_BYTES}"
echo "  timeout_ms:      ${KVC_PROBE_TIMEOUT_MS}"
echo "  ttl_sec:         ${KVC_PROBE_TTL_SEC}"
echo

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

cat <<YAML | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP}
  namespace: ${NAMESPACE}
  labels:
    app: ${APP}
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: ${APP}
  template:
    metadata:
      labels:
        app: ${APP}
    spec:
      nodeName: ${NODE_NAME}
      containers:
        - name: brpc-kvc-probe
          image: ${IMAGE}
          imagePullPolicy: IfNotPresent
          command:
            - /opt/pairec-brpc/bin/brpc_inference_server
          args:
            - --listen_port=18100
            - --backend=datasystem_kv_probe
            - --datasystem_host=${DATASYSTEM_HOST}
            - --datasystem_port=${DATASYSTEM_PORT}
            - --kvc_probe_object_count=${KVC_PROBE_OBJECT_COUNT}
            - --kvc_probe_object_bytes=${KVC_PROBE_OBJECT_BYTES}
            - --kvc_probe_timeout_ms=${KVC_PROBE_TIMEOUT_MS}
            - --kvc_probe_ttl_sec=${KVC_PROBE_TTL_SEC}
          ports:
            - containerPort: 18100
              name: brpc
          env:
            - name: DATASYSTEM_HOST
              value: "${DATASYSTEM_HOST}"
            - name: DATASYSTEM_PORT
              value: "${DATASYSTEM_PORT}"
            - name: LD_PRELOAD
              value: ""
            - name: LD_LIBRARY_PATH
              value: "/usr/local/lib/python3.11/site-packages/yr/datasystem/lib:/usr/local/lib:/usr/local/lib64:/usr/lib64:/usr/lib:/opt/openEuler/gcc-toolset-14/root/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
          resources:
            requests:
              cpu: "1"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 16Gi
          startupProbe:
            tcpSocket:
              port: 18100
            periodSeconds: 5
            timeoutSeconds: 1
            failureThreshold: 36
          readinessProbe:
            tcpSocket:
              port: 18100
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 6
          livenessProbe:
            tcpSocket:
              port: 18100
            periodSeconds: 15
            timeoutSeconds: 3
            failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP}
  namespace: ${NAMESPACE}
  labels:
    app: ${APP}
spec:
  type: ClusterIP
  selector:
    app: ${APP}
  ports:
    - name: brpc
      port: 18100
      targetPort: 18100
YAML

kubectl -n "$NAMESPACE" rollout status "deployment/${APP}" --timeout=5m
kubectl -n "$NAMESPACE" get pods -l "app=${APP}" -o wide
kubectl -n "$NAMESPACE" get svc "$APP" -o wide

echo
echo "Verify with:"
echo "  INFERENCE_SERVICE=${APP} REQUESTS=1 TOPK=1 bash scripts/test_brpc_kvc_mset_mget_probe.sh"
