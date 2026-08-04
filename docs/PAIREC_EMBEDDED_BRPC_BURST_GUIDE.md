# PaiRec 内嵌 BRPC Burst 实验

## 链路

实验链路独立于原 PaiRec：

```text
HTTP -> pairec-brpc-wrapper
     -> 内嵌 burst coordinator
        -> 1 路 Recommend -> 192.168.100.11:18103 -> 18100 TRT-LLM
        -> N-1 路 100KB Health -> 192.168.100.11:18103 本地终止
```

并发数仅由实验 ConfigMap 配置。一次只执行一个 burst；后来的 PaiRec 请求等待。
正式 Recommend 完成后立即返回，Health 尾部异步收口。Health 失败不改变成功的推荐
结果，但该样本的 `burst_valid` 为 false。

启动时必须预连接全部 N 个 BRPC session，否则 PaiRec 初始化失败且 Pod 不会 Ready。
结果缓存、HTTP fallback 和 BRPC 重试均关闭。每次请求输出按 `request_id` 关联的
`pairec_brpc_burst_start`、`pairec_brpc_burst_business_complete` 和
`pairec_brpc_burst_complete` JSON 事件。

## 构建和导入

在 master 仓库目录执行：

```bash
IMAGE=docker.io/library/pairec-server:k8s-arm64-brpc-v1
TAR=/home/zcx/pairec-server-k8s-arm64-brpc-v1.tar

bash scripts/build_pairec_binary_image.sh "$IMAGE"
docker save -o "$TAR" "$IMAGE"

IMAGE="$IMAGE" \
IMAGE_TAR="$TAR" \
APP_LABEL=app=pairec-brpc-wrapper \
  bash scripts/import_pairec_server_image_to_k8s.sh
```

## B 组：并发 1 smoke

```bash
bash scripts/k8s_apply_pairec_brpc_wrapper.sh
bash scripts/k8s_switch_pairec_brpc_burst_mode.sh c1

EXPECTED_CONCURRENCY=1 REQUESTS=3 \
  bash scripts/benchmark_pairec_brpc_burst.sh |
  tee /tmp/pairec-brpc-burst-c1-smoke.log
```

## C 组：并发 1000

```bash
bash scripts/k8s_switch_pairec_brpc_burst_mode.sh c1000

# 先做结构 smoke
EXPECTED_CONCURRENCY=1000 REQUESTS=3 \
  bash scripts/benchmark_pairec_brpc_burst.sh |
  tee /tmp/pairec-brpc-burst-c1000-smoke.log

# 再采初步分布
EXPECTED_CONCURRENCY=1000 REQUESTS=100 \
  bash scripts/benchmark_pairec_brpc_burst.sh |
  tee /tmp/pairec-brpc-burst-c1000-n100.log

# 只有前两步全通过后才采正式 p99
EXPECTED_CONCURRENCY=1000 REQUESTS=1000 \
  bash scripts/benchmark_pairec_brpc_burst.sh |
  tee /tmp/pairec-brpc-burst-c1000-n1000.log
```

验收不设置人为时延目标。报告给出推荐 HTTP E2E、前段 BRPC、Wrapper、推理和
runner 的 p50/p95/p99。样本有效要求业务成功、全部 Health 成功、Wrapper trace
完整、预连接数和 armed 数等于配置并发，且三个 Deployment 均不重启。
