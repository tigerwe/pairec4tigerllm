# bRPC UB 部署前资格验证

本阶段不部署 Kubernetes。目标是在相同业务构造下完成三项门禁：

1. post-rank Hop1 到 Hop2：`c1000 × 102400 bytes` TCP/UB同口径验证；
2. DataSystem KVC：`c32 × 3.5 MiB`和`c32 × 8 MiB`完整性验证；
3. post-rank UB c1000与KVC UB c32联合负载。

传输边界固定为：

```text
Go/PaiRec --TCP--> C++ Hop1 --TCP或UB（本次A/B变量）--> C++ Hop2
```

Hop1前端始终为TCP。UB模式只修改Hop1拥有的1000个`brpc::Channel`和Hop2的
business/pressure监听端，不引入Bridge，不改变1路业务加999路Health、候选顺序、
102400字节payload或950路启动marker。

## 1. 构建

在master执行固定版本构建：

```bash
cd /home/zcx/workspace/pairec4tigerllm

BRPC_ROOT=/home/zcx/workspace/brpc \
BAZEL_OUTPUT_BASE=/root/.cache/bazel/_bazel_root/02df77b0294ccdcf08a6a8a39050de3a \
LOCAL_BCR_REGISTRY=/home/zcx/bazel-local-registry/bcr \
LOCAL_SECRET_REGISTRY=/home/zcx/bazel-local-registry/secretflow \
INSTALL_DIR=/opt/pairec-brpc-ub-recommend-known-good \
BUILD_JOBS=32 \
bash scripts/build_brpc_ub_recommend_known_good.sh
```

成功标志必须包含：

```text
BRPC_KNOWN_GOOD_POST_RANK_BUILD_OK
```

将`brpc_post_rank_hop`复制到node1的同一安装目录。node1还需要通过
`scripts/build_kvc_burst_wrapper.sh`获得最新的`kvc_ub_integrity_probe`。

## 2. post-rank c1000 × 100 KiB

先在node1启动Hop2：

```bash
cd /home/zcx/workspace/pairec4tigerllm
ROLE=hop2 TRANSPORT=ub \
LOG_DIR=/root/brpc-ub-post-rank/log \
bash scripts/run_brpc_ub_post_rank_hop.sh
```

再在master启动Hop1，其中`HOP2_HOST`必须是node1的UB可达地址：

```bash
cd /home/zcx/workspace/pairec4tigerllm
ROLE=hop1 TRANSPORT=ub HOP2_HOST=141.62.33.117 \
LOG_DIR=/root/brpc-ub-post-rank/log \
bash scripts/run_brpc_ub_post_rank_hop.sh
```

Hop1就绪日志必须同时包含`connected_sessions=1000`对应JSON字段、
`transport=ub`、`Use Bonding`和trace-key ready marker。保持两个进程运行，在master执行：

```bash
cd /home/zcx/workspace/pairec4tigerllm
HOP1_LOG=$(ls -1t /root/brpc-ub-post-rank/log/post-rank-hop1-ub-*.log | head -1)

TRANSPORT=ub REQUESTS=3 SERVER=127.0.0.1:18311 \
HOP1_LOG="$HOP1_LOG" \
bash scripts/run_brpc_ub_post_rank_qualification.sh
```

门禁标志：

```text
PAIREC_POST_RANK_UB_C1000_100K_OK
```

TCP对照只把两个进程及资格脚本的`TRANSPORT`改为`tcp`。正式比较按
`TCP → UB → UB → TCP`执行，每组使用相同二进制、机器、端口、超时和请求数。

## 3. KVC UB c32双对象

在node1执行，DataSystem Worker保持在master `141.62.33.105:32501`：

```bash
cd /home/zcx/workspace/pairec4tigerllm
WORKER_HOST=141.62.33.105 WORKER_PORT=32501 ITERATIONS=3 \
bash scripts/run_kvc_ub_c32_qualification.sh
```

探针为每个lane建立独立跨节点KVClient。每轮先同时释放32路Set，全部Set完成后再通过
第二个栅栏同时释放32路Get；每个对象逐字节比较并核对SHA-256。Access Log必须精确为
UB Set/Get且`tcp=0`。门禁标志：

```text
KVC_UB_C32_DUAL_OBJECT_QUALIFICATION_PASS
```

## 4. 联合负载

保持UB Hop1/Hop2运行。在master执行；脚本通过SSH在node1预生成32份KVC payload，
并让KVC与post-rank资格客户端都完成初始化、停在各自的start-file栅栏；确认两端ready后
背靠背释放两道闸门，确保c32与c1000同时起跑：

```bash
cd /home/zcx/workspace/pairec4tigerllm
HOP1_LOG=$(ls -1t /root/brpc-ub-post-rank/log/post-rank-hop1-ub-*.log | head -1)

KVC_CLIENT_HOST=node1 \
KVC_REMOTE_REPO=/home/zcx/workspace/pairec4tigerllm \
KVC_WORKER_HOST=141.62.33.105 KVC_WORKER_PORT=32501 \
KVC_ITERATIONS=10 POST_RANK_REQUESTS=3 \
POST_RANK_SERVER=127.0.0.1:18311 HOP1_LOG="$HOP1_LOG" \
bash scripts/run_brpc_ub_post_rank_kvc_joint_qualification.sh
```

联合门禁分别运行3.5MiB和8MiB KVC负载，最终必须输出：

```text
PAIREC_BRPC_UB_C1000_KVC_UB_C32_JOINT_PASS
```

任一阶段出现bthread非法key、候选顺序/SHA错误、pressure不是999/999、KVC内容错误、
DataSystem Access Log出现TCP或进程非零退出，均不得进入Kubernetes部署阶段。
