> 2026-08-12 F19 overlay 新gateway部署成功但旧SHA门禁误报已修复：Pod PID1已加载worker1新gateway SHA `c9d723...`和V3 TRT库SHA `86aef5...`，rollout/Ready均成功；脚本仍默认固定V1 SHA `39d858.../c646...`，导致合法重编必然在disabled验证阶段失败。现将EXPECTED SHA改为可选发布锁：默认严格校验PID1 overlay路径、gateway/native capability marker、无缺库、`ldd`实际路径及loaded TRT SHA等于overlay TRT SHA，并打印本轮检测值；显式传EXPECTED值时仍执行精确发布锁定。当前Deployment停留在attribution disabled，下一步拉取后重新执行apply即可继续enable和exact smoke。F19保持in_progress。
>
> 2026-08-12 F19 V3 暖态样本进一步定位并修正gateway闭合计时：同Pod后续请求runner=`184ms`、31 output token，prefill已由首请求`109.333ms`回落到`8.002ms`，确认首请求存在约101ms预热；暖态decode gap=`154.294ms`、30个间隔约`5.143ms/步`，仍是主要耗时。该请求真实2 Set=`14.243ms`，但native归因lookup/record仅`63/68us`。报告触发的gateway closure=`982us`并非真实漏算，而是旧实现把整数毫秒`butil::Timer::m_elapsed()`乘1000，与微秒阶段和比较产生0~999us量化误差；现改为同一`steady_clock`直接记录runner total_us并继续执行<=100us严格门禁，response trace也使用精确微秒值。worker1构建脚本新增`BUILD_TRTLLM=0`，该gateway-only修复可复用已验证V3 native库，仅重编gateway。单请求摘要在PaiRec未回传per-token字段时也会由gateway runner/output token派生。下一步worker1执行gateway-only构建，master重新apply overlay后连续采样；当前性能根因已收敛为“首请求预热约101ms + 31-token暖态decode约154ms”，不是DataSystem归因bookkeeping。F19保持in_progress。
>
> 2026-08-12 F19 V3 首个远端单请求阶段闭合完成：gateway runner=`296.000ms`，gateway闭合误差=`90us`；native生命周期=`294.174ms`，native闭合误差=`16us`，两层证据有效。native耗时主要为prefill gap=`109.333ms`和decode gap=`174.007ms`，合计`283.340ms`、占生命周期约`96.32%`；该请求`add_token_count=31`，DataSystem Get/Set均为0，request attribution lookup=`87us`、sequence lookup=`1us`、KV update=`132us`、phase record=`83us`。这直接排除了DataSystem I/O和归因bookkeeping解释296ms，剩余变量是首请求预热、prefill执行以及接近32-token上限的decode长度。单请求摘要已补充`tr_output_tokens`、每token runner、每decode间隔和模型执行占比，避免继续把token语义差异误判成观测开销。下一步在不rollout的同一Pod连续采集暖态请求，对比output token、prefill gap和每步decode；只有同token长度暖态仍显著高于历史约94ms，才进入scheduler/GPU执行路径修复。F19保持in_progress。
>
> 2026-08-12 F19 V3 gateway protobuf构建阻塞已定位并修复：worker1的TRT-LLM `libtensorrt_llm.so`已100%编译成功，Stage 2失败是仓库内protoc 3.12.4生成代码与镜像Protobuf C++ 4.25.1不兼容（旧`generated_message_table_driven.h`已移除），不是V3 native逻辑问题。现用官方protoc 25.1重新生成`recommend`与`pipeline_service`四份架构无关C++文件，CMake明确生成器/runtime版本，单测固定检查4.25.1 tctable接口且仅在protoc 25.1下做字节一致性审计。下一步worker1拉取后重跑双阶段脚本；Stage 1可增量复用，Stage 2应完成gateway。F19保持in_progress。
>
> 2026-08-11 F19 V3 worker1构建预检修正：首次重编在镜像选择阶段误报“无CUDA静态库”，实际尚未启动CMake；原脚本使用不跟随符号链接的`find /usr/local/cuda`并吞掉所有候选错误。现改为跟随`/usr/local/cuda*`链接搜索`libcudadevrt.a`，预检容器显式清空`LD_PRELOAD`，并逐候选报告镜像缺失、架构、编译器、CUDA静态库或BRPC/protobuf SDK的具体失败原因。下一步worker1拉取后直接重跑构建脚本；若仍失败，日志将给出唯一阻塞项。F19保持in_progress。
>
> 2026-08-11 F19 V3 单请求原生时延闭合实现完成、本地待远端编译：为避免继续依赖失真的 disabled/enabled A/B，gateway 在 runner 计时结束后输出唯一 `trt_executor_request_complete`，拆分 request setup、enqueue、await final 和 response extraction；native tracker 在同一 UUID 的 `datasystem_request_complete` 上追加 enqueue→addSequence→prefill gap→逐轮 addToken/decode gap→finalization gap→removeSequence 两层闭合，并单列 attribution map、sequence lookup、KV update 和 tracker record 子项。严格单请求及1000请求门禁统一要求 gateway/native 事件完整、phase_unknown=0、两层闭合误差<=100us且生命周期数量一致；汇总器输出各阶段p50/p95/p99与runner-minus-native。Gateway/native阶段计时都只在归因开启时进入热路径，每token不打日志，完成事件仍仅各一条；已知DataSystem DEBUG格式串3个占位符仅2个参数的未定义行为也已修复。补丁支持原始/V1/V2/早期无条件计时V3/最终V3升级，并对保留的真实TensorRT-LLM源码副本连续应用两次成功；C++ tracker runtime smoke、完整65项Python测试、Go services测试及Python/Shell静态检查通过。下一步worker1重编V3 native library/gateway，master应用overlay后先跑单请求trace，据阶段占比直接定位约85ms落在prefill、decode、KV更新还是finalization；暂不以A/B作为根因判断。F19保持in_progress。
>
> 2026-08-11 F19 native编译镜像纠正：双容器脚本第一阶段在`zcx-pairec-image:v1.1`内预检失败于缺少`libcudadevrt.a`，未进入CMake。重新按shell history的命令顺序核对后确认，CUDA静态runtime的`find/test`发生在`trtllm-build-f154`容器，其镜像为`zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1`；`zcx-pairec-image:v1.1`只是后续探索命令。脚本现为TRT候选增加动态预检，必须同时具备arm64、cmake/C++、`libcudadevrt.a`和同目录`libcudart_static.a`，并优先选择该SDK tag；gateway仍独立执行protobuf/brpc SDK预检。下一步worker1拉取后两阶段都显式使用该SDK镜像重跑。F19保持in_progress。
>
> 2026-08-11 F19 gateway去除远程`protoc`依赖：历史`zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1`预检仍失败，确认该镜像虽用于TRT-LLM/BRPC链接，但不保证包含`protoc`。现用项目已确认的`libprotoc 3.12.4`预生成`recommend`/`pipeline_service` C++源码并纳入`cpp/brpc_gateway/generated`；CMake新增默认关闭的`PAIREC_USE_PREGENERATED_PROTO`，仅F19 worker脚本显式开启。gateway容器现只要求protobuf headers/libs和brpc C++ SDK，不再要求`protoc`；现有镜像构建仍保持现场生成默认行为。下一步worker1拉取后重跑双容器脚本。F19保持in_progress。
>
> 2026-08-11 F19 gateway历史编译镜像已恢复：worker1 shell history确认上次完整TRT-LLM/BRPC SDK环境为`zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1`，当时容器名为`trtllm-build-f154`。双容器脚本先前只扫描BRPC inference tags，因此在编译前正确失败于“no local arm64 gateway image”。现已将该SDK tag加为gateway第一候选，仍强制`protoc/protobuf/brpc/arm64`动态预检；下一步worker1拉取并重跑，配置应显示`gateway_build_image=zcx-pairec-trtllm-brpc-sdk:parallel-get-ctx224-v1`。F19保持in_progress。
>
> 2026-08-11 F19 V2 worker1编译拆分为双容器：第二次远程预检确认`zcx-pairec-image:v1.1`有CUDA/TRT-LLM/DataSystem编译环境但没有`protoc`，证明它只是上次native library编译环境，不是gateway编译环境。`scripts/build_f19_attribution_runtime_worker1.sh`现分为两阶段：`TRT_BUILD_IMAGE=zcx-pairec-image:v1.1`只编译`libtensorrt_llm.so`；第二阶段从worker1本地BRPC inference镜像中逐个执行`protoc + protobuf headers + brpc SDK + arm64`预检，选中后只编译`brpc_inference_server`。两份产物先写staging，只有两阶段、V2/token marker和`ldd`全部通过才发布到`/home/zcx/pairec-f19-runtime`，防止半更新。下一步worker1拉取后重跑脚本；F19保持in_progress。
>
> 2026-08-11 F19 V2 worker1编译脚本挂载修正：远程首次执行因将宿主机`/home/zcx/TensorRT-LLM`挂到镜像内同名的非目录路径而在container init失败。用户恢复的历史命令确认上次实际使用`zcx-pairec-image:v1.1`、容器内`/TensorRT-LLM`以及`/lib64:/host-driver:ro`。脚本现已按该已验证形态修正：宿主机与容器路径分离，仓库挂到`/mnt/pairec-src`、TRT-LLM挂到`/TensorRT-LLM`、driver挂整个目录，并将默认并行度从worker1的308核封顶为32，避免链接前编译内存风险。下一步拉取修正后重跑同一脚本。F19保持in_progress。
>
> 2026-08-11 F19 V2 worker1编译流程已固化：新增`scripts/build_f19_attribution_runtime_worker1.sh`，默认先幂等重应用当前仓库补丁，再在worker1自动选择本地`pairec-brpc-inference:k8s-arm64-trtllm-multisequence-kvc-ctx224-v1`优先镜像，同时重编`libtensorrt_llm.so`和`brpc_inference_server`。这保证先前已打补丁的源码树也会更新托管的V2 ready marker。脚本固化了上次手工编译暴露的`/host-driver/libcuda.so.1`挂载以及`libcudadevrt.a/libcudart_static.a`链接搜索路径，并在覆盖`/home/zcx/pairec-f19-runtime`前强制检查V2 marker、output-token trace字段和`ldd`无缺库。先前临时编译容器的精确名称未写入项目交接记录，因此不作伪造还原；脚本会尝试扫描worker1 shell history中的相关命令，并支持`BUILD_IMAGE`显式覆盖。本地Shell/Python静态检查和8项专项测试通过；下一步在worker1运行该脚本，然后由master apply overlay并执行3请求smoke与3对1000请求A/B。F19保持in_progress。
>
> 2026-08-11 F19 85ms runner 回归重新开工：复核发现上一轮 A/B 的 disabled 仅关闭 tracker 统计，gateway 仍无条件设置 Executor `clientId`，TRT-LLM `KVCacheManager::addToken()` 仍在每个生成 token 上竞争核心 `mSequencesMtx`，因此“两组差值接近0”不能证明整套归因补丁无开销，F19 从 done 纠正为 in_progress。当前修复将 disabled 改为不分配关联号、不设置 clientId、不维护映射；enabled 的请求映射改用独立 mutex，避免与 scheduler/sequence 核心临界区竞争。Trace 新增 `output_token_count` 和 `runner_ms_per_output_token`，A/B 新增 token 语义一致性及 runner avg<=110ms 绝对门禁，禁止两组都约180ms时再次误判PASS。本地专项36项、Go race与离线构建、protobuf descriptor、Shell/Python静态检查均通过；下一步 worker1 重打 native library/gateway overlay，先跑3请求协议smoke，再跑3对1000请求A/B。只有 enabled/disabled 均回到<=110ms且原开销预算、精确DataSystem completion同时通过，才能重新关闭F19。
>
> 2026-08-11 F19 历史A/B记录（完成判定已撤销）：三组同镜像、同缓存准备的`disabled -> enabled`配对A/B全部完成，每轮1000请求均通过业务、trace、BRPC-only、重排、资源和Pod健康门禁；enabled三轮均`datasystem_complete=1000/1000`且各记录`get_count=54/set_count=1995`。配对开销中位数：client avg=`-0.672ms`、p99=`-1.947ms`、闭环吞吐损失=`0.076%`，runner avg/p99=`-0.188/-1.000ms`。后续复核发现两种模式共享clientId传播与每token核心锁竞争，该轮只能作为tracker统计/日志开销证据，不能排除归因补丁导致共同的约85ms回归。
>
> 2026-08-11 F19 归因开关性能A/B工具完成、本地待验收：新增一键三对`disabled -> enabled`驱动器和独立汇总器；每轮只复用现有镜像，设置归因模式后强制inference rollout以清空native缓存，再执行相同warmup和1000请求严格链路验收。enabled轮继续要求100%精确completion，disabled/enabled均保留业务、trace闭合、BRPC-only、rerank、资源和Pod健康门禁。正式workload新增独立墙钟区间，闭环吞吐不再由平均E2E反推，也不包含rollout或completion等待。最终按三组配对差值中位数执行avg<=0.1ms、p99<=0.5ms、吞吐损失<=1%门禁，并输出runner与DataSystem指标；shell/Python静态检查及F19相关30项测试通过。下一步master拉取后运行`scripts/benchmark_f19_attribution_ab.sh`；F19保持in_progress。
>
> 2026-08-11 F19 1000请求严格UUID归因验收通过：正式 inference 已确认 `TLLM_LOG_LEVEL=INFO` 生效，warmup 与 workload 均输出 `NATIVE_DATASYSTEM_COMPLETIONS_OK`；workload `valid=1000/1000`、`missing=0`、`invalid=0`、`datasystem_complete=1000`，证明每个 PaiRec request_id 均精确关联唯一、完整且无 failed/pending/unknown 的 native DataSystem completion，之前失败确认为DEBUG日志采集背压而非Set挂死。该轮E2E avg/p50/p95/p99=`195.395/194.584/203.213/207.795ms`，较F20历史约110ms的增量几乎全部位于生成式召回：generative avg=`184.332ms`、inference avg=`179.852ms`，向量召回avg=`8.883ms`、DeepFM avg=`9.249ms`均稳定。原生归因汇总为`set_count=1995`、每请求Set累计avg/p99=`11.496/14.201ms`，`get_count=66`、每请求Get累计avg/p99=`0.414/7.096ms`；API耗时均值合计约`11.91ms/请求`且Set不由PaiRec同步等待，不能解释约85ms增量。下一步执行至少三组同镜像、同缓存准备、仅切换归因开关的disabled/enabled交替A/B，验证avg开销<=0.1ms、p99<=0.5ms、吞吐损失<=1%。F19保持in_progress。
>
> 2026-08-11 F19 真实 DataSystem I/O 请求级归因通过：严格 runtime Health 正常；连续3轮均执行 inference reset、195请求 prime 和单次 PaiRec replay，`valid=3/3`。每轮旧 KVC 窗口证据均为3次offload+2次onboard，唯一同 UUID native completion 也均为`set_count=3/get_count=2`、`attribution_complete=true`，失败/pending/unknown为0，Pod无重启和crash。三轮E2E=`246.302/258.400/264.759ms`，均值`256.487ms`；BRPC差值均为`4ms`；KVC均值`33.714ms`（offload `19.881ms`、onboard `13.833ms`）。1000请求严格验收先后暴露输出目录、Pod选择、可轮转ready marker和DEBUG日志采集背压问题。最新失败时业务循环已经返回，但35秒后流式日志尾部仅追到native correlation 431且停在一次offload中，而起始correlation约199，说明collector被逐token DEBUG洪流显著拖后，不能据此判定Set挂死。正式验收现强制inference `TLLM_LOG_LEVEL=INFO`，warmup后若仍出现DEBUG立即失败关闭；PaiRec/inference日志继续从正式请求前流式落盘，completion的INFO事件不会丢失。F19相关26项测试通过。下一步拉取后重跑1000请求；若INFO门禁通过后仍有missing/incomplete，才按真实native pending故障处理。F19保持in_progress。
>
> 2026-08-11 F19 K8s runtime overlay 与严格 Ready 已通过：inference Pod 在 worker1 以 `/opt/pairec-f19/bin/brpc_inference_server` 作为 PID 1；gateway、检查目录和 loader 实际路径三处 SHA256 均匹配 F19 产物，`ldd` 无缺库且实际加载被 File hostPath 覆盖的 F19 `libtensorrt_llm.so`。attribution disabled/enabled 两次 rollout 均成功，严格环境下 BRPC Health 返回 `code=200/status=healthy`，证明 native capability 开关有效。部署脚本最初在首个 Recommend 前要求 `datasystem_attribution_ready`，但 Tracker 只在首次 `registerRequest()` 时构造并输出该事件；门禁顺序已改为先跑精确单请求、再检查 ready marker，并新增 `smoke` 子命令从当前严格 Pod 继续而不重复 rollout。下一步执行单请求 UUID completion smoke，再进行3/1000请求严格门禁与开关A/B。F19保持in_progress。
>
> 2026-08-10 F19 Native TRT KVC 请求级 DataSystem 归因远端编译完成、K8s 验收待执行：worker1 的历史 DataSystem/parallel-Get TensorRT-LLM 源码树已成功应用兼容补丁；在 GPU 构建容器补齐 CUDA driver 与静态 runtime 搜索路径后，`libtensorrt_llm.so` 完成链接，产物 535MiB 且包含唯一事件标识 `datasystem_request_complete`。同一源码和原生库随后成功编译 ARM64 `brpc_inference_server`，产物 261KiB，包含 `native_datasystem_completion_event_pending` 和严格 Ready 状态 `not_ready_datasystem_attribution`。两个待部署 overlay 已固化到 worker1 `/home/zcx/pairec-f19-runtime/{bin,lib}`。下一步先审计现有 inference Deployment 的 image/hostPath，排除旧二进制覆盖，再以 File hostPath 同时覆盖 gateway 与 native library；先关闭门禁完成容器依赖/Health smoke，再开启归因执行 3/1000 请求严格 UUID join 和开关 A/B。F19保持in_progress。
>
> 2026-08-10 F19 Native TRT KVC 请求级 DataSystem 归因本地实现完成、远端验收待执行：gateway为每个PaiRec UUID分配单调64位关联号并写入Executor `clientId`；TensorRT-LLM源码补丁把关联号保存到KV sequence，在真实Get/Set/MGet/MSet调用边界累计次数、整数微秒、失败和pending，最后一个native sample执行`removeSequence`后输出唯一`datasystem_request_complete`。汇总器按UUID精确join，严格拒绝missing/duplicate/pending/unknown/failure；单请求脚本可等待最终事件。业务响应不等待异步Set，无时间窗口推断。验证：补丁对全新源码副本连续应用两次通过；tracker C++17运行时smoke确认双lifecycle聚合`get=1/set=1,complete=true`及TTL pending超时`complete=false`；Python 8项单测、py_compile、bash语法、JSON与diff检查通过。worker1首次应用暴露历史latency trace单Get和`trtllm-parallel-get`分支的批量/并行Get源码形态；已从本地提交`c3b25b1`及两份parallel-get patch恢复准确实现，patcher现兼容内联hash/局部key、显式/auto vector Get，并在父线程为每个parallel Get预登记token后传入worker闭合，避免thread-local上下文跨线程丢失。本机缺brpc SDK，gateway真实ARM编译及K8s 3/1000请求严格门禁、开关A/B时延预算仍需master/worker1执行，F19保持in_progress。
>
> 2026-08-10 F20 DeepFM 后来源配额重排完成：隔离 `pairec-brpc-observed` 正式1000请求 `valid=1000/1000, missing=0, invalid=0`；客户端E2E p99=`120.318ms`（门禁`122.622ms`），rerank p99=`0.030ms`（门禁`1ms`），HTTP基线p99=`118.604ms`、BRPC增量=`1.713ms`；PaiRec CPU throttled period=`0.067%`，其余四容器均0%，无fallback。故障注入请求`cd303256-5a8b-483c-9ddc-a0cf3ebbe11c`确认`code=500/items=[]`且rerank与pipeline trace均为error；自动恢复原ConfigMap后，请求`4fc2fee7-5f61-4777-ac58-bbf380a92cec`返回`code=200`、8个向量商品加末尾2个生成式商品。正向、失败关闭、恢复及本地测试全部通过，F20标记done；下一任务为F19 native TRT KVC request identity与DataSystem Get/Set请求级归因。
>
> 2026-08-08 F18 session 正式交接完成：`HANDOFF.md` 已记录纯 BRPC 架构、1000请求正式指标、当前隔离 Deployment、master 工作树与模型/runtime hostPath、单请求复现和 trace 查看命令、CoreDNS 数字ClusterIP约束，以及 DataSystem attribution_complete=false 的准确边界。F18 保持 done；F19 native TRT KVC request identity 为 pending 且未开工。当前实验实例保持运行，未替换旧基线。
>
> 2026-08-08 F18 正式验收完成并收工：ARM64/Kubernetes 纯 BRPC observed 工作负载 1000/1000 成功，trace `valid=1000/1000, missing=0, invalid=0`，无 HTTP fallback。E2E avg/p50/p95/p99/max=`109.784/108.774/117.382/120.622/126.589ms`，PaiRec total p99=`119.627ms`；生成式/向量/DeepFM span p99=`108.794/11.923/12.081ms`。保留 HTTP 基线100请求p99=`118.422ms`，纯 BRPC p99增量仅=`2.200ms`。五容器无重启/OOM/健康故障，PaiRec CPU throttled period=`0.068%`、其余均0%，通过<=5%门禁。DataSystem `attribution_complete=0/1000` 是准确的已知限制，未伪造原生 KVC 归因；F18 标记 done，native TRT KVC request identity 转为 F19 pending。
>
> 2026-08-08 F18 纯 BRPC 三请求 smoke 全链路 PASS：`pairec-brpc-observed` Service Ready，生成式、Milvus 向量和 DeepFM 精排均通过 BRPC 且无 fallback；请求级 trace `valid=3/3, missing=0, invalid=0`，Prometheus 与五容器资源门禁通过，所有 CPU throttled period=0。E2E avg/p99=`110.843/112.792ms`，PaiRec total=`109.350/111.758ms`；生成式召回=`98.720/102.374ms`，向量召回=`9.955/11.158ms`，DeepFM精排=`10.090/12.095ms`。服务子阶段中生成式 inference p99=`96.920ms`、vector service p99=`9.888ms`、rank service p99=`11.034ms`。QuotaMultiRecall 门禁确认召回阶段生成式候选被选中；DataSystem attribution_complete=0/3 是 native TRT KVC 未传播 request identity 的已知限制。下一步保持归因门禁关闭，执行 1000 请求正式验收及 HTTP/BRPC A/B。
>
> 2026-08-08 F18 精排后来源门禁纠正：当前最新 warmup 已返回 `code=200`、size=10、DeepFM 排序后的 Top10 全部来自 `milvus_recall`；旧脚本错误地要求最终响应必须保留生成式商品，与已确认的业务语义“只要求召回阶段选择生成式候选，精排可自由淘汰”冲突。响应门禁现仅要求数量/唯一性及来源属于两路召回；正式 trace 汇总新增严格 QuotaMultiRecall 解析，每请求必须满足 `primary_selected>=primary_minimum>=1`、`final_count=50`、`degraded=false`，因此不会因移除最终 Top10 断言而丢失生成式召回保障。下一步拉取后无需重建或重启，直接重跑 3 请求 smoke。
>
> 2026-08-08 F18 adapter backend hostPath 分离修复：进一步审计发现一键脚本从隔离工作树 `/home/zcx/workspace/pairec4tigerllm-f18-4aadffd` 运行，但 manifest 将 Python backend 代码硬编码挂载自旧目录 `/home/zcx/workspace/pairec4tigerllm`，导致远端 Pod 未实际运行当前 F18 协议代码，可能直接造成 rank backend `400`。部署器现默认以 `pwd -P` 作为 `BACKEND_REPO_DIR`，并将代码与模型产物解耦：DSSM/DeepFM 分别通过独立可配置 hostPath 挂载到 `/models/dssm`、`/models/deepfm`；渲染前严格校验代码和 checkpoint 存在。下一步从 F18 工作树幂等重跑会重建两个 adapter Pod，使错误字段修复和结构化拒绝日志真正生效。
>
> 2026-08-08 F18 DeepFM BRPC `400` 错误可观测修复：请求级证据确认并行召回最终 `final_count=50`、rank adapter Health 正常、sort 在约 4.9ms 内收到 `rank service code=400`，排除候选不足、网络不可达和超时；真正拒绝原因因 Python 错误响应使用 HTTP 旧字段 `msg`，而 protobuf JSON bridge 只识别 `message`，被 `ignore_unknown_fields` 丢弃。Rank backend 现对所有拒绝同时返回 `message` 与 `msg`，并输出包含 request_id、reason、candidate_count、user_id_type 的 `deepfm_rank_rejected` 结构化日志；测试补充仅通过 `context.request_id` 传递身份的真实 protobuf JSON 形态。下一步 master 拉取后重启 deepfm-rank-brpc backend Pod 并重跑 smoke，若仍为 400 将直接得到精确 reason。
>
> 2026-08-08 F18 DeepFM BRPC fail-closed 定位工具新增：纯 BRPC observed 链路已通过 adapter Health、PaiRec rollout、Service Endpoint 与 `/ping`，首个真实 warmup 返回 `code=500/deepfm rank failed`，说明故障已收敛到精排业务调用而非部署连通性。新增只读 `diagnose_pairec_deepfm_brpc_failure.sh`，可接受 request_id 或自动提取最新 `deepfm_rank_error`，统一采集 PaiRec/rank adapter/backend 日志、Deployment/Pod/Service/Endpoints/渲染配置与 adapter Health，并分类候选数、内外层超时、模型异常、请求/响应契约、模型角色及网络故障。下一步在 master 针对失败 request_id 运行脚本，以分类证据决定预热还是协议修复。
>
> 2026-08-08 F18 PaiRec observed Service warmup 竞态修复：数字 ClusterIP 改造后 `pairec-brpc-observed` 已成功 rollout，证明 Pod Readiness 与纯 BRPC 依赖链路恢复；脚本随后立即通过 ClusterIP warmup 时遇到 `curl (7)`，说明 kube-proxy/IPVS 的 Service 数据面尚未同步。部署器现沿用已验证的多路召回门禁，同时等待 ready Endpoints 与 `/ping` 可访问（默认 60 秒），超时打印 Service/Endpoints/Pod 证据，避免把控制面 rollout 成功误当成 Service 已可用。下一步拉取后继续以 BUILD_IMAGES=0/IMPORT_IMAGES=0 重跑。
>
> 2026-08-08 F18 PaiRec observed Ready 去 CoreDNS 依赖：pause sandbox 与 pymilvus 注入修复后，vector/deepfm adapter 均达到 2/2 Running，但 `pairec-brpc-observed` 运行 6 分钟仍 0/1，当前模板的 Ready 和业务配置仍引用 adapter Service DNS 名称，而该集群 CoreDNS 长期不可用。部署器现于 adapter rollout 后读取两个 ClusterIP，与生成式 inference 一样把纯数字 endpoint 写入配置和 Readiness；渲染阶段断言 endpoint 精确匹配且无占位符残留。下一步拉取后以 BUILD_IMAGES=0/IMPORT_IMAGES=0 幂等重跑，预计只触发 PaiRec 配置 rollout。
>
> 2026-08-08 F18 pymilvus 运行时导出路径收口：首次真实导出确认健康 `dssm-recall` 容器内版本为 pymilvus 2.4.10/grpcio 1.67.1，但 `milvus-lite` distribution 元数据包含站点目录外的 `../../../bin/milvus-lite`，GNU tar 因路径穿越保护拒绝解包。导出器现对 root/source 执行真实路径解析，只允许 site-packages 根目录内文件进入归档并明确记录被跳过的外部 CLI；远程 Milvus 客户端不依赖该本地 CLI。下一步清空本次生成的临时目标目录后重跑导出，再继续 vector rollout。
>
> 2026-08-08 F18 首次远端部署阻塞修复：master 缺失 `pause-aarch64:3.8` 导致两个 adapter Pod 无法创建 sandbox，现已从 worker1 导入并恢复；DeepFM Pod 已 2/2 Running，vector Pod backend Ready 但 adapter 按严格门禁拒绝 Ready。日志确认根因不是 Milvus，而是 `pymilvus` 仅安装在既有 `dssm-recall` 运行容器、未固化进 `zcx-pairec-image:v1.1`。为避免重打/传输 121GiB 镜像，新增确定性依赖导出脚本，从健康容器提取 pymilvus/grpcio/ujson/milvus-lite 到 `/home/zcx/pairec-python-runtime`；vector 双容器 Pod 只读注入该目录并前置 PYTHONPATH，一键部署新增缺失门禁。下一步 master 拉取后执行依赖导出，重启 vector Deployment，确认 backend health `milvus=true` 和 Pod 2/2 Ready，再以 BUILD_IMAGES=0/IMPORT_IMAGES=0 重跑 smoke。
>
> 2026-08-08 F18 推荐全链路 BRPC 化与请求级可观测工程实现完成，待远端正式验收：新增统一 `pairec.pipeline_trace.v1` 协议、Controller 单一并发安全 TraceRecorder、整数微秒闭合、生成式/向量/DeepFM service trace、低基数 Prometheus 指标和一键 p50/p95/p99 汇总；向量召回与 DeepFM 分别新增独立 C++ BRPC adapter，采用 adapter + localhost Python backend 双容器 Pod，PaiRec 配置固定三路 BRPC、零重试且禁止 HTTP fallback。隔离 `pairec-brpc-observed` Deployment 保留原 HTTP 基线；Ready 同时依赖生成式 endpoint 与两个 adapter，vector adapter 仅在后端明确 `milvus=true` 时 Ready。DataSystem 不再从聚合日志猜测请求归因：显式相关 probe 可返回准确 Get/Set，原生 TRT KVC 未传播 request identity 时固定标记 `attribution_complete=false`，正式严格门禁需后续 native hook。验收脚本覆盖 1000 请求 trace 有效率>=99.9%、闭合误差、服务子阶段、Prometheus、HTTP/BRPC A/B、五容器 CPU throttling<=5%、restart/OOM/health。输入协议与 K8s/JSON/YAML/Shell/protoc 静态检查通过；Python/汇总 unittest 9/9、相关 Go race 7 包和完整 `./services/...` build 通过。远端前置条件是重建含 `recommend.proto` 30-39 字段的 TRT-LLM inference 镜像；旧 inference 会因不回显 request_id 被严格判为无效样本。下一步在 worker1 原地构建/导入新 inference 二进制，再在 master 执行 `REQUESTS=1000 REQUIRE_DATASYSTEM_ATTRIBUTION=0 bash scripts/deploy_and_validate_pairec_brpc_observed.sh`；拿到正式证据后再决定是否实现 native KVC request hook 并打开 DataSystem 严格门禁。
>
> 2026-08-08 F18 推荐全链路 BRPC 化与请求级可观测开工：基于 F17 已验收的生成式 + Milvus + DeepFM 链路创建 `pairec-brpc-observability` 分支。范围固定为 Controller 单一并发安全 TraceRecorder、整数微秒请求级闭合、低基数 Prometheus 指标、向量召回/DeepFM 两个独立 C++ BRPC adapter、零重试/无 HTTP fallback 的隔离实验 Deployment，以及 DataSystem 精确归因状态；当前进入协议和公共基础设施实现。
>
> 2026-08-08 F17 PaiRec 多路召回后 DeepFM 精排工程闭环完成：master 独立 `pairec-multi-recall-rank` Deployment/Service rollout 成功；engineering Rank 模型完成 size=50/10 各3次 smoke 和 size=10 连续100次稳定性，全部请求成功、分数降序且模型版本/角色一致。100次稳定性 E2E avg/p99=`107.720/121.109ms`，多路召回 avg/p99=`98.310/110.010ms`，Rank client avg/p99=`7.559/9.094ms`，Rank service avg/p99=`5.358/5.730ms`。旧部分词表模型观测到 item OOV=`76/5000`、category OOV=`503/5000`，engineering 模式按设计只记录不阻断；user/profile/gender/age/history 均无 OOV。故障注入确认 Rank 停止后 PaiRec 变为 NotReady，直连返回 `code=500/msg=deepfm rank failed/items=[]`，Rank 恢复后 Pod 重新 Ready，restart count=0，最终 `PAIREC_DEEPFM_RANK_VALIDATION_OK`。F17 按当前“流程跑通、不以推荐质量和零OOV为门槛”的工程范围完成；生产化仍需全量词表 DeepFM 与零 OOV 门禁验收。
>
> 2026-08-08 F17 PaiRec Rank 验收脚本 `set -u` 修复：独立 `pairec-multi-recall-rank` Deployment/Service 已成功创建并 rollout，但进入首个测试 phase 前，`run_phase` 在同一条 `local` 声明中用刚声明的 `phase` 计算 `phase_dir`，Bash 不保证该赋值在展开时可见，因而报 `phase: 未绑定的变量`。现将函数参数与派生路径拆成两条声明，并新增回归断言；`bash -n` 与 DeepFM unittest 7/7 通过。下一步 master 拉取后幂等重跑完整验收脚本。
>
> 2026-08-08 F17 engineering Rank Service 已在 master 启动并通过健康检查：`deepfm-rank` 监听 `127.0.0.1:18210`，返回 `code=200/status=healthy/backend=deepfm_rank`，模型版本=`c9d7ff83f6cc6893`、角色=`engineering`、checkpoint epoch=`10`、validation AUC=`0.715087`、validation logloss=`0.535314`；模型/词表规模与画像、类目产物均加载成功。下一步先执行真实特征 `/rank` 协议 100 次，再部署独立 `pairec-multi-recall-rank` 完成 3+3+100 E2E 与 fail-closed 故障注入。
>
> 2026-08-08 F17 Rank 容器 Python 模块路径修复：workdir 修正后容器进入 Python，但以 `/workspace/inference/deepfm_rank_server.py` 文件路径启动时 `sys.path` 不包含仓库根目录，导致 `ModuleNotFoundError: training` 并在 restart policy 下反复退出。入口现从 `/workspace` 使用 `python -m inference.deepfm_rank_server`，同时将 `/workspace` 前置到既有 `PYTHONPATH`；新增回归断言。下一步 master 拉取后重新运行启动脚本。
>
> 2026-08-08 F17 Rank 容器只读挂载启动修复：master 首次启动 `deepfm-rank` 在 OCI init 阶段失败，原因为基础镜像默认工作目录 `/workspace/pairec4tigerllm`，而脚本将仓库只读挂载到 `/workspace`，runc 无法在只读根下创建默认目录。`run_deepfm_rank_container.sh` 现显式设置已存在的 `--workdir /workspace`，保持仓库和模型目录只读；新增静态回归测试。下一步 master 拉取后重新启动 engineering Rank Service。
>
> 2026-08-08 DSSM all-rows 续训到 epoch 6 完成，确认停止继续堆 epoch：续训后的最佳 checkpoint 为 epoch 5；epoch 6 validation loss=`7.464948`，in-batch Recall@10/50/100=`5.367%/15.539%/23.575%`，相对 epoch 3 的 `4.468%/13.660%/21.202%` 明显上升。但固定 10,000-query 全库评估仅为 Recall@10/50/100=`0.27%/0.78%/1.38%`、MRR@100=`0.001204`；相对 epoch 3 的 `0.20%/0.79%/1.40%`，只有 Recall@10 和 MRR 小幅提升，Recall@50 少 1 个命中、Recall@100 少 2 个命中。由此排除“仅训练轮次不足”作为主因，并确认 batch 内指标与 235 万全库目标存在明显错位；当前产物继续保留作 A/B，不替换已部署 Milvus。下一步 DSSM 转为同类目/热门 hard-negative 实验，不再延长现有 all-rows 训练；全量 vocab/画像的覆盖价值独立于 DSSM 向量质量，可立即用于隔离重训 `deepfm_full_vocab_out`，同时推进 F17 工程验收。
>
> 2026-08-08 F17 DeepFM 精排工程链路在 DSSM 续训期间并行补强，待远端部署验收：Rank Service 新增 `engineering`/`production_candidate` 模型角色、checkpoint epoch/AUC/logloss、词表规模和 profile/category 数量健康证据；启动时严格核对 checkpoint `model_config.vocab_sizes` 与挂载词表，禁止模型/词表混用。每次 `/rank` 现返回 user/item/category/gender/age/history OOV、profile 缺失和分数分布，PaiRec 将同一证据写入请求级 trace，并要求响应模型角色与配置一致，否则 fail-closed。协议验收不再使用可能全 OOV 的固定 `1..50`，而是从模型产物选择画像完整的真实用户与 50 个 item/category 均命中的真实商品，连续 100 次必须零 OOV 且分数非退化；E2E 验收汇总真实召回候选 OOV，工程模型仅观测，`production_candidate` 默认开启零 OOV 硬门禁。当前旧词表 `deepfm_out` 明确只能以 `engineering` 角色运行；DSSM epoch 4-6 已确认全量词表可用但召回质量未突破，因此 DeepFM 可立即使用该全量词表独立重训，不需要等待 DSSM 向量上线。本地证据：DeepFM unittest 5/5、Python compile、Shell/JSON/YAML/diff check、Go race（sort/recall/web）和完整 services build 全部通过。下一步在 worker1 隔离训练 `deepfm_full_vocab_out`，同时在 master 用 engineering 模型执行 100 次真实特征协议、3+3+100 PaiRec E2E 与故障注入。
>
> 2026-08-08 DSSM all-rows 三轮远端结果完成并补齐续训能力：worker1 全量训练确认每 epoch 使用 `108,308,074` 个候选行，其中 `82,317,282` 个为 click=0 曝光，平均每 batch `3,692.95` 个唯一候选；epoch 3 仍为最佳 checkpoint，validation loss=`7.528249`、in-batch Recall@10/50/100=`4.468%/13.660%/21.202%`。固定 10,000-query、2,354,248 商品全库评估为 Recall@10/50/100=`0.20%/0.79%/1.40%`、MRR@100=`0.001154`，相对 positive-only 基线的 `0.14%/0.70%/1.14%` 与 `0.000924` 均名义改善，但 Recall@50 仅多命中 9 个 query，尚不能视为质量突破或替换 Milvus 的依据。由于最佳点仍在最后一个 epoch，先排除训练未收敛再引入 hard negatives。训练现支持 `LOAD_CHECKPOINT`，`EPOCHS` 明确定义为总目标 epoch，保留既有 history/best loss；新 checkpoint 每轮写 `dssm_last.pt` 并包含 AdamW state，可精确续训。现有 epoch-3 checkpoint 不含 optimizer，首次延长到 epoch 6 会明确记录 `optimizer_state=reset`，属于权重续训；worker1 路径自动转换为容器 `/workspace`。专项及关联测试 13/13、py_compile、Shell/JSON/diff check 通过。下一步使用原目录从 epoch 3 续到总 epoch 6、patience=3，再用相同 10,000 query 比较；若无持续改善，停止堆 epoch，进入同类目/热门 hard-negative 实验。
>
> 2026-08-07 DSSM 全 batch 候选训练目标已实现，待 worker1 远端全量 A/B：正样本行只负责提供 user query，默认 `candidate_mode=all_rows` 使用 batch 内所有有效 item（包括 click=0 曝光）作为候选；同 item 的所有候选列仍进入正样本集合，避免重复曝光形成假负样本。保留 `candidate_mode=positive_rows` 精确复现旧目标。训练摘要新增候选行、click=0 候选行和每 batch 唯一候选数证据，checkpoint 固化 candidate mode；validation in-batch Recall 同步改用一致候选口径。新流水线默认输出 `dssm_all_candidates_out`，不会覆盖 `dssm_full_out` 的 10,000-query 基线；worker1 包装器透传 `CANDIDATE_MODE`，文档提供隔离 A/B 命令。全库评估在未设置门槛时新增 `METRICS_RECORDED_NO_QUALITY_GATE` 分类和 `DSSM_FULL_CORPUS_EVALUATION_RECORDED` 标记，不再把执行成功表述为质量通过。微型真实 train -> export -> evaluate 与专项 6/6 测试通过，关联 DSSM/审计/DeepFM 共 13/13 测试通过；下一步在 worker1 默认运行 3 epochs、10,000-query 的 `all_rows` 实验，与固定 Recall@10/50/100=`0.14%/0.70%/1.14%` 基线比较，再决定是否引入同类目/热门 hard negatives。
>
> 2026-08-07 DSSM 全量重训远端质量复核完成，当前结论为“工程与覆盖通过，召回质量未通过替换门槛”：worker1 完成 3 epochs 全量训练，每轮 `108,308,074` train + `12,034,232` validation 样本，最佳 checkpoint 为 epoch 3；导出 `2,354,248` 个商品向量、`2,157,486` 个商品类别和 `999,447` 个用户画像，held-out 查询目标缺失数为 0。扩大到 10,000 个 held-out positive query 后，全库指标为 Recall@10=`0.14%`、Recall@50=`0.70%`、Recall@100=`1.14%`、MRR@100=`0.000924`；Recall@50 与此前 1,000-query 的 `0.70%` 完全一致，已排除主要由小样本波动导致的可能。脚本输出 `PASS` 仅表示 `min_recall_at_50=0` 时评估成功执行，不表示质量验收。根因候选已收敛到当前 in-batch loss 只保留 `click=1` 行，约 75.8% 的 `click=0` 曝光没有进入候选负样本，训练时每个 query 看到的候选规模也远小于 235 万全库。旧 Milvus 向量与现有 DeepFM 暂不替换；下一步先将 loss 改为“正样本 query 对 batch 全部 item 候选”，保留同 item 多正样本语义，并补充同类目/热门 hard negatives 后做隔离 A/B 重训与同口径 10,000-query 全库评估。
>
> 2026-08-07 DSSM 全量质量重训工程已实现，待 worker1 远端执行：旧 `dssm_out` 保留为工程链路基线，新流水线固定输出 `dssm_full_out`；`0` 明确定义为扫描/训练完整 CSV，词表、商品类别和用户画像均覆盖全文件。训练采用确定性 90/10 row split、每 epoch 流式 chunk 确定性打乱，并将同 batch 重复 item 作为多正样本，避免误当负样本；按 validation loss 保存最佳 checkpoint，同时记录 validation in-batch Recall@10/50/100。导出后新增分布于完整 CSV 的 held-out positive priority sample，对全部导出商品向量计算 full-corpus Recall@10/50/100 与 MRR@100，查询分批避免分数矩阵撑爆显存。新增 worker1 Docker 启动器和运行文档；本地真实微型数据已完成 train -> export -> full-corpus evaluate，DSSM 专项 unittest 4/4、关联测试 7/7、py_compile、Shell 与 diff check 通过。下一步在 worker1 默认跑 3 epochs/1000 个评估 query，首次结果建立质量基线；通过后用新 vocab 重训到隔离的 `deepfm_full_vocab_out`，旧 Milvus/Rank 在新产物验收前不替换。
>
> 2026-08-07 DSSM 审计容器启动修复：worker1 首次执行在 OCI init 阶段失败，原因为 `zcx-pairec-image:v1.1` 默认工作目录 `/workspace/pairec4tigerllm`，而审计脚本将仓库只读挂载到 `/workspace`，runc 因而无法在只读挂载内创建默认目录。`docker run` 现显式使用已存在的 `--workdir /workspace`，不修改镜像和 DSSM 产物；新增参数回归测试。本地 Docker daemon 不可访问，待 worker1 拉取后重跑形成真实审计报告。
>
> 2026-08-07 DSSM 既有训练审计工具已完成，待 worker1 实测：代码核对确认旧流水线固定只扫描/训练/导出前 `1,000,000` 行，batch=4096 时每 epoch 仅 244 个完整 batch，解释了 DSSM 相比本次 1.203 亿样本/epoch 的 DeepFM 明显更快；现有 DSSM checkpoint 仅记录 train avg loss，没有验证集 AUC/Recall@K，因此 `avg_loss=6.5701` 不能单独证明召回质量。新增 `scripts/audit_dssm_training_and_artifacts.sh` 一键复用 worker1 的 `zcx-pairec-image:v1.1`，检查 checkpoint/词表/向量维度与有限值、L2 归一化、导出 ID 一致性、向量塌缩、DeepFM 全量商品集合覆盖率，并默认抽查 CSV 前 500 万行的 user/item/history/category OOV；结构损坏返回 FAIL，产物完整但训练范围/质量证据不足返回 WARN。合成产物端到端 unittest 3/3、py_compile、Shell 语法通过。下一步在 worker1 运行审计，以报告决定是否必须用全量词表重训 DSSM；由于 DeepFM 复用了 DSSM vocab，若 OOV/覆盖率异常，DeepFM 也需随新词表重训。
>
> 2026-08-07 F17 DeepFM 远端全量训练完成，待服务部署验收：worker1 使用完整数据完成 10 epochs，末轮 train logloss=`0.514731`、validation AUC=`0.715087`、validation logloss=`0.535314`，每 epoch 为 `108,308,074` train + `12,034,232` validation 样本；最佳 checkpoint 为 epoch 10，六类产物均已写入 `deepfm_out`。在启动 Rank Service 前先审计其复用的 DSSM vocab 对完整数据的覆盖，随后执行独立 Rank 100 次协议验证、PaiRec 3+3+100 稳定性与故障注入验收。
>
> 2026-08-07 F17 DeepFM 精排工程实现完成，待远端全量训练与部署验收：新增 PyTorch DeepFM，严格使用 `user_id/item_id/video_category/gender/age/hist_1..hist_10` 与 `click`，复用 DSSM vocab/OOV=0，确定性 90/10 切分、最多 10 epochs、validation logloss early stop patience=2，并固定导出 checkpoint、词表、配置、训练摘要、商品类别和用户画像六类产物。新增独立 CPU `deepfm-rank` HTTP 服务，严格接收 50 个唯一候选并仅返回同集合分数；PaiRec 新增手动注册的 `DeepFMRankSort`，100ms/零重试、严格校验 request ID/model version/候选集合/finite score，按 DeepFM 分数稳定排序，同分保留召回顺序。框架默认吞 Sort error 的行为已通过受控 vendor controller patch 收口：任何 Rank 超时/协议错误均清空候选，HTTP 仍为 200，业务体固定 `code=500/msg=deepfm rank failed/items=[]`，不回退召回顺序。新增独立 `pairec-multi-recall-rank` Deployment/ClusterIP，原 F16 实例不变，readiness 强依赖 Rank health；新增训练、Rank 容器启动和一键部署验收脚本，验收覆盖 Rank 协议 100 次、size=50/10 各 3 次 smoke、size=10 的 100 次稳定性、召回/精排/E2E p50/p95/p99、至少一次重排、模型版本一致，以及停止 Rank 后 Pod NotReady + 直连 Pod 返回 code500 的故障注入并自动恢复。本地证据：500 行 Tenrec 格式数据完成 1 epoch 真实训练与六产物导出，Rank Service 50 候选 smoke 成功（forward 约 0.875ms），短数组/重复 ID 均拒绝；Python unittest 3/3、py_compile、Shell/JSON/YAML、Go race（sort/recall/config/web）与完整 Go build 全部通过。下一步在远程 GPU 执行全量训练，将 `deepfm_out` 放到 master，启动 `deepfm-rank:18210`，再运行一键脚本形成正式远端证据后才将 F17 标记 done。
>
> 2026-08-07 F16 多路召回远端验收完成：独立 `pairec-multi-recall` Deployment 经 1 次预热后，3 次 smoke 全部为 `2+48`，E2E avg/p99=`102.363/102.811ms`，生成式 avg/p99=`99.333/100.980ms`，Milvus avg/p99=`5.471/5.771ms`。随后 100 次稳定性请求全部有效，E2E avg/p99/max=`101.671/110.601/114.258ms`，生成式=`99.390/108.050/113.000ms`，Milvus=`5.217/7.853/7.924ms`；组合分布为 `1+49` 5 次、`2+48` 95 次，证明生成式至少 1 个、Milvus 补齐 50 的契约稳定生效。最终分类 `PAIREC_MULTI_RECALL_GENERATIVE_MINIMUM_OK`、status=PASS；F16 标记完成，下一任务转入精排接入。
>
> 2026-08-07 多召回有效性契约调整：远端 3 次 smoke 已证明 GenerativeRecall 与 MilvusRecall 并行执行，稳态生成式约 100ms、Milvus 约 5-7ms、组合耗时取两者最大值；首请求约 3.9s 来自 PaiRec 首次加载 `user_features.json`，不是 Milvus 时延。100 次稳定性测试在第 57 个样本因生成式只返回 1 个合法商品，被旧版固定 `2+48` 门禁误判失败；对应日志确认 `generative=1`、`milvus=50`、最终合并为合法 `1+49` 且总数 50。现策略改为生成式至少 1 个、最多 2 个，Milvus 按实际生成式数量补齐到 50；`0+50`、Milvus 无入选、总数不足或生成式异常超过 quota 仍标记 degraded/判失败。部署验证脚本新增默认 1 次不计入统计的预热，并统计 `1+49`/`2+48` 实际分布。本地证据：Shell/8 个内嵌 Python/JSON 契约检查通过，`go test -race -mod=vendor ./services/recall` 与 `go build -mod=vendor ./services/...` 通过。下一步远端重建并导入 PaiRec 镜像，重跑 3 次 smoke + 100 次稳定性，验收每次至少 1 个生成式、Milvus 补齐 50、两路 trace 完整且 Pod 不重启。
>
> 2026-08-06 多召回 smoke 证据链修复：远端第二轮已通过独立 Deployment rollout、Pod `1/1 Running`、挂载配置和 Service ready endpoint 门禁，并完成 3 个响应的 `generative_recall=2`、`milvus_recall=48` 前置校验；随后脚本因容器 stdout 缺少 `MilvusRecall source=milvus` 判定失败。根因是 GenerativeRecall 已使用 `writeTraceStdout`，而新增 MilvusRecall/QuotaMultiRecall 仅调用 PaiRec logger，`kubectl logs` 证据口径不一致。现两者成功路径均在 `PAIREC_TRACE_STDOUT=1` 时输出包含双 request ID、数量、source、2+48 合并统计和耗时的结构化 stdout trace；需重建 PaiRec 镜像后重跑。
>
> 2026-08-06 多召回首次部署后 Service 收敛竞态修复：远端已确认 `pairec-multi-recall` Pod `1/1 Running`、重启 0、挂载配置正确，但 Service 创建后脚本立即开始 smoke，ClusterIP 在 Endpoints/kube-proxy 尚未收敛时返回 connection refused。部署脚本现于 smoke 前等待 ready endpoint 出现且 ClusterIP `/ping` 返回 success，默认最多 60 秒；超时会自动输出 Service、Endpoints 和 Pod 诊断，不再把控制面传播延迟误判为业务失败。
>
> 2026-08-06 PaiRec 生成式 + Milvus 多路召回实验链路已实现，待远端验收：新增 `QuotaMultiRecall`，内部并行执行 `GenerativeRecall` 与 `MilvusRecall`，按 item_id 去重后固定选择 2 个生成式商品、用 Milvus 补齐到 50，Milvus 不足时再由生成式回填；子召回 panic 被局部恢复，原始 `retrieve_id` 保留。新增固定数字地址配置模板、独立 `pairec-multi-recall` Deployment/ClusterIP 和严格 Ready 门禁；生成式关闭缓存/HTTP fallback/重试，Milvus 超时 300ms、零重试。新增 `scripts/deploy_and_validate_pairec_multi_recall.sh`，自动发现 inference ClusterIP、校验 DSSM `milvus=true`、构建/导入最新 PaiRec 镜像、动态渲染配置，并依次执行 3 次 smoke 与 100 次稳定性验证；每次必须 code=200、50 个唯一商品、`generative_recall=2`、`milvus_recall=48`，且两路日志、BRPC 协议、Milvus source、Pod UID/重启数均通过门禁。实验实例单独启用 `PAIREC_ALLOW_PARTIAL_RESULTS=1`，向量召回中途失败时允许非空生成式结果以 code=200 降级返回，但严格实验样本仍判无效；原实例默认行为不变。远端首次预检发现 master 的 `http_proxy=141.5.133.189:3128` 会错误代理 `141.61.91.189/192.168.100.12`，而 DSSM 在 `0.0.0.0:18200` 与 `127.0.0.1/health` 均正常；脚本现对集群内 curl 强制 `--noproxy '*'`，实验 Pod 动态注入 inference/DSSM 数字地址的 `NO_PROXY/no_proxy`，Ready 探针主动清除代理变量，避免预检和业务调用被代理污染。本地证据：`go test -race -mod=vendor ./services/recall github.com/alibaba/pairec/v2/web`、`go build -mod=vendor ./services/...`、Shell/内嵌 Python/JSON/YAML 和 `git diff --check` 全部通过。下一步在 master 拉取后执行一键脚本，采集真实 3+100 请求的 E2E、生成式、Milvus 和组合召回 p50/p95/p99。
>
> 2026-08-06 CoreDNS ImagePullBackOff 修复脚本：确认集群 `coredns` 已持续 45 天处于 `0/1 ImagePullBackOff`，`kube-dns` Endpoints 只有 `notReadyAddresses`，因此 Pod 查询 `10.96.0.20:53` 返回 connection refused。新增 `scripts/repair_coredns_imagepullbackoff.sh`：备份现有 Deployment/Service/Endpoints，优先复用 master containerd 别名镜像或 Docker 本地镜像，其次执行限时 ARM64 拉取，再尝试从 worker1 containerd 传输；镜像就绪后将 Deployment 修正为 `docker.io/coredns/coredns:1.8.3` + `IfNotPresent`，并严格验证 rollout、ready endpoints 和内部 Service DNS。外部 DNS 单独报告，默认不阻止内部 DNS 修复；脚本不修改宿主 `/etc/resolv.conf`。下一步在 master 执行脚本并根据 summary 判断是否还需配置企业上游 DNS。
>
> 2026-08-06 DSSM 镜像 APT/DNS 阻塞绕过：`python:3.10-slim` 基础层历时约 71 分钟后已完整下载并缓存，后续失败发生在 build 容器无法解析 `deb.debian.org`。`curl` 只用于镜像健康检查，不是运行依赖，因此 `docker/Dockerfile.dssm_recall` 已移除 `apt-get`/Debian 镜像依赖，改用 Python 标准库 `urllib.request` 健康检查；构建现在仅需访问可配置的 PyPI 源。下一步使用 `--network=host` 和清华 PyPI 参数复用已缓存基础层重新构建。
>
> 2026-08-06 Milvus standalone 已跑通：embedded etcd + ConfigMap 固定 127.0.0.1:2379 方案验证成功，Pod `1/1 Running`，`/healthz` 返回 `OK`。下一步灌库并部署 DSSM 召回服务。
>
> 2026-08-06 DSSM ARM64 镜像国内源：master 从 Docker Hub 拉取 `python:3.10-slim` 约 30 分钟仍未完成，构建以 `context canceled` 结束。`docker/Dockerfile.dssm_recall` 现默认通过 DaoCloud 拉取 Docker Hub 基础镜像，使用清华 Debian/PyPI 镜像，并从 PyPI 安装具备 ARM64 wheel 的 `torch==2.1.2`，固定 `numpy==1.26.4`；所有源均保留 build-arg 回退入口。下一步在 master 重新构建、导入 containerd，再运行 Milvus loader。
>
> 2026-08-06 Milvus CrashLoop 第二轮定位与修复：完整日志确认 manifest 中 `ETCD_USE_EMBEDDED` 未被 Milvus 识别，应为 `ETCD_USE_EMBED`；修正后仍有 RootCoord/DataCoord 同时绑定 `19530`。结合 manifest 确认 ConfigMap 整目录挂载 `/milvus/configs`，遮蔽了镜像自带 `milvus.yaml`，导致各组件端口默认配置丢失。现改为 `embedEtcd.yaml` 与 `user.yaml` 两个 `subPath` 文件挂载，保留镜像默认配置；诊断脚本默认采集完整日志并新增 `MILVUS_CONFIG_DIRECTORY_MASKED_PORT_COLLISION` 分类。远端 apply 后已验收 RootCoord `53100`、DataCoord `13333`、Pod `1/1 Running`/重启 0、`/healthz=OK`；Milvus 部署阻塞解除，下一步灌入 DSSM item 向量。
>
> 2026-08-06 Milvus CrashLoop 诊断脚本：新增 `scripts/diagnose_milvus_crashloop.sh`，只读收集 Deployment/Pod/ConfigMap/Service/Node、事件、当前日志和 previous 日志，并对镜像架构、OOM、磁盘、权限、embedded etcd IPv6/未就绪、配置、拉镜像和健康探针问题输出明确 `classification`。本地 `bash -n` 和分类夹具验证通过；下一步在 master 执行脚本，以远端证据确定 Milvus 当前崩溃根因。
>
> 2026-08-06 PaiRec 多路召回（Milvus + DSSM）阶段性收工：PaiRec 侧 `services/recall/milvus_recall.go`、`services/main.go` 注册、`configmap-brpc.yaml` 多路召回配置已落地并推送；DSSM 训练/导出在 188 ARM 4090D 上全量跑通（10 epochs avg_loss=6.5701，275413 item 向量，7455 用户画像）；Milvus 部署探索了 embedded etcd（IPv6 localhost 解析问题）、三容器 standalone（etcd+minio+milvus）、以及回归单容器 embedded + ConfigMap 固定 127.0.0.1:2379，目前 Pod 仍 CrashLoopBackOff，待下次根据日志继续定位。新增 `docker/Dockerfile.dssm_recall`、`k8s/deployment-dssm-recall-server.yaml`、`k8s/job-load-milvus.yaml`、`scripts/run_dssm_train_and_export.sh`。代码已推送 gitcode `pairec-multi-recall-ranking`。
>
> 2026-08-05 PaiRec 多路召回（Milvus + DSSM）PaiRec 侧代码落地：新增 `services/recall/milvus_recall.go`，遵循 vendor `BaseRecall` + `RecallAlgo` JSON 配置模式，通过 HTTP 调用 `dssm_recall_server` 的 `/recall`，失败返回 nil 由 generative_recall 兜底；`services/main.go` 注册 `MilvusRecall` 分支；`k8s/configmap-brpc.yaml` 将 `milvus_recall` 加入 `home_feed` 召回路。新增 `docker/Dockerfile.dssm_recall` 与 `k8s/deployment-dssm-recall-server.yaml`，用于远端部署 DSSM 召回服务。本地验证：`go build -mod=vendor ./services/...`、`python3 -m py_compile` 覆盖 DSSM 训练/推理/Milvus 灌库脚本、`yaml.safe_load_all` 覆盖新增 K8s manifest 与 configmap 均通过。下一步远端全量训练/导出/灌库并部署，验证 PaiRec E2E code=200 且召回结果含两路 item。
>
# 工作进度

> 2026-08-05 不加压完整推荐链路 + KVC 3+2 基线跑通（分支 `pairec-multi-recall-ranking`）：新增 `k8s/deployment-inference-brpc-trtllm-25g.yaml`，固定 inference 在 worker1/188、DataSystem 走 25G master `192.168.100.12:18482`、scheduler=`max_utilization`、host_cache_size=100MiB，并去掉 ETCD pool。使用 `scripts/calibrate_brpc_kvc_cache_shape.sh` 校准出 `PRIME_REQUESTS=195`，每轮重启 inference 清 HBM 后连续 3 轮确认稳定 `offload=3 onboard=2`。基线指标：E2E avg/p99≈141.7/142.9ms、brpc/TCP≈2.3ms、KVC total≈34.2ms（offload≈21.0ms、onboard≈13.2ms）、server-other≈92.5ms。注意：3+2 不是单次 Recommend 天然固定触发，而是特定 HBM 容量/scheduler/prompt 历史/prime 次数共同形成的缓存状态，必须严格按 `RESET_INFERENCE_BEFORE_ROUND=1` + `PRIME_REQUESTS=195` 复现；默认 DataSystem pool worker（188:18481）仍报 `Worker not ready`，Set 会失败，不能用于该基线。

> 2026-08-05 分支切换：KVC burst wrapper 暂停在 `1eebb0c`，保留分支 `kvc-burst-wrapper-wip`；当前开发分支为 `pairec-multi-recall-ranking`，下一主线是接入 PaiRec 多路召回（Milvus 向量）和精排。

> 最后更新: 2026-08-04 | 当前状态: F14 已实现独立实验 PaiRec 和内嵌 BRPC burst coordinator；本地并发/race/构建验证通过，待远端按 c1 三次 smoke、c1000 三次/100次/1000次顺序验收。严格25G历史结果见 `docs/F14_STRICT_25G_PRESSURE_RESULTS_2026-08-02.md`；F15 NPU迁移记录继续保留。
> 最新补充(8/4-F14 KVC burst wrapper第一阶段): 已按dsbench一次性Get模型实现独立`cpp/kvc_burst`工程。`kvc_burst_wrapper`为C-1条压力lane建立长驻exclusive `KVClient`、独立预填充Key和常驻线程，每轮在计时窗口外按可复现seed随机打乱Key映射；`kvc_burst_business_probe`作为第C条模拟业务lane。两个进程使用固定ABI共享控制块和process-shared futex generation Barrier统一放行，默认10ms超时且业务Get仍继续；结果以同主机`CLOCK_MONOTONIC_RAW`区间计算真实峰值、start skew及与业务Get重叠数。最大并发固定100，结构门禁要求全部Get成功、`max_active>=ceil(C*95%)`且业务重叠压力数`>=ceil((C-1)*95%)`，时延只报告不设目标。新增构建、`1/10/100` benchmark和共享协议测试脚本及设计文档。本地严格C++编译、完整CMake双目标构建、futex跨进程放行/超时测试和带2ms模拟Get的三档各3轮验证均通过；c100每轮均为`max_active=100/overlap=99`。本地无真实DataSystem C++ SDK，因此尚未形成25G真实Get证据。下一步在worker1/DataSystem 0.8.1环境构建，先跑1792KiB的`1/10/100`三轮smoke，再跑17920KiB的`1/10`；通过后才进入同Pod独立sidecar和TensorRT首次真实Get hook第二阶段。
> 最新补充(8/4-F14 PaiRec内嵌burst): 原PaiRec保持不变，新增独立`pairec-brpc-wrapper` Deployment/ClusterIP，固定走`192.168.100.11:18103 -> 18100`、8 CPU Guaranteed、关闭结果缓存/HTTP fallback/重试。PaiRec调用层新增burst coordinator：N路session启动时严格预连接并统一屏障放行，确定性轮换其中1路执行真实Recommend，其余N-1路发送100KB Health；真实Recommend完成立即返回，压力尾部异步收口，压力失败不改变业务结果但令`burst_valid=false`；同一时刻只允许一个burst。新增c1/c1000独立ConfigMap、模式切换和端到端benchmark脚本，按request_id校验三类JSON事件、全部压力成功、Wrapper trace和Pod稳定，并输出E2E/front BRPC/inference/runner的p50/p95/p99。验证：`go test -race ./services/config ./services/recall`通过，PaiRec主程序离线链接通过，YAML/嵌套JSON、Shell/Python语法和`git diff --check`通过。远端下一步严格执行B(c1,3次smoke)，再执行C(c1000,3次smoke->100次初测->条件满足后1000次正式p99)。
> 最新补充(8/4-F14 PaiRec内嵌burst远端C组): c1以`SIZE=1`完成3/3 smoke，E2E p99=`112.908ms`、front BRPC p99=`2.318ms`、inference p99=`108.860ms`。c1000完成3次smoke及100次初测后，正式1000次全部成功；正式结果E2E avg/p99/max=`143.618/223.553/263.764ms`，front BRPC=`9.399/79.096/99.417ms`，inference=`111.767/132/137ms`，runner=`111.690/132/137ms`，Wrapper total=`112.053/132.010/138ms`。说明100KB×999路压力主要抬高PaiRec到Wrapper前段尾延迟，后端BRPC差值仍约1ms；推理/runner p99相对c1约增加23ms，存在轻度同进程资源污染，不能表述为完全无影响。1000/1000业务请求成功，阶段性BRPC wrapper目标完成；下一步进入独立KVC wrapper设计和实现，保留本结果作为BRPC基线。
> 最新补充(8/4-F14预连接实测与门禁修正): 18103 Wrapper预连接三轮中，各档`connected_sessions=N`且所有Recommend/Health成功。100档`max_active_min=54/100`、`start_skew p99=4.442ms`；1000档`max_active_min=775/1000`、`start_skew p99=50.289ms`，但业务`front_brpc p99=2.974ms`、runner p99相对并发1为-0.860ms，说明后端推理未被污染。1000路乘100KB约100MB，25Gbps纯串行发送下限约32ms，前发Health可在后续lane开始前完成，因此95%同时在途不是统一屏障burst的合理默认硬门禁。现默认`REQUIRE_CLIENT_ACTIVE_RATIO=0`，有效性改为预连接N路、armed=N、统一放行、全部RPC成功、Wrapper trace存在且Pod稳定；`max_active/start_skew`继续报告，专项目标可显式恢复active门禁。正式Recommend的lane按轮次确定性轮换，避免固定第一路的调度顺序偏差。下一步拉取Go脚本后无需重建或部署Wrapper，先跑3轮确认`PASS_SMOKE`，再跑30-100轮观察分布，最后仅对选定并发档采集足量p99样本。
> 最新补充(8/4-F14 Wrapper分发修正): 远端实测 `pairec-brpc-inference:k8s-arm64-v1` 镜像为130.68GB、docker save tar为122GB；通过管理网向188全量scp三小时仍未完成，且中断后远端同名tar与master SHA256不一致，禁止导入。Wrapper本身不需要重新分发整个基础镜像，新增 `scripts/ship_brpc_burst_wrapper_binary_to_worker.sh`：从master新镜像提取单个 `brpc_burst_wrapper` 可执行文件，经25G地址`192.168.100.11`复制到`/home/zcx/bin`，传输前后强制SHA256一致。188 Deployment改为通过单文件hostPath将该二进制挂到已有同ABI `pairec-brpc-inference:k8s-arm64-v1`运行时；部署脚本在apply前远程验收文件存在且可执行。下一步拉取本修正，执行二进制分发脚本后直接部署，不再传输122GB tar。
> 最新补充(8/4-F14 Wrapper首轮smoke): 独立Wrapper部署成功，本地Health为1ms，经Wrapper转发Recommend为211ms。三轮同步burst中全部业务与压力RPC成功，后端`runner_generate p99`从并发1的100.920ms到并发1000的101.920ms，仅增加1ms；前段BRPC p99则从4.751ms升至368.992ms，说明压力已集中在客户端到Wrapper入口，TRT后端基本未被污染。10/100/1000原先标记FAIL仅因`wrapper_max_active_total`为2/3/3，未达到95%硬门禁；该指标统计的是极短Health回调与Recommend回调的同时执行数，不是同步发送或入口排队请求数，因此门禁口径错误。现改为`REQUIRE_SERVER_WRAPPER=1`只强制Wrapper trace存在，服务端回调重叠峰值默认作为诊断；如确需该实验才显式设置`REQUIRE_SERVER_OVERLAP=1`。下一步更新脚本后重跑三轮，确认结构门禁PASS，再扩大样本。
> 最新补充(8/4-F14 burst预连接): 门禁修正后的第二轮三次样本中，1/10/100并发均PASS；1000档业务和999路Health仍全部成功，front BRPC p99为50.498ms、runner p99相对并发1增加7.060ms，但客户端实际峰值仅760/1000且start skew p99约53.6ms，因此按95%真实在途门禁正确判为FAIL。根因是burst统一放行后每路仍先建立TCP连接，1000路握手和Go调度扩散进入测量窗口。现将已有`BRPCRecommendSession.Connect`扩展到Recommend，burst默认`BURST_PRECONNECT=1`：为1路业务与N-1路压力分别预建TCP连接，全部连接成功后才统一放行RPC；JSON新增`preconnect/connected_sessions/preconnect_ms`，有效性要求每轮`connected_sessions=N`并继续要求实际max active达到95%。本地session Health+Recommend单连接race测试、probe race/build、shell与嵌入Python检查均通过。下一步仅拉取并重跑，不需要重建Wrapper镜像或重新部署。
> 最新补充(8/4-F14服务端BRPC burst wrapper): 新增独立 `brpc_burst_wrapper`，监听 `18103`。同步 burst 的 1 路正式 `Recommend` 由 wrapper 通过连接池转发到真实推理 `192.168.100.11:18100`，其余 `N-1` 路 100KB `Health` 在 wrapper 本地终止，不进入 TRT-LLM；因此可以对同一到达时刻下的前段 BRPC、wrapper 本地开销、后段 BRPC和真实 runner 时延分层观测，同时避免把所有压力请求直接打进推理进程。proto TraceInfo 新增23-29可选字段记录 wrapper 总耗时、后端 RPC、两段 BRPC 和业务执行期间服务端实际重叠峰值；Go手写proto、客户端trace和burst probe已同步兼容。`benchmark_go_brpc_burst_wrapper.sh` 新增 `REQUIRE_SERVER_WRAPPER=1` 严格模式，要求每轮wrapper trace存在、服务端重叠峰值达到配置比例、业务和压力请求全成功且inference/wrapper Pod不重启、无crash marker；报告不设置人为时延目标，继续输出inference/runner相对并发1基线的p99增量。新增188 hostNetwork/8 CPU Guaranteed Deployment和一键部署smoke脚本。验证证据：proto C++生成、YAML parse、4个shell `bash -n`、嵌入Python `py_compile`、`git diff --check`、`services/recall` race、probe定向race及Go build均通过；本机因缺少bRPC SDK不能链接C++，远端镜像构建已增加wrapper `ldd`硬门禁。下一步在master构建并传输ARM64镜像，在188部署`18103`，先跑`REPEATS=3` smoke，再根据服务端实际重叠度决定正式1000轮采样。
> 最新补充(8/3-F14 Health预连接同步探针): `BRPCRecommendSession.Connect` 现在可只建立TCP连接而不发送RPC；`probe_go_brpc_client.go --method=health --preconnect=true` 会为每个worker创建独立session，并发建立全部连接，确认全部成功后等待所有请求goroutine进入armed状态，再统一放行，每条连接只承载一个计时Health请求。该模式要求`REQUESTS=CONCURRENCY`且`QPS=0`；建链阶段只保留`connected_sessions`成功数，不统计建链时间、偏差或并发峰值，请求阶段输出`armed_workers/max_active/start_skew_us/request_total_ms`。可选`PRECONNECT_HOLD_MS`提供已建立连接的netstat观测窗口，该时间不计入请求延迟。`benchmark_go_brpc_payload_latency.sh` 增加`PRECONNECT=1`入口，并把连接数门禁和请求阶段证据写入summary JSON。本地`services/recall`单测和race、probe定向单测和race、probe build、shell语法及`git diff --check`均通过；尚未把本地静态证据当作远程并发结果。下一步从master/189向`192.168.100.11:18101`执行1000并发，在worker1同步观察ESTABLISHED峰值，并验收`connected_sessions=1000`、`armed_workers=1000`和请求阶段`max_active/start_skew_us`。
> 最新补充(8/3-F14 burst p99修正): 明确目标是测正式Recommend在同步burst中的BRPC p99，不是寻找并发100或30-40ms目标。`scripts/benchmark_go_brpc_burst_wrapper.sh` 已移除目标区间、自动细化和selected.env，默认测试10/100/1000并发并自动加入并发1无压力基线；各档按轮次交错执行以降低时间、温度和缓存漂移，每档默认1000轮，少于1000轮只标记`PASS_SMOKE`。报告对成功的正式请求分别输出client wall、服务端inference、TRT `runner_generate`和BRPC差值的p50/p95/p99/p999/max，并给出相对并发1基线的p99变化；请求失败另行计数。有效性门禁仅包含Recommend/Health全成功、实际max active达到95%、Pod UID/restart不变且无crash marker，不包含延迟目标。同实例Health压力可能竞争BRPC worker、CPU、内存和网络队列，因此通过inference与runner p99增量区分推理受扰和纯通信排队。本地Go build/单测/`services/recall`测试/race detector、shell/JSON校验及交错基线聚合夹具均通过；夹具3轮正确输出`PASS_SMOKE`和inference/runner基线增量。下一步在真实`192.168.100.11:18100`先以`REPEATS=3` smoke，再以`REPEATS=1000`正式采样。
> 最新补充(8/2-F14收工结论): 所有正式样本继续严格验收 `3 Set + 2 Get`、无重启和无crash。没有任何一组有效实验同时达到原定均值区间 E2E `320-360ms`、BRPC `40-50ms`、KVC `200-210ms`、server-other `80-90ms`。最接近的是17.5MiB大对象的4 Get+6 Set压力：10轮E2E avg/p95=`204.072/347.103ms`，KVC avg/p95/max=`90.791/206.359/219.082ms`，BRPC avg=`1.900ms`，server-other avg=`100.509ms`；它只在p95口径命中E2E和KVC目标。独立BRPC当前最强有效档为16 CPU c100，约`24.68k QPS/20.22Gbps`，但前台BRPC仍仅`2.333ms`。c1000、同cgroup双进程和shared-service分别暴露压力源吞吐下降、CPU限流和推理Pod CPU竞争，不能作为40-50ms纯BRPC时延证据。本轮停止继续加压及shared-c100；下一步先决定是否将验收改为固定负载下的p95上界，再决定是否继续实验，禁止sleep或人工时延注入。
> 最新补充(8/2-F14严格25G BRPC压力): 已恢复 `192.168.100.11:18100` 前台推理、独立BRPC压力目标和 `192.168.100.12:18482` DataSystem Worker，`PRIME_REQUESTS=195` 可连续保持前台严格 `3 Set + 2 Get`。同环境10轮baseline为E2E `143.355ms`、BRPC `2.600ms`、KVC `36.662ms`、server-other `91.638ms`。8 CPU单进程c100达到约 `21.4k QPS/17.5Gbps`；提高到request8/limit16后c100达到 `24.68k QPS/20.22Gbps`、限流 `2.486%`，前台E2E `166.556ms`、BRPC仍为 `2.333ms`、KVC `57.525ms`。同资源c1000反降至 `19.14k QPS/15.68Gbps`、限流 `6.648%`；同一16 CPU cgroup内双进程c100x2也只达到 `19.47k QPS/15.95Gbps`、限流 `11.295%`，证明增加并发或拆进程但不增加CPU预算都无法叠加吞吐。shared-service c10将100KB Health压力改发真实18100后，结构门禁虽PASS，但inference Deployment仅request2/limit4 CPU，三轮限流周期达到 `95.082%/98.347%/98.347%`，BRPC仍为 `1.333ms`、KVC `35.192ms`，server-other却升至 `121.474ms`；该结果明确属于推理Pod CPU竞争，不是BRPC通信压力，停止shared c100。当前network-only最强有效档仍是单进程16 CPU c100；下一步先检查worker1可分配CPU、已请求资源和inference/pressure cpuset，确认有余量后再将18101/18102拆为独立cgroup并增加总压力CPU预算，以真实BRPC流量逼近25G，同时继续用server-other和CPU限流门禁排除宿主CPU污染。目标仍按均值验收：E2E `320-360ms`、BRPC `40-50ms`、KVC `200-210ms`、server-other `80-90ms`，禁止sleep或时延注入。
> 最新补充(7/28-vLLM-Ascend NPU灰度链路): 训练 checkpoint 已导出为 Hugging Face 模型 `/data/models/qwen3_rec_vllm_ascend_v1`，校验结果为 tokenizer/embedding rows `152693`、semantic token `1024`、输入输出 embedding 共享权重、模型 reload logits finite、checkpoint SHA256 `bf65b92cab4ed131cbb975397e753ede1a36251a1ff163e079b7d64e7145f99e`。运行时固定使用 `quay.io/ascend/vllm-ascend:v0.19.1rc1` 和宿主 CANN `8.5.1`；物理 NPU Chip1 映射为容器逻辑 Device0，并绑定 NUMA6、CPU `144-167`。最终 ARM64 镜像为 `pairec-brpc-inference:vllm-ascend-npu-v1`，容器 `pairec-vllm-ascend-brpc-chip1` 已达到 `running/healthy/restarts=0`。直连 brpc Health 返回 `code=200/status=healthy/backend=vllm_ascend`；并发2/4严格 smoke 共20请求全部成功，每个请求返回8个合法商品，未出现 EngineCore、OOM 或重启。PaiRec 灰度容器通过 `/api/recommend` 完成 `HTTP -> PaiRec -> brpc -> vLLM-Ascend -> item` 闭环，重复单请求 E2E 约 `306.8-370.2ms`。独立长驻 OpenAI 服务的32-token单并发样本为 `TTFT avg=57.481ms`、`TPOT avg=8.048ms`、`E2E avg=306.963ms`。有效前缀约束采用离线紧凑索引、`valid_prefix + temperature=0`，当前验收目标是返回可映射合法商品；离线 test sequence 的 hit/MRR 仍为0，尚未形成推荐质量结论。
> 最新补充(7/28-vLLM-Ascend DataSystem边界): 当前 NPU 镜像内尚未安装版本匹配的 `openyuanrong-datasystem`，服务显式关闭 prefix cache，输入上限64 tokens、输出上限32 tokens；vLLM-Ascend 0.19.1 在 prefix cache/chunked prefill 场景使用128-token KV block。当前 prompt 小于一个完整 block，按现有 connector 的完整块传输语义，即使只补 SDK 和 endpoint，一次推荐预计仍是 `1次 brpc + 0次 DataSystem Set + 0次 DataSystem Get + 完整NPU推理`。不再把 TRT 运行态中的 `3 Set + 2 Get` 作为 NPU 迁移硬门槛。下一步先基于当前健康镜像构建 DataSystem 变体，安装与 Worker 匹配的 SDK，配置并验证 AscendStoreConnector/Yuanrong backend、健康检查、失败回退和运行指标；确认默认128 block的真实调用数后，再根据“必须观测KVC时延”的实验目标选择调整真实输入工作负载，或评估维护32-token block定制分支。禁止用测试专用 sleep/时延注入代替真实过载。
> 7/23 TRT/KVC历史状态: F14 PaiRec Go brpc client 端到端验证通过；DataSystem pool 已在 worker1 与 master 上达到 Pod Running，但完整 ServiceDiscovery 接入暂存为阻塞任务：当前 `zcx-pairec-image:v1.1` 内的 `yr.datasystem` SDK 缺少 `ServiceAffinityPolicy` 与 `yr.datasystem.service_discovery`；brpc + DataSystem pool 基线摸测已固化为并发脚本 `scripts/benchmark_brpc_datasystem_pool_baseline.sh` 和单请求分解脚本 `scripts/trace_single_brpc_datasystem_request.sh`。单请求脚本已跑出当前 PaiRec Go brpc 链路口径：一次 PaiRec 请求触发 1 次 brpc `Recommend`，C++ brpc server 约 107ms，client E2E 约 114-118ms，KVC/DataSystem 本轮观测为 2 次 offload、0 次 onboard；指标口径已修正：`brpc-inference latency_ms` 是 C++ inference 服务端推荐总耗时，不是 PaiRec 到 inference 的纯 brpc/TCP 通信耗时。跨节点验证当前切到 `brpc-cross-node-189-ds188` 分支的反向形态：`189 PaiRec -> brpc/TCP -> 188 inference -> 189 DataSystem`；远端 apply 后 Pod placement 已符合预期，inference 日志确认 `Rank 0 is using GPU 0` 且 DataSystem 固定到 `141.61.91.189:18481`。当前 `189 -> ClusterIP -> 188 inference` 的 Go brpc 探针已证明可用，PaiRec fallback 数据同步脚本也已补齐。探针基线已跑出：synthetic 50/200 请求均为 `brpc_calls_per_probe=1`、`offload_set_per_probe=1`、`onboard_get_per_probe=0`，`brpc_comm_est_ms p99≈3ms`，`offload.set_ms p99≈318ms`；真实用户 pressure/replay 测得 `offload_set_per_probe≈2.99`、`onboard_get_per_probe≈0.09`，KVC Set/Get 单次 p99 均约 315-318ms，瓶颈在跨节点 DataSystem Set/Get，不是 brpc 通信。已生成 `docs/BRPC_KVC_CROSS_NODE_LATENCY_REPORT_2026-06-27.md` 汇总数据、口径、结论和下一步三段式强 onboard 方案。
> 最新补充(7/23-dsbench真实并发观测v4): 17.5MiB×10与1.75MiB×100旧结果只证明worker启动，不能证明replay窗口内存在对应数量的在途DataSystem RPC；同时旧mixed脚本给GET和SET各分配完整key_count，所谓175MiB实际为约350MiB。新增增量补丁 `datasystem-dsbench-sustained-observability.patch`：dsbench输出GET/SET各自的calls/errors/QPS/Gbps/max_inflight，并用prepared/start/ready三阶段文件实现统一放行。benchmark每轮改为“背景对象预填充并停门→重启inference→prime前台3 Set+2 Get→启动BRPC压力→放行KVC→稳定后replay”，避免prime后背景预填充淘汰前台KV。矩阵现在精确拆为17.5MiB的4 GET keys+6 SET keys和1.75MiB的40+60 keys，两档总常驻量均为175MiB。严格门禁要求GET/SET分别calls>0、errors=0、QPS>0且max_inflight达到4/6或40/60；ready_workers不再作为并发证据。正反夹具验证：4/6峰值PASS，GET峰值降为3即FAIL；两阶段假dsbench验证start前无RPC统计、放行后ready_workers=10。当前待188对0.8.1源码应用增量补丁、增量编译并重新生成wrapper，再先跑KVC c10实测。
> 最新补充(7/23-dsbench Get生命周期诊断): 17.5MiB×10远端首跑中，4路Get和6路Set均已进入prepared，但放行后Get返回`Cannot get objects from worker`，底层仍按60秒超时。新增`scripts/diagnose_datasystem_sustained_get_lifecycle.sh`，在master一条命令完成188预填充、reset前Get探测、inference重启与prime、prime后Get探测、10路放行和证据回收，以`before_reset/after_prime/PASS`区分key失效阶段。竞争脚本现会把上层300秒超时传给远端wrapper；wrapper也会识别`/proc`中的僵尸子进程并立即报告真实退出码，不再等待到超时。当前下一步是在master运行该诊断脚本，根据`diagnosis.txt`决定是否调整背景对象生命周期或继续修放行路径。

> 最新补充(7/22-指定压力矩阵): 按新口径将BRPC压力固定为100KB payload并支持10/100/1000并发。为避免1000并发下短连接堆积TIME_WAIT和源端口耗尽，新增 `BRPCRecommendSession`，压力probe可通过 `--reuse_connections=true` 为每个worker复用一条TCP连接；业务默认短连接路径不变。probe新增 `max_active` 和ready文件，竞争脚本只有在实际同时在途数达到配置并发后才执行前台replay。KVC按要求切回DataSystem自带dsbench、固定 `batch_num=1`：17.5MiB使用 `17920KB/client_num=10/num=10`，1.75MiB使用 `1792KB/client_num=100/num=100`，两档驻留对象总量均约175MiB。新增 `scripts/benchmark_brpc_kvc_pressure_matrix.sh` 统一运行五档。验证：长连接单测确认两次Health只Accept一条TCP连接，`go test -mod=vendor ./services/...`、Go probe build、三个Shell脚本bash语法、五档假执行传参和 `git diff --check` 均通过。
> 最新补充(7/22-压力矩阵首跑修正): BRPC 100KB × 10并发首轮实际进入replay，三次前台E2E为 `144.437/163.934/158.239ms`、server为 `137/156/152ms`，说明压力已生效；但三轮缓存形态均为 `3 Set + 0 Get`，不满足总控脚本继承的严格 `3+2` 条件，因此 `valid=0/3`，不能直接和旧 `3+2` baseline做严格A/B。矩阵现已新增同环境baseline，默认 `MATRIX_STRICT_COUNTS=0`，将真实Set/Get数作为报告字段而非矩阵中止条件；可设置为1恢复严格形态验收。单档失败也会继续执行剩余档位，最后统一汇总，避免首档不匹配导致整套压力矩阵提前退出。
> 最新补充(7/22-压力矩阵v2首轮): 同环境baseline三轮E2E平均 `108.359ms`、server `100.333ms`、KVC `13.591ms`，均为 `3 Set + 0 Get`。BRPC 100KB压力下，10并发E2E平均约 `144.089ms`；100并发约 `368.659ms`；1000并发约 `495.892ms`，且100/1000档出现显著BRPC、KVC和server-other尾延迟。原汇总中的 `pressure_qps/gbps/active=0` 是只读取KVC persistent stats的观测缺口，不代表BRPC未达到并发；现已让Go probe按秒写BRPC calls/errors/QPS/Gbps/max_active并纳入汇总。dsbench 17.5MiB×10在 `num=100` 预填充约1.71GiB时明确返回 `Out of memory * 8`，不是单纯ready超时；两档num进一步降为10/100，使驻留对象总量均约175MiB。
> 最新补充(7/22-真实过载实验v3): 明确禁止测试延迟注入，目标通过真实BRPC/TCP和DataSystem竞争逼近 `E2E 320-360ms / BRPC 40-50ms / KVC 200-210ms / server-other 80-90ms`。新增 `scripts/calibrate_brpc_kvc_cache_shape.sh`，对 `PRIME_REQUESTS=192..200`逐项执行“重启inference→固定prime→replay”，候选值首次命中后还必须连续3轮保持 `3 Set + 2 Get`，才导出可source的 `selected.env`；压力矩阵恢复 `MATRIX_STRICT_COUNTS=1`且默认每轮重置inference，样本同时验收Pod不变、restart_delta=0和无crash marker。DataSystem侧新增 `datasystem-dsbench-sustained-pressure.patch`及应用脚本：在原dsbench中复用KVClient和固定key集合，越过线程barrier后写ready，并在duration内持续执行真实API；矩阵KVC两档由Get-only改为3:2 mixed，分别为17.5MiB的6 Set+4 Get和1.75MiB的60 Set+40 Get。新增188独立BRPC压力Deployment，压力走 `192.168.100.11:18101`，正式推荐保持 `18100`，避免Health压力直接占用TRT进程；combined模式会分别验收BRPC和KVC实际并发。新增目标分布评估器和中文执行指南。本地证据：相关shell `bash -n`、Python `py_compile`、Go services测试、YAML/JSON解析、DataSystem干净源码patch-check、假dsbench mixed active=10（含help非零兼容）、缓存形态单轮/三轮确认门禁、矩阵10/100/combined传参及目标评估器均通过。当前仅完成代码和本地验证；下一步需在188编译新dsbench、部署18101压力Pod、远端校准3+2后按baseline→BRPC→KVC→combined实测。
> 最新补充(7/22-dsbench wrapper): DataSystem `0.9.1` dsbench 与当前 `0.8.1` Worker 协议不兼容；基于 `0.8.1` 源码重编译的持续模式 dsbench 已完成单次 Set 冒烟验证。新增 `scripts/create_datasystem_dsbench_wrapper.sh`，自动发现构建依赖、校验无缺失动态库、强制核对版本 `0.8.1` 和持续模式参数，并生成固定 `LD_LIBRARY_PATH` 的 `/home/zcx/bin/dsbench-v081-sustained`。下一步在188运行20秒 mixed 冒烟，再校准前台 `3 Set + 2 Get` 并执行压力矩阵。
> 最新补充(7/22-KVC稳态压力工具): `KVC_GET_CLIENTS=8/16` 的旧dsbench结果均未显著抬高前台时延；源码确认dsbench的 `--num=256` 是全进程总key数，增加client只会重新分摊同一批key，脚本还会在每轮重新WarmUp和初始化客户端。新增 `scripts/datasystem_kv_pressure.cpp`：每个线程长驻复用独立KVClient、持续执行单key Get/Set，ready前强制验收 `max_active_calls` 达到配置并发，并按秒输出calls/errors/QPS/Gbps/平均与最大时延。远端188真实链接和三轮 `kvc-get/get_clients=16` 已通过：`max_active_calls=16`，应用层有效载荷吞吐约 `696-739 QPS/20.4-21.7Gbps`，每轮前台请求均准确观测 `3 Set + 2 Get`。相对baseline，平均E2E由 `117.984ms` 增至 `120.838ms`（+2.4%），KVC由 `22.430ms` 增至 `27.439ms`（+22.3%），BRPC仍为 `0.667ms`；但KVC增量主要由第2轮 `40.662ms` 尾延迟贡献，另两轮仅 `20.648/21.007ms`，当前只能证明会出现KVC竞争尾延迟，三轮样本不足以证明稳定均值回退。注意工具的Gbps是成功对象字节/QPS计算的应用层有效载荷，不等同于网卡计数器的物理线速。
> 最新补充(7/21-25G自然构造3 Set + 2 Get): 在历史提交 `ba07448`、`189 PaiRec -> ClusterIP/brpc -> 188 inference -> 192.168.100.12:18482 DataSystem` 拓扑下，关闭人工 `trt_datasystem_mget_probe` 后，先用真实用户集合 `5,6312,130,2184,303,1190,1191,1192,1193,1194` 串行发送 `REQUESTS=200/CONCURRENCY=1/HISTORY_MAX_LENGTH=20/VARY_USER_ID=true` 的 Go brpc 压力请求，再立即通过 `USER_ID=5 SIZE=1` 执行一次完整 PaiRec replay，基本可以稳定自然构造单请求 `offload/Set=3`、`onboard/Get=2`。压力轮 `200/200` 成功，平均每请求 `offload=2.98/onboard=1.90`，25G 单块 `Set p50=3.210ms/p99=4.928ms`、`Get p50=2.405ms/p99=3.562ms`，brpc 通信差值 p99 `3ms`。随后 replay 实测 E2E `126.248ms`、server `118ms`、brpc 差值约 `3ms`；3 次 offload total 合计 `22.646ms`，2 次 onboard total 合计 `7.476ms`，KVC 合计 `30.122ms`。该 `3+2` 是当前 32-token block、`64 input + 32 generation`、有限 HBM KV 容量及压力后的 LRU/secondary 命中共同形成的可复现运行态结果，不是协议硬保证；Pod重启、缓存残留、用户集合或容量参数变化后需重新做 pressure -> replay 验证。
> 最新补充(7/21-BRPC+KVC全25G): 将 brpc endpoint 改为 worker1/188 的 25G 地址 `192.168.100.11:18100`，DataSystem 继续使用 master/189 的 `192.168.100.12:18482`。同样的 200 请求 pressure 全部成功，平均每请求 `2.99 Set + 1.96 Get`；brpc RPC/server 平均分别为 `103.165/102.685ms`，通信差值平均 `0.480ms`、p99 `1ms`。单块 Set 平均 `3.311ms`、Get 平均 `2.835ms`，按实际调用次数折算每请求 KVC 平均 `20.139ms`，server 内其他阶段约 `82.546ms`。随后 replay 准确观测 `3 Set + 2 Get`：E2E `124.072ms`、PaiRec RPC `119ms`、server `118ms`、brpc 差值 `1ms`；3 次 offload total 合计 `22.441ms`，2 次 onboard total 合计 `7.394ms`，KVC 合计 `29.835ms`，server 内非 KVC 阶段 `88.165ms`，PaiRec 外层开销 `5.072ms`。相比 ClusterIP/brpc 的上一轮 replay，E2E 从 `126.248ms` 降至 `124.072ms`，主要来自 brpc 差值从 `3ms` 降至 `1ms`；KVC 基本持平。
> 最新补充(7/21-BRPC/KVC竞争压测工具): 新增 `scripts/run_datasystem_dsbench_pressure.sh`，用于在 188 上以 `3584KB`、`batch_num=1` 持续产生单 key Get/Set/mixed 压力，支持独立 client 数、CPU taskset、预填充、PID/ready 文件及退出清理。新增 `scripts/benchmark_brpc_kvc_contention.sh`，在保持前台推荐单并发的前提下，支持 `baseline/brpc/kvc-get/kvc-set/kvc-mixed/combined` 六种模式；每轮先执行 200 请求 pressure 重建缓存状态，再启动 BRPC Health payload 或远程 dsbench 压力，执行一次 PaiRec replay，严格验收 `3 Set + 2 Get`，并汇总 E2E/server/BRPC/KVC/server-other/PaiRec 外层时延和 25G 网卡字节增量。Go BRPC probe 新增 `--quiet` 以避免长时间 Health 压力产生大量客户端日志。本地验证：Go build、`go test -mod=vendor ./services/recall`、相关 `bash -n`、`git diff --check`通过；假 dsbench mixed 生命周期和模拟 `3+2` JSON 汇总均通过。当前仅完成工具和本地校验，远程 188/189 25G 压力矩阵尚未执行。
> 最新补充(7/21-BRPC压力源端口耗尽修复): baseline 3 轮全部通过且均为 `3 Set + 2 Get`，平均 E2E `117.984ms`、BRPC `0.667ms`、KVC `22.430ms`、server-other `87.570ms`。首次 BRPC 压力试验在第 2 轮 prime 报 `dial tcp 192.168.100.11:18100: connect: cannot assign requested address`；源码确认 Go BRPC client 每次 RPC 都 `DialContext` 新建连接并 `Close`，无节流 Health 负载会在 189 积累 `TIME_WAIT` 并耗尽临时源端口，不是 inference/KVC 故障。Go probe 已新增全局 `--qps`限流；竞争脚本默认改为 `300 QPS + 8MiB payload + concurrency=16`，用约 19.2Gbps 理论应用层流量制造 BRPC/25G 压力，同时控制新建连接速率。脚本同步新增 prime 最多 3 次重试、轮间冷却以及中途失败时的部分结果汇总。
> 最新补充: master/189 的 Calico 路由已恢复，master host 访问 worker1 PodIP `172.16.235.138:18100` 与 ClusterIP `10.107.65.159:18100` 均已连通；master 上 pause sandbox 镜像与 PaiRec 镜像导入流程已脚本化。当前已验证 `PaiRec -> brpc -> inference-brpc-kvc-probe -> datasystem_kv_probe`：一次 PaiRec 请求触发 `MCreate(4) + MSet(4 buffers) + Get(vector<4 keys>)`，日志显示 `user=6312 object_count=4 total_bytes=14680064 mset_ms=1256 mget_ms=1252 found=4 latency_ms=2513 backend=datasystem_kv_probe`。注意该链路仍是 KVC probe，不是真实 `trtllm_cpp` 推理路径；下一步应切回 `inference-brpc-trtllm`，重新采真实 TRT-LLM + KVC offload/onboard 单请求 trace。
> 最新补充(7/2): 仓库进度对齐完成。当前分支为 `brpc-cross-node-189-ds188`，调研开始时工作区干净；本次只更新 `progress.md` 与 `feature_list.json`。`HANDOFF.md` 停留在 6/25，后续应以本文件 7/1-7/2 记录为准。`feature_list.json` 当前仍保留 F12/F13/F14 为 pending，其中 F14 已有 6/23 brpc E2E 通过证据，但下一步已从 CoreDNS/镜像脚本化推进到“切回真实 `inference-brpc-trtllm` 链路并重新采 `trtllm_cpp` + KVC offload/onboard trace”。注意：本分支未跟踪 `docs/BRPC_KVC_CROSS_NODE_LATENCY_REPORT_2026-06-27.md`，如需引用该报告需补回文件或改用现有 `docs/BRPC_DATASYSTEM_POOL_BASELINE_PLAN.md`。
> 最新补充(7/3): master 新 Pod 创建过程中连续暴露 pause sandbox 镜像缺失、Calico `calico/cni` 镜像拉取证书问题与 PaiRec server 镜像拉取超时。新增统一入口 `scripts/sync_master_runtime_images.sh`，默认从 worker1/188 的 `k8s.io` containerd namespace 同步 pause + Calico 镜像到 master/189，并默认导入 master 本地 `/home/zcx/pairec-server-k8s-arm64-brpc-v1.tar`；可用 `EXTRA_IMAGES` 追加 nvidia-device-plugin 等运行时镜像，也可用 `LOCAL_IMAGE_TARS` 追加本地 tar。旧的 `scripts/sync_pause_image_to_master.sh` 和 `scripts/sync_calico_images_to_master.sh` 保留为兼容 wrapper，且保持各自只同步 pause/Calico。验证：三个脚本 `bash -n` 通过，新增脚本已设置可执行权限，`git diff --check` 通过。
> 最新补充(7/3-真实链路 E2E Set/Get): 明确当前任务不是旁路 `datasystem_kv_probe`，而是在真实 `PaiRec -> brpc -> inference-brpc-trtllm -> backend=trtllm_cpp` 推荐链路内同步构造 DataSystem 写读，并让这部分耗时进入端到端时延。`brpc_inference_server` 保留兼容开关 `--trt_datasystem_mget_probe=1`，默认关闭；开启后 `trtllm_cpp` 后端每次 Recommend 在完成 TRT runner 和 item mapping 后逐对象执行 `Create + Set` 4 次，再逐对象执行 `Get` 4 次，日志输出 `method=TrtllmDatasystemSetGet object_count=4 set_call_count=4 get_call_count=4 found=4 backend=trtllm_cpp`。该同步阶段会写入 response trace 的 `kv_write_ms/kv_lookup_ms/kv_source=trtllm_cpp_datasystem_set_get`，并参与 `backend_total_ms`，因此客户端 `scripts/trace_single_brpc_datasystem_request.sh` 打印的 `E2E client_ms` 会包含这 8 次 DataSystem API 调用。两个 cross-node manifest 已开启该开关；下一步远端重建/导入 brpc TRT-LLM inference 镜像，apply cross-node manifest，使用 `SIZE=1` 与具体 `BRPC_TARGET=pod/inference-brpc-trtllm-...` 采一次真实链路证据。
> 最新补充(7/3-build fix): 远端构建 `build_brpc_trtllm_inference_image.sh` 时 CMake 停在 DataSystem SDK 探测阶段，报 `DataSystem C++ headers were not found. Set DATASYSTEM_INCLUDE_DIR.`。根因是实际头文件在 `/usr/local/lib/python3.11/site-packages/yr/datasystem/include/datasystem/kv_client.h`，而脚本没有传默认 `DATASYSTEM_INCLUDE_DIR`，CMake hints 也未包含该路径。已为 TRT-LLM brpc 构建脚本补默认 `DATASYSTEM_INCLUDE_DIR` 与 `DATASYSTEM_LIBRARY`，并在 CMake 自动探测里加入 Python site-packages 下的 DataSystem include 路径。
> 最新补充(7/6-诊断脚本): 本轮真实 TRT-LLM brpc Pod 的 `Unknown argument: --trt_datasystem_mget_probe=1` 根因不是镜像 tag 或 imageID 错，而是 Deployment 残留历史 `brpc-dev-bin` hostPath，把 `/opt/pairec-brpc/bin/brpc_inference_server` 覆盖成旧开发二进制；移除该 overlay 后 Pod 已能 `1/1 Running`，日志确认 TensorRT-LLM、DataSystem KV 与 `trtllm_datasystem_set_get_probe` 初始化成功。增强 `scripts/diagnose_pairec_brpc_route.sh`：现在会输出 latest Pod/imageID/containerID/state、检查 `brpc-dev-bin` 覆盖挂载、在 Pod 内用 `strings`/`--help` 自检实际二进制是否包含 `trt_datasystem_mget_probe`/`TrtllmDatasystemSetGet`，并从 PaiRec Pod 对 configured endpoint、Service DNS、FQDN、ClusterIP、Endpoint PodIP 做 TCP 矩阵检查。当前下一步是用该脚本定位 PaiRec request 仍显示 `brpc_calls=0` 的原因：若 inference 无 `method=Recommend` 日志，则问题在 PaiRec -> Service/Endpoint 的配置、DNS、TCP 或 Go brpc client 调用路径，而不是 inference server 或 DataSystem 初始化。
> 最新补充(7/7-MSet/MGet): 真实 `backend=trtllm_cpp` 推荐链路内的 DataSystem probe 已从逐 key `Create + Set + Get` 循环改成批量接口口径：一次 `MCreate(keys[4])`、一次 `MSet(buffers[4])`、一次 `Get(keys[4], out_buffers, timeout)`，其中 SDK 的 vector `Get` 即当前可用的 MGet 形态。日志方法名改为 `method=TrtllmDatasystemMSetMGet`，同时输出 `mcreate_call_count=1/mset_call_count=1/mget_call_count=1` 与 `set_buffer_count=4/get_key_count=4/found=4`，避免把“batch API 调用次数”和“key/buffer 个数”混在一起。`trace_single_brpc_datasystem_request.sh` 与 `diagnose_pairec_brpc_route.sh` 已同步识别新日志名和新字段；本地验证 `bash -n` 与 `git diff --check` 通过，真实 ARM/GPU/SDK 编译和 K8s 验证需重建 brpc inference 镜像后执行。

## 时间线

| 日期 | 进度 |
|------|------|
| **2026-08-06** | **Milvus standalone 部署成功：embedded etcd + ConfigMap 方案 Pod `1/1 Running`，healthz 返回 OK；准备灌库与部署 DSSM 召回服务。** |
| **2026-08-06** | **PaiRec 多路召回阶段性收工：PaiRec 侧代码与 K8s manifest 推送；188 全量 DSSM 训练/导出完成（275413 item 向量）；Milvus standalone 部署探索 embedded etcd / 三容器 / ConfigMap 固定 127.0.0.1 三种方案，当前 Pod CrashLoopBackOff 待日志定位。** |
| **2026-08-05** | **PaiRec 多路召回（Milvus + DSSM）PaiRec 侧代码落地：`services/recall/milvus_recall.go`、`services/main.go` 注册 `MilvusRecall`、`configmap-brpc.yaml` 增加 `milvus_recall`；新增 DSSM 召回服务 Dockerfile 与 K8s Deployment/Service；本地 go build / py_compile / YAML parse 通过，远端 E2E 待执行。** |
| **7/23** | **dsbench真实负载观测v4完成本地实现：新增prepared/start门控和GET/SET分项calls/QPS/Gbps/max_inflight；压力预填充移到inference重启与prime之前；修正mixed对象总量翻倍；严格门禁不再用ready线程冒充在途RPC。两阶段与正反汇总夹具通过，远端0.8.1增量编译和25G复测待执行。** |
| **7/22** | **真实过载实验v3完成本地实现：严格校准3 Set+2 Get；dsbench新增长驻固定key持续模式；KVC改为3:2 mixed；BRPC压力拆到188独立18101进程；combined双压力、restart/crash和目标时延门禁已补齐。远端构建与25G实测待执行。** |
| **7/22** | **压力矩阵v2跑出同形态baseline和BRPC 10/100/1000结果，E2E平均约108/144/369/496ms。修复BRPC压力QPS/Gbps/active汇总缺口；KVC 17.5MiB预填充60秒未ready，已将超时提升为300秒并降低单轮预填充总量。** |
| **7/22** | **BRPC 100KB × 10首跑证明压力生效，但实际缓存形态变为3 Set + 0 Get，旧3+2严格门禁导致矩阵首档退出。矩阵已改为先跑同环境baseline、默认记录实际调用数但不强制3+2，并在单档失败后继续后续档位。** |
| **7/22** | **压力矩阵调整为指定口径：BRPC 100KB × 10/100/1000并发，压力worker复用TCP并严格验收实际在途数；KVC切回dsbench，测试17.5MiB × 10和1.75MiB × 100，均为batch_num=1。新增五档矩阵脚本，本地回归通过，远端待跑。** |
| **7/22** | **KVC长驻SDK压力工具远端验证通过。188到189 DataSystem的16路Get达到 `max_active_calls=16` 和约 `696-739 QPS`；三轮前台均保持 `3 Set + 2 Get`，E2E平均较baseline增加2.4%，KVC平均增加22.3%，但增量集中在单轮尾延迟。下一步扩大重复数，并分别比较get/set/mixed。** |
| **7/21** | **BRPC 竞争压测首轮定位并修复短连接源端口耗尽。根因为 Go BRPC client 每 RPC 新建/关闭 TCP，无节流 Health 负载产生大量 `TIME_WAIT`；新增 `--qps`，竞争脚本改用 `300 QPS + 8MiB payload`，并增加 prime 重试、轮间冷却和部分结果汇总。baseline 3/3 PASS，平均 E2E/BRPC/KVC 分别为 `117.984/0.667/22.430ms`。** |
| **7/21** | **新增 BRPC/KVC 竞争压测工具。`run_datasystem_dsbench_pressure.sh` 在 188 上使用 DataSystem 自带 `dsbench_cpp`持续生成 Get/Set/mixed 负载；`benchmark_brpc_kvc_contention.sh` 在 189 上统一编排缓存 pressure、BRPC Health 负载、远程 KVC 负载和 PaiRec replay，支持严格 `3 Set + 2 Get` 过滤与分阶段聚合。本地 Go build/test、shell 语法、模拟 dsbench 生命周期及汇总器通过；远程实验待执行。** |
| **7/21** | **25G跨节点真实KVC pressure -> replay构造验证通过。复现顺序：先执行 `REQUESTS=200 CONCURRENCY=1 TOPK=1 TIMEOUT_MS=120000 HISTORY_SOURCE=user_features UIDS=5,6312,130,2184,303,1190,1191,1192,1193,1194 USER_FEATURES_PATH=/home/zcx/workspace/pairec4tigerllm/data/user_features.json SEMANTIC_MAP_PATH=/home/zcx/workspace/pairec4tigerllm/data/tenrec/processed/semantic_id_map.json HISTORY_MAX_LENGTH=20 VARY_USER_ID=true bash scripts/benchmark_go_brpc_probe_kvc_latency.sh`，再执行 `USER_ID=5 SIZE=1 TIMEOUT=30 bash scripts/trace_single_brpc_datasystem_request.sh`。压力轮 `200/200` 成功，随后单请求稳定观测到 `3 Set + 2 Get`；E2E `126.248ms`、server `118ms`、KVC `30.122ms`、BRPC通信差值约 `3ms`。边界：这是自然缓存状态构造，不是固定调用协议，复测必须保持25G endpoint `192.168.100.12:18482`、关闭人工probe，并按 pressure -> replay 顺序执行。** |
| **7/7** | **真实 `trtllm_cpp` 推荐链路内 DataSystem probe 改为 MSet/MGet 批量接口。`RunDatasystemSetGetProbe()` 现在构造 4 个 key 后执行 1 次 `MCreate(keys, sizes)`、1 次 `MSet(buffers)` 和 1 次 vector `Get(keys, out_buffers, timeout)`，日志改为 `method=TrtllmDatasystemMSetMGet`，输出 batch call count 与 key/buffer count：`mcreate_call_count=1/mset_call_count=1/mget_call_count=1/set_buffer_count=4/get_key_count=4/found=4`。`scripts/trace_single_brpc_datasystem_request.sh` 增加 `trt_datasystem_mset_mget_probe_events` 与 mset/mget 字段摘要，诊断脚本同步匹配新方法名。验证：`bash -n scripts/diagnose_pairec_brpc_route.sh scripts/trace_single_brpc_datasystem_request.sh`、`git diff --check` 通过；远端下一步需重建/导入 brpc inference 镜像并复跑真实 PaiRec 请求。** |
| **7/6** | **补齐 PaiRec -> brpc inference 路由诊断脚本。`scripts/diagnose_pairec_brpc_route.sh` 新增 latest Pod/runtime identity、imageID/containerID/state、`brpc-dev-bin` hostPath overlay 检查、Pod 内实际 `brpc_inference_server` binary marker 自检、PaiRec 到 configured endpoint/Service DNS/FQDN/ClusterIP/Endpoint PodIP 的 TCP 矩阵、以及 `Unknown argument`/`method=Recommend`/`method=TrtllmDatasystemSetGet` 日志筛选。该脚本用于复盘并避免再次被 image tag 正确但二进制被 hostPath 覆盖的问题误导；当前真实链路剩余排查重点是 PaiRec 请求 `brpc_calls=0`，需确认 PaiRec 是否真正连到 `inference-brpc-trtllm:18100` 并触发 C++ `Recommend`。验证：`bash -n scripts/diagnose_pairec_brpc_route.sh` 通过。** |
| **7/3** | **合并 master 运行时基础镜像同步脚本。新增 `scripts/sync_master_runtime_images.sh`，统一处理 `pause-aarch64:3.8`、Calico DS/Deployment 镜像、master 本地 PaiRec server tar 和可选 `EXTRA_IMAGES` 的离线导入；默认会从 worker1→master 同步 pause/Calico，并导入 `/home/zcx/pairec-server-k8s-arm64-brpc-v1.tar`，解决 PaiRec Pod `ImagePullBackOff` 拉取 `docker.io/library/pairec-server:k8s-arm64-brpc-v1` 超时。脚本支持 `LOCAL_IMAGE_TARS`/`LOCAL_IMAGE_CHECKS` 扩展本地 tar，支持 `RESTART_KUBELET`、`RESTART_CONTAINERD`、`DELETE_MASTER_CALICO_PODS`、`DELETE_NON_RUNNING_PODS`。`scripts/sync_pause_image_to_master.sh` 与 `scripts/sync_calico_images_to_master.sh` 改为 wrapper，并显式 `SYNC_LOCAL_TARS=0` 以保留旧脚本语义。验证：`bash -n` 覆盖三个脚本，`git diff --check` 通过。** |
| **7/3** | **真实推荐链路内 DataSystem 4*Set + 4*Get E2E 构造能力落地。`cpp/brpc_gateway/brpc_inference_server.cpp` 为 `backend=trtllm_cpp` 保留兼容开关 `--trt_datasystem_mget_probe=1`，开启后同一次真实 Recommend 会先走 TRT-LLM C++ runner 和 semantic map，再构造 4 个 `3670016B` 对象逐个执行 `Create + Set`，随后逐 key 执行 4 次 `Get`；日志字段包含 `object_count/create_call_count/set_call_count/get_call_count/found`，用于证明一次请求内确实发生 4 次 Set 与 4 次 Get。该同步阶段写入 `kv_write_ms/kv_lookup_ms/kv_source=trtllm_cpp_datasystem_set_get` 并计入 `backend_total_ms`，所以客户端 E2E 包含这部分耗时。两个 cross-node manifest 已显式开启该开关；`scripts/trace_single_brpc_datasystem_request.sh` 新增 `trt_datasystem_set_get_probe_events` 汇总，同时保留旧 JSON key 兼容。验证：`bash -n scripts/trace_single_brpc_datasystem_request.sh`、`python3 -m json.tool feature_list.json`、两份 cross-node YAML `safe_load_all`、`git diff --check` 通过；真实 C++ 编译和远端 GPU/SDK runtime 验证待重建镜像后执行。** |
| **7/3** | **修复 brpc TRT-LLM inference 镜像构建的 DataSystem SDK 路径默认值。远端 CMake 配置阶段报 `DataSystem C++ headers were not found`，实际 SDK 头文件路径为 `/usr/local/lib/python3.11/site-packages/yr/datasystem/include`。`scripts/build_brpc_trtllm_inference_image.sh` 现在默认导出 `DATASYSTEM_INCLUDE_DIR=/usr/local/lib/python3.11/site-packages/yr/datasystem/include` 和 `DATASYSTEM_LIBRARY=/usr/local/lib/python3.11/site-packages/yr/datasystem/lib/libdatasystem.so`；`cpp/brpc_gateway/CMakeLists.txt` 的 `find_path` 同步增加 site-packages include hint。验证：相关脚本 `bash -n` 与 `git diff --check` 通过；远端需重新执行镜像构建确认。** |
| **7/2** | **仓库调研/进度对齐完成。当前分支为 `brpc-cross-node-189-ds188`，调研开始时本地 `git status --short` 干净；`progress.md` 比 `HANDOFF.md` 更新，真实下一步是从临时 `inference-brpc-kvc-probe` 切回 `inference-brpc-trtllm`，恢复/确认 PaiRec brpc endpoint 后重跑 `scripts/trace_single_brpc_datasystem_request.sh`，采集真实 `backend=trtllm_cpp` 下 brpc、TRT-LLM 和 KVC offload/onboard 单请求 trace。本机验证：`feature_list.json` JSON parse 通过；`python -m py_compile` 覆盖 `scripts/benchmark_e2e_latency.py`、`scripts/benchmark_trt_runner_samples.py`、`scripts/verify_datasystem_layers.py`、`inference/trt_llm/server.py`、`inference/trt_llm/trt_qwen3_backend.py`、`training/decoder/train.py`、`training/decoder/qwen3_generative_rec.py` 通过；`bash -n` 覆盖 brpc trace/benchmark/apply/KVC probe 与 7/1 新增脚本通过；`go test -mod=vendor ./services/...`、`go build -mod=vendor ./services/...`、`go build -mod=vendor -o /tmp/probe_go_brpc_client ./scripts/probe_go_brpc_client.go` 均通过。按速查命令做 Python import 时，本机因缺 `tensorboard` 在 `training/__init__.py -> training.rqvae.train -> torch.utils.tensorboard` 处失败，因此当前本机只能判定语法与 Go 链路可编译，训练/推理导入仍需补本机依赖或走远端 runtime。** |
| **7/1** | **阶段性收口：master 基础网络与 KVC probe 链路验证完成。master/189 Calico 恢复后，`ip route get 172.16.235.138` 已走 `tunl0 via 141.61.91.188`，master host 到 worker1 PodIP 与 `inference-brpc-kvc-probe` ClusterIP 均可 `nc` 连通；之前的 timeout 根因是 master Calico 路由缺失。master 新建 Pod 的 pause 镜像与 PaiRec 镜像缺失问题分别通过 `scripts/sync_pause_image_to_master.sh` 和 `scripts/import_pairec_server_image_to_k8s.sh` 固化处理。已将 PaiRec endpoint 临时切到 `inference-brpc-kvc-probe`，并验证 PaiRec 请求 `request_id=9ea63e7e-d55f-4b4e-bba7-71d891e71c8e` 返回 `code=200`，KVC probe 日志同步出现 `method=KvcMSetMGetProbe user=6312 object_count=4 object_bytes=3670016 total_bytes=14680064 mset_ms=1256 mget_ms=1252 found=4` 与 `method=Recommend ... backend=datasystem_kv_probe latency_ms=2513`。口径确认：这不是 4 次独立 Get RPC，而是 1 次同步 `KVClient::Get(vector<4>)`；开启 `enable_worker_worker_batch_get=true` 后 worker-worker 为 1 次 batch remote get 携带 4 个 subrequest，当前 TCP + 4 subrequest 源码分支为串行处理。该验证不是完整真实推理路径，下一步需切回 `inference-brpc-trtllm` 并重新统计真实 `trtllm_cpp` 推理链路中的 brpc、TRT-LLM 和 KVC offload/onboard 耗时。** |
| **7/1** | **新增 `scripts/import_pairec_server_image_to_k8s.sh`，用于 master 本地已有 `/home/zcx/pairec-server-k8s-arm64-brpc-v1.tar` 时，一键导入 `docker.io/library/pairec-server:k8s-arm64-brpc-v1` 到 `k8s.io` containerd namespace。脚本会同时用 `ctr`/`crictl` 校验镜像可见性，并支持 `DELETE_NON_RUNNING_PODS=1` 清理当前非 Running 的 PaiRec Pod 触发重建。** |
| **7/1** | **新增 `scripts/sync_pause_image_to_master.sh`，专门处理 master 新建 Pod 时反复 `failed to get sandbox image docker.io/library/pause-aarch64:3.8` 的问题。脚本默认从 worker1/188 的 `k8s.io` containerd namespace 导出 pause 镜像，复制并导入 master 本机 `k8s.io` namespace，同时用 `ctr`/`crictl` 双视角检查本地是否可见；可选 `RESTART_KUBELET=1`、`RESTART_CONTAINERD=1`、`DELETE_NON_RUNNING_PODS=1` 做运行时刷新和坏 Pod 清理。** |
| **7/1** | **新增 `scripts/diagnose_node_to_service_network.sh`，用于优先排查 master host 到 worker1 PodIP/ClusterIP 不通的问题。脚本默认只读采集：自动发现目标 Service/Pod/ClusterIP/PodIP，分别检查当前 host 到 PodIP/ClusterIP 的 TCP、路由、Calico 链路、kube-proxy iptables/ipvs 规则、目标 brpc Pod 自测，以及 PaiRec Pod 到目标的 TCP；输出按结果给出下一步判断：Pod 自测通过但 host 失败则定位到 node host dataplane，PodIP 失败优先刷新 Calico，PodIP 通但 ClusterIP 失败优先刷新 kube-proxy。** |
| **7/1** | **修复 KVC MSet/MGet probe 部署脚本的 DataSystem 预加载链：`scripts/k8s_apply_brpc_kvc_probe.sh` 新增 `KVC_PROBE_LD_PRELOAD`，默认加载 `/opt/pairec/lib/block_ds_consumer.so`、`/opt/pairec/lib/stub_gpu.so` 和 DataSystem `libabseil_dll.so.2407.0.0`，并在 apply 输出中展示该值。该修复复用此前 TRT-LLM DataSystem runtime 的 workaround，避免 DataSystem SDK 在 ARM 上解析空 GPU 标识时触发 `basic_string::substr(4)` 崩溃；如需排障可显式 `KVC_PROBE_LD_PRELOAD=""` 覆盖。** |
| **6/29** | **DataSystem MSet/MGet 简单探针版已实现：`brpc_inference_server` 新增可选 `backend=datasystem_kv_probe`，通过 `MCreate + MSet(buffers)` 写入批内 4 个对象，再用 `Get(vector)` 批量读取 4 个对象；默认对象数 `4`、对象大小 `3670016B`、TTL `120s`，可用 `--kvc_probe_object_count/--kvc_probe_object_bytes/--kvc_probe_ttl_sec` 覆盖。该后端只做 CPU buffer 模拟，不进入 TRT/GPU copy，服务端日志输出 `method=KvcMSetMGetProbe object_count/object_bytes/mcreate_ms/fill_ms/mset_ms/mget_ms/found`，用于先单独验证 DataSystem 批量接口和调用次数口径。构建层新增 `PAIREC_ENABLE_DATASYSTEM_KV_PROBE`，`scripts/build_brpc_trtllm_inference_image.sh` 默认打开该开关；新增 `scripts/k8s_apply_brpc_kvc_probe.sh` 独立部署 `inference-brpc-kvc-probe`，不覆盖现有 `inference-brpc-trtllm` 主链路；新增 `scripts/test_brpc_kvc_mset_mget_probe.sh` 发 brpc 请求并回看探针日志。本地验证：`bash -n`、`git diff --check` 通过；本机 CMake 仍因缺 brpc SDK 按预期无法配置，需在 ARM brpc/DataSystem runtime 镜像中完成编译验证。** |
| **6/29** | **阶段性做减法：DataSystem worker pool / ServiceDiscovery 暂时退出当前主线，不再作为 brpc/KVC 基线摸测目标。当前主线统一按固定 endpoint 口径：`189 PaiRec -> brpc/TCP -> 188 inference -> 189 DataSystem(141.61.91.189:18481)`；inference manifest 继续不设置 `DATASYSTEM_ETCD_ADDRESS`，避免进入 pool 选点逻辑。现有 pool 脚本、文档和 patch 保留为后续能力储备，但后续测试报告只按“单 DataSystem worker 固定 TCP endpoint”解释 brpc 通信时延、KVC Set/Get 次数和阶段耗时。注意：如果要清理远端 `datasystem-pool-*` Pod，必须先确认有单 worker DataSystem 正在对应节点监听 `18481`，否则当前 inference 的 KVC endpoint 会被删断。** |
| **6/27** | **PaiRec 完整召回链路支持 100KB brpc payload：`GenerativeRecallConfig` 与 `RecallAlgo` 新增 `brpc_payload_bytes`，默认 0；`TRTLLMClient.Recommend` 在 brpc 协议下把该值写入 `RecommendRequest.PayloadPaddingBytes`，HTTP JSON 路径继续忽略；`GenerativeRecall` stdout trace 增加 `brpc_payload_bytes` 字段，`scripts/trace_single_brpc_datasystem_request.sh` 会展示该字段。新增 `scripts/k8s_patch_pairec_brpc_payload_bytes.sh`，可安全 patch live ConfigMap 内转义的 `RecallAlgo` JSON 并重启 PaiRec，设置 102400 或恢复 0。验证：`bash -n`、`go test -mod=vendor ./services/...`、`go build -mod=vendor ./services/...`、`git diff --check` 均通过。** |
| **6/27** | **新增 Go brpc probe 并发能力：`scripts/probe_go_brpc_client.go` 增加 `--concurrency`/`CONCURRENCY`，Health 和 Recommend 两条路径都可按固定 in-flight 数并发发送，输出仍按 index 排序，summary 记录 concurrency。`scripts/test_go_brpc_client_probe.sh`、`scripts/benchmark_go_brpc_payload_latency.sh`、`scripts/benchmark_go_brpc_probe_kvc_latency.sh` 同步透传 `CONCURRENCY`。并发 Recommend 场景下，KVC 汇总脚本不再依赖 server log 顺序配对，优先用同一 brpc response 的 `latency_ms - inference_ms` 计算 `brpc_comm_est_ms`，避免并发日志乱序造成通信时延误判。验证：`bash -n` 三个脚本、`go build -mod=vendor ./scripts/probe_go_brpc_client.go`、`go test -mod=vendor ./services/recall`、`git diff --check` 均通过。** |
| **6/27** | **新增 100KB brpc payload latency 探针能力：`services/recall` 的手写 brpc protobuf 增加 probe-only `payload_padding` 字段，`RecommendRequest.PayloadPaddingBytes` 不进入 HTTP JSON；`scripts/probe_go_brpc_client.go` 新增 `--payload_bytes`/`PAYLOAD_BYTES`，Health 与 Recommend 都可携带 padding。新增 `scripts/benchmark_go_brpc_payload_latency.sh`，默认 `PAYLOAD_BYTES=102400`、`REQUESTS=200`，通过 Health RPC 把 100KiB padding 发到 `inference-brpc-trtllm`，跳过 TRT/KVC，输出 brpc payload RPC 的 avg/p50/p95/p99/p9999/max；既有 `scripts/benchmark_go_brpc_probe_kvc_latency.sh` 同步支持 `PAYLOAD_BYTES`，可测 Recommend 业务链路下 `client - server_method` 的通信上界。验证：`go test -mod=vendor ./services/recall`、`go build -mod=vendor ./scripts/probe_go_brpc_client.go`、三个脚本 `bash -n`、`git diff --check` 均通过。** |
| **6/27** | **生成 brpc + KVC 跨节点时延基线报告：新增 `docs/BRPC_KVC_CROSS_NODE_LATENCY_REPORT_2026-06-27.md`。报告汇总 synthetic、user pressure、user replay 三组数据：synthetic 200 请求 `brpc_comm_est_ms p99=3ms`、`offload_set_per_probe=1`、无 onboard；user pressure 500 请求 `offload_set_per_probe=2.990`、`onboard_get_per_probe=0.084`、`server_method_ms p99=1351ms`、`offload.set_ms p99=317.7ms`、`onboard.get_ms p99=317.2ms`；user replay 200 请求 `offload_set_per_probe=2.995`、`onboard_get_per_probe=0.090`、`server_method_ms p99=1352ms`、`offload.set_ms p99=317.7ms`、`onboard.get_ms p99=314.8ms`。结论：当前 brpc/TCP 通信 p99 稳定约 3ms，跨节点 DataSystem/KVC Set/Get 是主要耗时；尚未复现目标 `KVC Get * 2`，下一步需要 seed/evict/replay 三段式构造强 onboard 场景。** |
| **6/27** | **probe brpc/KVC 基线结果：`REQUESTS=50/200 TOPK=1` synthetic 模式均只观测到 offload，没有 onboard。200 请求结果为 `go_probe_brpc_rpc_ms p99≈401ms/p9999≈411ms`、`server_method_ms p99≈399ms/p9999≈408ms`、`brpc_comm_est_ms p99=3ms`；KVC `offload.set_ms p99≈317.8ms/p9999≈318.0ms`，`onboard.get_ms` 无样本。结论：当前链路的 brpc/TCP 通信开销约 3ms，上界稳定；服务端耗时主要由跨节点 DataSystem Set/offload 贡献，synthetic 探针不能覆盖 KVC Get/onboard。已扩展 `scripts/probe_go_brpc_client.go`、`scripts/test_go_brpc_client_probe.sh`、`scripts/benchmark_go_brpc_probe_kvc_latency.sh`，支持 `HISTORY_SOURCE=user_features`、`UIDS`、`USER_FEATURES_PATH`、`SEMANTIC_MAP_PATH`、`HISTORY_MAX_LENGTH`，下一步用真实用户 history 跑 pressure/replay 两段式基线，专门观察 `onboard_get_per_probe` 与 `onboard.get_ms p99/p9999`。** |
| **6/26** | **新增探针式 brpc/KVC 跨节点时延脚本：`scripts/probe_go_brpc_client.go` 支持 `--requests` 在单进程内连续发送请求，避免 `go run` 编译开销混入单次延迟；`scripts/test_go_brpc_client_probe.sh` 同步支持 `REQUESTS`。新增 `scripts/benchmark_go_brpc_probe_kvc_latency.sh`：自动读取 `inference-brpc-trtllm` ClusterIP，启动 inference 日志采集，运行 Go brpc probe，并汇总 probe stdout 与 brpc inference/DataSystem 日志。输出口径：`go_probe_brpc_rpc_ms` 是 Go brpc client 调用 wall-clock，`server_method_ms` 是 C++ brpc server method latency，`brpc_comm_est_ms` 是两者差值上界，包含 Go encode/decode、brpc framing、TCP/CNI/kube-proxy 和 server method 外排队；KVC 口径来自 TRT-LLM C++ `[Datasystem][TRACE]`，按 per-block 汇总 offload/onboard 的 p50/p95/p99/p9999/max，并输出 `offload_set_per_probe`、`onboard_get_per_probe`、`datasystem_access_per_probe` 作为每次 brpc Recommend 的 KVC 访问次数口径。本地 `gofmt`、`bash -n` 与 `go build -mod=vendor ./scripts/probe_go_brpc_client.go` 通过。** |
| **6/26** | **跨节点 PaiRec 请求未进入 inference 的根因定位完成：Go brpc client 探针成功，inference 日志出现 `[brpc-inference] method=Recommend user=go_brpc_probe code=200 items=8 latency_ms=89 backend=trtllm_cpp`，证明 `189 -> ClusterIP -> 188 inference` 的 Go brpc/TCP 通道正常。PaiRec `/tmp/recall_debug.log` 显示 `getUserHistory error: user 6312 not found in fallback`，PaiRec 日志显示 master/189 上 fallback 只加载了 2 个用户。结论：当前 `code=299` 不是 brpc/RPC/DataSystem 问题，而是 PaiRec 固定到 master 后 hostPath `/home/zcx/workspace/pairec4tigerllm/data/user_features.json` 与 worker1 不一致。新增 `scripts/sync_pairec_fallback_features_to_master.sh`，默认从 `141.61.91.188:/home/zcx/workspace/pairec4tigerllm/data/user_features.json` 同步到 master 同路径，先备份目标文件，再重启 PaiRec，因为 fallback 数据在进程内缓存。** |
| **6/26** | **补充 Go brpc client 独立探针：新增 `scripts/probe_go_brpc_client.go` 和 `scripts/test_go_brpc_client_probe.sh`。诊断已推进到新阶段：PaiRec ConfigMap 已改为 `brpc_endpoint=10.96.15.101:18100`，PaiRec Pod 内 `nc` 显示 TCP open，inference Pod 内 C++ brpc smoke 对 `127.0.0.1:18100` 返回 Recommend OK；但 PaiRec `/api/recommend` 后 inference 仍只有 smoke 的 `method=Recommend`，没有 user=`6312`。该探针复用 `services/recall` 中同一套 Go 手写 baidu_std brpc client，直接从 master host 访问 inference Service ClusterIP，区分“Go brpc client 协议/响应解析问题”和“PaiRec 召回流程没有走到 client 调用”。** |
| **6/26** | **跨节点路由诊断结果明确：`scripts/diagnose_pairec_brpc_route.sh` 输出显示 inference Pod 内 direct brpc smoke 成功，`brpc_inference_server` 返回 Health OK 且 Recommend user=`brpc_smoke_1` code=200/items=5/latency_ms=222；但 PaiRec `/api/recommend` 返回 `code=299` 后，inference 近 5 分钟日志只有 smoke 的 `method=Recommend`，没有 user=`6312` 的调用。PaiRec Pod 内 `/etc/resolv.conf` nameserver 为 `10.96.0.20`，`nslookup` 对该 DNS 超时并出现 `Message too large`，说明当前断点是 PaiRec 到 inference Service DNS 解析失败，而不是 inference/TRT-LLM/DataSystem 服务端故障。已修复诊断脚本对转义 `RecallAlgo` 的 `brpc_endpoint` 解析，并新增 `scripts/k8s_patch_pairec_brpc_endpoint_to_cluster_ip.sh` 自动将 `brpc_endpoint` 改为 inference Service ClusterIP，重启 PaiRec 以绕过 CoreDNS。** |
| **6/26** | **新增跨节点 PaiRec -> brpc inference 路由顺序诊断脚本：`scripts/diagnose_pairec_brpc_route.sh`。脚本会依次打印 Pod 拓扑、DataSystem pool、Service/Endpoints、Deployment nodeName、inference 的 `DATASYSTEM_*` env 与 args、PaiRec 配置内 `brpc_endpoint`，再从 PaiRec Pod 内检查 `/etc/resolv.conf`、Service DNS 与 TCP 连通性；可选执行 `scripts/test_brpc_native_inference_smoke.sh` 直连 inference Pod 的 `127.0.0.1:18100`，再发送一次 PaiRec `/api/recommend` 请求，并收集 inference `method=Recommend`、KVC/DataSystem trace、PaiRec stdout 和 `/tmp/recall_debug.log`。本地 `bash -n` 通过，目标是定位当前 `189 PaiRec -> 188 inference -> 189 DataSystem` 形态下 PaiRec 返回 `code=299` 且 inference 无 `method=Recommend` 的断点。** |
| **6/26** | **反向跨节点拓扑进入 inference 启动参数兼容问题：当前 `189 PaiRec -> 188 inference -> 189 DataSystem` 已调度成型，`pairec` 在 master/189 Running，`inference-brpc-trtllm` 在 worker1/188 CrashLoop；日志显示镜像内 `/opt/pairec-brpc/bin/brpc_inference_server` 不认识 `--trt_max_batch_size=1`，Usage 中也没有 `--trt_max_num_tokens`。这是远端镜像内 brpc 二进制版本与清单参数不一致导致。已从 `k8s/deployment-inference-brpc-trtllm.yaml`、`k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml`、`k8s/deployment-inference-brpc-trtllm-cross-node-188-ds189.yaml` 中移除 `--trt_max_batch_size` 与 `--trt_max_num_tokens`，先兼容当前已导入镜像；若后续需要容量参数，应重建并导入支持这些 flags 的 brpc inference 镜像。** |
| **6/26** | **因 189 GPU 不再作为 inference 资源，新增反向跨节点验证方案：目标拓扑改为 `189 PaiRec -> brpc/TCP -> 188 inference -> 189 DataSystem`。新增 `k8s/deployment-inference-brpc-trtllm-cross-node-188-ds189.yaml`：保持 `inference-brpc-trtllm` Service/Deployment 名称，`nodeName=worker1` 固定推理到 188，`DATASYSTEM_HOST=141.61.91.189`/`DATASYSTEM_PORT=18481` 固定访问 189 DataSystem，并继续省略 `DATASYSTEM_ETCD_ADDRESS` 避免 ServiceDiscovery。新增 `k8s/deployment-pairec-brpc-master.yaml`：PaiRec CPU 服务 `nodeName=master` 固定到 189，使用 `pairec-server:k8s-arm64-brpc-v1` 与 `PAIREC_TRACE_STDOUT=1`。新增 `scripts/k8s_cleanup_master_gpu_attempt.sh` 用于 scale 到 0、删除失败 inference Pod、移除 master 的 `pairec/gpu` label 并清理 master 上的 nvidia-device-plugin Pod；新增 `scripts/k8s_apply_189_pairec_188_inference_189_ds.sh` 一键 apply 反向拓扑并输出验证命令。验证：两个脚本 `bash -n` 通过，两个新增 YAML parse 均为 2 docs，`git diff --check` 通过。** |
| **6/26** | **master/189 GPU 分配继续诊断：`inference-brpc-trtllm` scale 到 0 后，master 的 `Allocated resources` 显示 `nvidia.com/gpu 0/0`，说明 GPU 没被业务 Pod 占用；但 `kube-system/nvidia-device-plugin-daemonset-644p2` 在 master 上变为 `CrashLoopBackOff`，worker1 上同 DaemonSet 正常 Running。此前 `OutOfnvidia.com/gpu` 不是业务 Pod 抢占导致，而是 master 的 NVIDIA device plugin 不健康。用户曾用 `kubectl -n pairec describe pod nvidia-device-plugin...` 查询，因 namespace 错误返回 NotFound；下一步应在 `kube-system` 查看该 Pod 的 current/previous logs 与 describe，定位 device plugin crash 原因。** |
| **6/26** | **跨节点 inference apply 后发现远端 live Deployment 残留开发二进制 overlay：Pod 事件显示 `MountVolume.SetUp failed for volume "brpc-dev-bin": hostPath type check failed: /home/zcx/pairec-brpc-dev/bin/brpc_inference_server is not a file`。本地 `k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml` 并无 `brpc-dev-bin`，说明这是早期为避免重传 150G 镜像而在远端 live Deployment 上留下的临时 hostPath 覆盖。跨节点稳定验证应移除该 `volumeMount`/`volume`，使用镜像内 `/opt/pairec-brpc/bin/brpc_inference_server`；也可临时在 master 上补文件，但不建议作为稳定方案。** |
| **6/26** | **master/189 GPU 接入完成：`nvidia-device-plugin-daemonset` 在 master 与 worker1 上均为 `1/1 Running`，`kubectl describe node master` 的 Capacity 与 Allocatable 均出现 `nvidia.com/gpu: 1`。此前的 master Calico、containerd `nvidia` runtime handler、device plugin 镜像问题均已越过。下一步可以正式 apply `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`，验证 `inference-brpc-trtllm` 调度到 master/189，并从日志确认 TensorRT-LLM 使用 GPU 且 DataSystem 连接固定到 `141.61.91.188:18481`。** |
| **6/26** | **新增 master/189 NVIDIA device plugin 镜像同步脚本：`scripts/sync_nvidia_device_plugin_image_to_master.sh`。脚本默认在 master 本机执行，自动读取 `kube-system/nvidia-device-plugin-daemonset` 当前镜像（如 `nvcr.io/nvidia/k8s-device-plugin:v0.19.2`），通过 ssh 在 worker1/188 用 `ctr -n k8s.io images export` 导出该镜像，复制并导入 master 本机 `k8s.io` containerd namespace，然后只重建调度到 master 的 nvidia-device-plugin Pod，并打印 device plugin Pod 与 master `nvidia.com/gpu` allocatable 验证结果。验证：`bash -n scripts/sync_nvidia_device_plugin_image_to_master.sh` 与 `git diff --check` 通过。** |
| **6/26** | **master/189 Calico 已恢复：`calico-kube-controllers`、master/worker1 上的 `calico-node` 均为 `1/1 Running`，说明 master 的 CNI 网络创建问题已越过。新的阻塞是 master 上 `nvidia-device-plugin` 拉取 `nvcr.io/nvidia/k8s-device-plugin:v0.19.2` 失败，报 `x509: certificate signed by unknown authority`，状态 `ImagePullBackOff`；worker1 上已有同镜像运行，因此下一步应从 worker1 的 containerd 导出该 nvidia device plugin 镜像并导入 master，再重建 master 上的 nvidia-device-plugin Pod。** |
| **6/26** | **新增 master/189 Calico 镜像同步脚本：`scripts/sync_calico_images_to_master.sh`。脚本默认在 master 本机执行，从当前 kube-system 的 `calico-node` DaemonSet 和 `calico-kube-controllers` Deployment 自动提取镜像列表，通过 ssh 在 worker1/188 用 `ctr -n k8s.io images export` 导出这些镜像，复制到 master 并导入本机 `k8s.io` containerd namespace，随后只重建调度到 master 的 Calico Pod。该脚本不直接修改 `/etc/cni/net.d`，优先解决 master 上 `calico-node Init:ImagePullBackOff`。验证：`bash -n scripts/sync_calico_images_to_master.sh` 与 `git diff --check` 通过。** |
| **6/26** | **master/189 Calico CNI 根因继续定位：`calico-kube-controllers` 被调度到 master 后同样因 Calico CNI `error getting ClusterInformation: connection is unauthorized` 无法创建 sandbox；`kubectl -n kube-system get pods` 显示 master 上 `calico-node-47v5z` 已 33 天处于 `Init:ImagePullBackOff`，worker1 上 calico-node 仍在运行。这说明 master 上新 Pod 失败不是 nvidia-device-plugin 特例，而是 master 节点 Calico CNI/节点网络组件长期未正常初始化。下一步应先让 master 的 calico-node 正常起来，优先同步 worker1 已有 Calico 镜像到 master containerd 并重建 master calico-node；若 calico-node Running 后仍 Unauthorized，再检查/同步 `/etc/cni/net.d` 下 Calico CNI kubeconfig。** |
| **6/26** | **master/189 NVIDIA runtime 配置后进入下一阻塞：`nvidia-device-plugin-daemonset` 已同时出现在 worker1 与 master，其中 master Pod 能被调度但卡 `ContainerCreating`；事件已从此前的 `no runtime for "nvidia" is configured` 变为 Calico CNI `plugin type="calico" failed (add): error getting ClusterInformation: connection is unauthorized: Unauthorized`。这说明 containerd `nvidia` runtime handler 已不再是当前报错点，新的阻塞是 master 节点 Calico CNI 本地 kubeconfig/token/RBAC 授权异常，导致新 Pod sandbox 网络创建失败。下一步优先重启/重建 master 上 calico-node 以刷新 `/etc/cni/net.d`，若仍失败再对比 worker1 与 master 的 Calico CNI kubeconfig 并谨慎同步。** |
| **6/26** | **新增 master/189 NVIDIA runtime 一键同步脚本：`scripts/sync_nvidia_runtime_to_master.sh`。脚本默认在 master/189 本机执行，从 worker1/188 通过 ssh 读取 `/etc/containerd/conf.d/99-nvidia.toml`，安装到本机同路径，备份 `/etc/containerd/config.toml` 后只补齐 `imports` 中的 `99-nvidia.toml`，不覆盖整份 containerd 配置；随后可自动重启 containerd/kubelet、重建 nvidia device plugin Pod、给 master 打 `pairec/gpu=true` 标签，并打印 `containerd config dump`、`crictl info` 与 `kubectl describe node master` 的验证结果。验证：`bash -n scripts/sync_nvidia_runtime_to_master.sh` 与 `git diff --check` 通过。** |
| **6/26** | **追溯 GPU 节点差异来源：项目历史在 6/9 只记录 `worker1 已注册 nvidia.com/gpu=1` 并固化 inference hostPath manifest，K8s README 也只把 `nvidia.com/gpu` 与 `runtimeClassName=nvidia` 写成部署前置条件；仓库内没有安装/同步 NVIDIA device plugin、containerd `99-nvidia.toml` 或 master GPU runtime 的脚本。结合现场 `containerd config dump` 对比，结论是当时只在 worker1/188 做过节点本地 NVIDIA runtime 配置，master/189 虽有 GPU 和 `/usr/bin/nvidia-container-runtime`，但没有导入 `/etc/containerd/conf.d/99-nvidia.toml` 到 containerd CRI runtime handler，因此 K8s 无法在 master 上使用 `RuntimeClass nvidia`。** |
| **6/26** | **master/189 containerd NVIDIA runtime 差异已定位：worker1/188 的 `containerd config dump` 显示 `imports = ["/etc/containerd/config.toml", "/etc/containerd/conf.d/99-nvidia.toml"]`，并在 CRI runtimes 中存在 `nvidia` handler，`BinaryName=/usr/bin/nvidia-container-runtime`；master/189 的 `containerd config dump` 与 `crictl info` 中均无 `nvidia`。两台机器都有 `/usr/bin/nvidia-container-runtime`，因此修复方向是把 worker1 的 `/etc/containerd/conf.d/99-nvidia.toml` 与必要的 imports 配置同步到 master，并重启 containerd/kubelet，再让 `nvidia-device-plugin-daemonset` 在 master 变为 Running。** |
| **6/26** | **master/189 GPU 接入继续推进：给 master 补齐调度条件后，`nvidia-device-plugin-daemonset` 已能被调度到 master，但 Pod 卡在 `ContainerCreating`，事件为 `failed to get sandbox runtime: no runtime for "nvidia" is configured`。用户检查到 worker1/188 与 master/189 都存在 `/usr/bin/nvidia-container-runtime`，但直接 grep `/etc/containerd/config.toml /etc/containerd/*.toml` 未发现 `nvidia` 配置；因此当前差异不在二进制是否存在，而在 containerd 实际加载的 CRI runtime handler 配置。下一步应对比两台机器的 `containerd config dump`、`crictl info` 与 `systemctl cat containerd`，找出 worker1 上 `nvidia` handler 的来源并同步到 master，重启 containerd/kubelet 后确认 master allocatable 出现 `nvidia.com/gpu`。** |
| **6/26** | **跨节点拓扑前置状态更新：用户确认 master/189 物理有 GPU，且 master/189 上的 `datasystem-pool-worker` 已从 Unknown 恢复为 `1/1 Running`，说明 pause/sandbox 侧问题暂时越过；但 `kubectl describe node master | grep nvidia.com/gpu` 仍无输出，而 worker1/188 有 `nvidia.com/gpu: 1`。`nvidia-device-plugin-daemonset` 当前只在 worker1 上有 Pod，master 上没有 device-plugin Pod，因此 189 仍不能承载请求 GPU 的 `inference-brpc-trtllm`。当前剩余关键动作是让 NVIDIA device plugin 在 master 上运行并使 node allocatable 出现 `nvidia.com/gpu`，随后再 apply 跨节点 inference。** |
| **6/26** | **远端跨节点拓扑前置检查发现尚未满足：`kubectl get nodes` 显示 master/189 与 worker1/188 均为 Ready，但 `kubectl describe node master` 的 Allocatable 未出现 `nvidia.com/gpu`，说明 189 当前还没有把 GPU 资源暴露给 K8s；同时 `datasystem-pool-worker` 在 master/189 上为 Unknown，事件显示 pause sandbox 镜像 `docker.io/library/pause-aarch64:3.8` 拉取超时以及 `cpuset.mems` 缺失导致 sandbox 反复重建。当前实际拓扑仍是 188 PaiRec -> 188 inference -> 188 DataSystem；要切到 188 PaiRec -> 189 inference -> 188 DataSystem，需先修 189 的 pause/sandbox 与 NVIDIA device plugin/GPU allocatable，再导入 `pairec-brpc-inference:k8s-arm64-trtllm-v1` 到 189 containerd，最后 apply `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`。** |
| **6/26** | **修正基线指标口径：`brpc@TCP*n` 不应直接使用 C++ `brpc-inference` 日志里的 `latency_ms`。该字段覆盖 brpc server 收到请求后执行 Recommend 的服务端总耗时，包含 TensorRT-LLM generate、semantic map、KVC offload/onboard 等后端处理；真正的 PaiRec -> inference brpc/TCP 通信/RPC耗时需要在 PaiRec Go brpc client 调用前后打点，或在 brpc 框架/client 侧采集 request send/response receive wall-clock。后续基线报告需拆成：E2E client、PaiRec Go 内部、brpc RPC wall-clock、inference server total、KVC Get/Set、other residual。** |
| **6/25** | **新增稳定跨节点固定 endpoint 验证分支：已切到 `brpc-cross-node-189-ds188`，新增 `k8s/deployment-inference-brpc-trtllm-cross-node-189-ds188.yaml` 与 `scripts/k8s_apply_inference_brpc_trtllm_cross_node_189_ds188.sh`。该 manifest 保持 `inference-brpc-trtllm` Deployment/Service 名称不变，PaiRec `brpc_endpoint` 不用改；通过 `nodeName: master` 将 inference 固定到 189，通过 `DATASYSTEM_HOST=141.61.91.188`、`DATASYSTEM_PORT=18481` 固定访问 188 DataSystem，并故意不设置 `DATASYSTEM_ETCD_ADDRESS`，避免 ServiceDiscovery 在 189 上优先选本地 worker。`docs/DATASYSTEM_WORKER_POOLING.md` 已补验证和回滚说明。** |
| **6/25** | **补 PaiRec Go stdout 观测增强：`services/recall/generative_recall.go` 新增 `PAIREC_TRACE_STDOUT=1` 门控的 `[PAIREC_TRACE]` 输出，默认关闭，不改变推荐链路行为；开启后在 cache hit、history error/empty、convert error、inference error 和成功返回时向 stdout 输出 `requestId/request_id/module=GenerativeRecall/from/protocol/cache_ms/history_ms/convert_ms/rpc_ms/brpc_ms/http_ms/items_ms/tr_*` 等字段，便于 `kubectl logs --tail=0 -f deploy/pairec` 稳定采集单请求内部阶段。`scripts/trace_single_brpc_datasystem_request.sh` 已同步展示 `from/protocol/rpc_ms/brpc_ms/inference_svc_ms` 字段。本机验证：`gofmt`、`bash -n scripts/trace_single_brpc_datasystem_request.sh`、`go test -mod=vendor ./services/...`、`go build -mod=vendor ./...`、`git diff --check` 均通过。下一步远端重建 PaiRec 小镜像并开启 `PAIREC_TRACE_STDOUT=1` 复跑单请求 trace。** |
| **6/25** | **手工交接状态更新：当前推荐链路是 `HTTP /api/recommend -> PaiRec Go -> Go baidu_std brpc/TCP client -> C++ brpc_inference_server -> TensorRT-LLM C++ -> KVC/DataSystem`。`brpc_recommend_client` 只是 C++ smoke/probe client，不是当前 PaiRec Go 改造口径。最近两次 `scripts/trace_single_brpc_datasystem_request.sh` 结果分别为：E2E `114.158ms/117.509ms`，brpc calls `1/1`，brpc server `107ms/107ms`，KVC `offload=2/onboard=0`，offload total 分别约 `14.479+8.010ms` 和 `16.816+7.427ms`。PaiRec Go 内部阶段日志仍未从 `kubectl logs` 或 Pod `/tmp` glog 文件中抓到；下一步应给 `services/recall/generative_recall.go` 增加可开关的 stdout trace，再重建 `pairec-server:k8s-arm64-brpc-v1` 复测。** |
| **6/25** | **补充单请求链路分解脚本：新增 `scripts/trace_single_brpc_datasystem_request.sh`，默认只发一次 `USER_ID=6312 SIZE=1` 的 PaiRec `/api/recommend` 请求，自动选择 port-forward 空闲端口，并以 `kubectl logs --tail=0 -f` 只采本次请求后的 PaiRec/brpc TRT-LLM 日志；请求完成后按 response `request_id` 额外在 PaiRec Pod `/tmp` glog 文件中检索 `GenerativeRecall/RecommendTrace` 行。输出包含 client E2E latency、response code/size、brpc `Recommend` 调用次数与 server `latency_ms`、PaiRec `cache/history/convert/http/items` 及 `tr_*` 阶段耗时（若日志可抓到）、KVC/DataSystem `offload/onboard` 次数，以及每次 `Create/D2H/Set`、`Get/H2D`、`total_ms` 明细；输出目录为 `/tmp/pairec_single_request_trace/<run_id>`。该脚本用于先解释一次请求内部发生了什么，再做并发基线。** |
| **6/25** | **固化 brpc + DataSystem pool 基线一键脚本：新增 `scripts/benchmark_brpc_datasystem_pool_baseline.sh`，默认检查 `pairec/inference-brpc-trtllm/datasystem-pool` Pod 与 Service，验证 PaiRec `brpc_endpoint` 配置，跑一次 PaiRec `/api/recommend` 功能 smoke，再用 `scripts/test_brpc_native_inference_smoke.sh` 做 C++ brpc/TCP 串行 smoke；随后启动本地 port-forward 与分轮日志采集，默认跑 `size=1, concurrency=10` 的系统 E2E 基线，并可选跑 `size=10, concurrency=10` 的召回质量基线。每轮会输出 benchmark 原始 stdout、JSON、PaiRec/brpc 日志，并自动统计 brpc Recommend 调用数、code 分布、item 分布、brpc server latency p50/p95/p99/p9999/max、DataSystem offload/onboard 事件数、offload/onboard per brpc，以及 `set_ms/get_ms/total_ms` p99/p9999。脚本已补充空闲本地端口自动选择：默认从 18080 开始，若被旧 port-forward 占用则自动换后续端口并同步更新 benchmark URL；日志采集改为 `kubectl logs --tail=0 -f`，避免历史 brpc/KV 日志混入本轮 summary；默认 `E2E_WARMUP=0`，让 summary 的 brpc/KV 调用数与 pressure/replay 请求数对齐，需预热时可显式设置。`docs/BRPC_DATASYSTEM_POOL_BASELINE_PLAN.md` 已改为优先使用该脚本。** |
| **6/25** | **DataSystem worker pool 任务暂存并转入基线摸测：远端 `datasystem-pool-etcd` 与两个 `datasystem-pool-worker` 已经在 worker1/master 上 Running，说明 hostNetwork、DaemonSet 和共享 ETCD 部署形态已通过 Pod 级验证；但 `debug_datasystem_pool_smoke.sh` 在 Python import 阶段确认当前 `docker.io/library/zcx-pairec-image:v1.1` 里的 DataSystem SDK 没有 `ServiceAffinityPolicy`，也没有 `yr.datasystem.service_discovery` 子模块，可用子模块仅包括 `cli/ds_client/ds_tensor_client/hetero_client/kv_client/object_client/stream_client/util`。结论：这不是 K8s 调度问题，而是运行镜像 SDK 版本不支持当前 ServiceDiscovery smoke/API；完整池化接入需后续基于 `/home/vivwimp/workspace/yuanrong-datasystem` 重建 DataSystem SDK/runtime，并重新构建 TRT-LLM/brpc runtime。当前决策：池化完整接入暂列阻塞，先在 worker1+master DataSystem pool 已启动的环境下摸 E2E/brpc/KV 基线，并在报告中标注 TRT-LLM C++ 可能仍走固定 endpoint 的限制。** |
| **6/25** | **补充并修复 DataSystem pool smoke 卡顿诊断脚本：远端 `datasystem-pool-etcd` 与两个 `datasystem-pool-worker` 已进入 Running，但 `scripts/test_datasystem_pool_smoke.sh` 执行无输出，后续诊断确认根因是 `kubectl exec ... python -` 缺少 `-i`，导致 heredoc 没有传入容器，Python 实际没有执行脚本内容。修复后进一步定位到当前 worker 镜像的 `yr.datasystem` 顶层没有导出 `ServiceAffinityPolicy`；脚本已改为优先从 `yr.datasystem` 导入，失败时 fallback 到 `yr.datasystem.service_discovery` 子模块，并在子模块也缺失时打印 `yr.datasystem.__file__`、可用 ServiceDiscovery 符号与子模块列表，便于判断是导出问题还是 SDK 版本过旧。验证：`bash -n scripts/test_datasystem_pool_smoke.sh scripts/debug_datasystem_pool_smoke.sh` 与 `git diff --check` 通过。** |
| **6/24** | **DataSystem worker 池化第一版本地落地：新增 `k8s/deployment-datasystem-pool-hostnetwork.yaml`，以 `datasystem-pool-etcd` 提供共享 ETCD `141.61.91.188:12379`，以 `datasystem-pool-worker` DaemonSet 在各节点启动 `<HOST_IP>:18481` worker，并通过 `HOST_IP` 写入 `host_id_env_name` 支持同节点亲和；新增 `scripts/k8s_apply_datasystem_pool.sh` 和 `scripts/test_datasystem_pool_smoke.sh`，用于部署 pool、验证 ServiceDiscovery 至少发现多个 worker 并做一次 KV `set/get`。新增 `trtllm-datasystem-service-discovery.patch` 和 `scripts/apply_trtllm_datasystem_service_discovery_patch.sh`，将 TensorRT-LLM `KvCacheManagerDataSystem/Tmp` 的 `KVClient` 初始化改为：存在 `DATASYSTEM_ETCD_ADDRESS` 时使用 `ConnectOptions.serviceDiscovery`，否则回退原 `DATASYSTEM_HOST/PORT`；同步给 `k8s/deployment-inference-brpc-trtllm.yaml` 增加 `DATASYSTEM_ETCD_ADDRESS`、`DATASYSTEM_CLUSTER_NAME`、`DATASYSTEM_HOST_ID_ENV_NAME`、`DATASYSTEM_AFFINITY_POLICY=PREFERRED_SAME_NODE`、`DATASYSTEM_ENABLE_CROSS_NODE_CONNECTION=true` 等运行时参数。新增 `docs/DATASYSTEM_WORKER_POOLING.md` 记录部署/重建/验证/回滚流程。验证：`bash -n` 三个新增脚本通过；两个 K8s YAML parse 输出 `yaml ok`；`git -C /home/vivwimp/TensorRT-LLM apply --check trtllm-datasystem-service-discovery.patch` 通过；`git diff --check` 通过。远程遗留：需要实际 apply pool、确认两个节点均有本地 DataSystem runtime 镜像，并重建包含该 TensorRT-LLM patch 的 runtime/base image，否则运行中 brpc TRT-LLM 仍不会真正使用 worker pool。** |
| **6/24** | **调研最新 `/home/vivwimp/workspace/yuanrong-datasystem` 及 RH2D：最新仓 `ConnectOptions` 已包含 `serviceDiscovery`、`enableRemoteH2D`、`fastTransportMemSize`、`deviceId`，C++ `ObjectClientImpl::Init()` 在传入 `serviceDiscovery` 时会通过 `ServiceDiscovery::SelectWorker()` 从 ETCD ready worker 中选点，并支持 `PREFERRED_SAME_NODE`/`REQUIRED_SAME_NODE`/`RANDOM` 亲和策略；Python `ServiceDiscovery`/`KVClient`/`DsClient` 也暴露相同能力。上游 Helm DaemonSet 已有 `host_id_env_name`、`etcd_address`、`cluster_name`、`enable_worker_worker_batch_get` 等参数，说明共享后端和 worker 池化应优先通过同一 ETCD + ServiceDiscovery 接入。RH2D 方面，最佳实践明确该路径是 Ascend NPU + CANN + RoCE 的 remote host-to-device 传输，worker 端需 `--remote_h2d_device_ids`，client 端需 `enableRemoteH2D=true`，API 侧以 `HeteroClient::MSetD2H/MGetH2D` 或 `KVClient::MGetH2D` 的设备指针/DeviceBlobList 为核心；当前 NVIDIA CUDA/TensorRT-LLM 部署和现有 patch 的 `Create/Set/Get` host buffer staging 不能直接启用 RH2D。建议阶段顺序：先把当前 TRT-LLM `KVClient` 初始化改造成可选 ServiceDiscovery 并验证跨 worker Set/Get，再在 Ascend/NPU 环境单独做 RH2D standalone benchmark，最后再改 KV block transfer 逻辑对接设备 blob，保留 host staging fallback。** |
| **6/24** | **从 `/home/vivwimp/TensorRT-LLM` 侧细化未使用 DataSystem worker 池化能力的根因：当前 patch 只在 `kvCacheManager.h` 引入 `datasystem/kv_client.h`，没有引入 `ServiceDiscovery`/`RouterClient`；`KvCacheManagerDataSystem` 与 `KvCacheManagerDataSystemTmp` 两个单例都只从 `DATASYSTEM_HOST`/`DATASYSTEM_PORT` 构造 `datasystem::KVClient`，没有读取 ETCD 地址、clusterName 或 affinity 参数，并显式设置 `enableCrossNodeConnection=false`、`enableRemoteH2D=false`；`kvCacheTransferManager.cpp` 的 offload/onboard 只通过这两个单例执行 `Create/Set/Get`，因此所有 KV block 都落到同一个固定 worker endpoint。结论：当前是为验证 C++ KV offload/onboard 做的最小 host/port 接入，不是 DataSystem 官方多 worker service discovery 接入。** |
| **6/24** | **调研 yuanrong-datasystem 共享后端能力：上游 README 明确 DataSystem 依赖 ETCD 做节点发现/健康检测/扩缩容，每节点部署 worker 并注册 ETCD，worker-worker 当前支持 TCP 传输；Helm chart 默认 `replicaNum=2`，并有 `enableWorkerWorkerBatchGet`、`enableRedirect`、`enableDataReplication` 等集群/远端取数相关配置。源码层 `ConnectOptions` 已包含 `serviceDiscovery`，`ObjectClientImpl::Init()` 可通过 `ServiceDiscovery::SelectWorker()` 从 ETCD 选择 worker，`KVClient` 内部复用 `ObjectClientImpl`；同时 worker 侧存在 `BatchGetObjectRemote` / `GetObjectRemote` 远端取数实现。当前项目差距是 TensorRT-LLM patch 只从 `DATASYSTEM_HOST/PORT` 构造 `KVClient`，且 K8s `inference-brpc-trtllm` 固定指向 worker1 `141.61.91.188:18481`，`deployment-datasystem-hostnetwork.yaml` 也是单 worker + isolated etcd，因此当前运行时尚未真正启用共享后端。下一步应先用两个 worker 注册同一 ETCD 做跨 worker Set/Get 验证，再决定是否改 TensorRT-LLM DataSystem 初始化参数** |
| **6/23** | **F14 PaiRec Go brpc client 远端 K8s 端到端验证通过：用户重新构建 `pairec-server:k8s-arm64-brpc-v1`、导入 worker1 并重启 PaiRec 后，`kubectl -n pairec exec deploy/pairec -- wget ... /api/recommend` 返回 `{"code":200,"msg":"success","size":10,...}`，10 个 item 均来自 `retrieve_id=generative_recall`；同时 `kubectl -n pairec logs deploy/inference-brpc-trtllm -c brpc-inference --since=2m | grep method=Recommend` 输出 `[brpc-inference] method=Recommend user=6312 code=200 items=10 latency_ms=228 backend=trtllm_cpp`。这证明最终链路已打通：PaiRec Go `GenerativeRecall` -> Go baidu_std brpc/TCP client -> `inference-brpc-trtllm:18100`/ClusterIP -> C++ `brpc_inference_server` -> TensorRT-LLM C++ backend。遗留：CoreDNS 当前 `0/1 ContainerCreating`，Pod 内 DNS 解析 `10.96.0.20` 超时，因此本次用 `10.96.15.101:18100` ClusterIP 绕过；后续需修 CoreDNS 并恢复 Service DNS 配置** |
| **6/23** | **修正 PaiRec Go baidu_std brpc `RpcMeta` 字段号：远端切到新镜像和 ClusterIP 后，`/tmp/recall_debug.log` 显示 brpc 请求已不再 DNS 超时，但返回 `service error: ;`，且 C++ brpc inference 未打印 `method=Recommend`，说明 Go client 能收到 brpc frame 但服务端没有正确识别请求业务 meta。对照 Apache brpc `baidu_rpc_meta.proto` 后确认 Go 手写 meta 字段号错误；已改为 `request=1,response=2,compress_type=3,correlation_id=4,attachment_size=5,content_type=10,checksum_type=11`，并补齐 `CONTENT_TYPE_PROTO` 与 `CHECKSUM_NONE`。验证：`go test -mod=vendor ./services/...`、`go build -mod=vendor ./...`、`git diff --check` 通过。下一步远端 pull 后重新构建 `pairec-server:k8s-arm64-brpc-v1` 并复测 `/api/recommend`** |
| **6/23** | **补充 PaiRec 小镜像完全绕过 Docker Hub builder 镜像的构建路径：用户远端构建继续失败于 `FROM golang:1.24-alpine`，错误为 BuildKit layer `failed size validation`，判断为 Docker registry mirror/cache 返回异常 layer，不是 Go 代码问题。新增 `docker/Dockerfile.pairec.binary` 与 `scripts/build_pairec_binary_image.sh`：先在宿主机执行 `GOPROXY=off GOSUMDB=off CGO_ENABLED=0 go build -mod=vendor` 生成静态 `/app/pairec-server`，再以本地已有 `docker.io/library/pairec-server:k8s-arm64-static` 为 runtime base 只覆盖二进制和配置，避免拉取 `golang:1.24-alpine` 和 `alpine:3.19`。验证：`bash -n scripts/build_pairec_binary_image.sh` 与 `git diff --check` 通过。远端下一步优先执行 `bash scripts/build_pairec_binary_image.sh docker.io/library/pairec-server:k8s-arm64-brpc-v1`** |
| **6/23** | **修复 PaiRec 小镜像构建依赖外网 Alpine apk 的问题：用户远端构建 `docker/Dockerfile.pairec` 时在 runtime stage `apk add --no-cache ca-certificates curl bash` 因 Alpine package index 拉取失败而中断。由于 `pairec-server` 是 CGO disabled 静态 Go 二进制，当前 K8s hostpath 部署也直接执行 `/app/pairec-server`，runtime 不需要联网安装 `bash/curl`；已删除 Dockerfile runtime stage 的 `apk add`，将健康检查从 `curl` 改为 Alpine busybox 自带 `wget`，并将 `docker/entrypoint-pairec.sh` 从 bash 改为 POSIX sh。验证：`sh -n docker/entrypoint-pairec.sh` 与 `git diff --check` 通过。下一步远端 `git pull` 后重新执行 `docker build -f docker/Dockerfile.pairec -t docker.io/library/pairec-server:k8s-arm64-brpc-v1 .`** |
| **6/23** | **F14 PaiRec Go brpc client 第一版落地：在 `services/recall` 新增手写 protobuf 消息和 baidu_std brpc/TCP frame client，不引入在线 Go SDK 依赖，覆盖 `RecommendService.Health/Recommend`；`TRTLLMClient` 新增 `protocol=brpc` 分支，brpc 成功时直接返回，失败时按 `brpc_fallback_to_http` 回退原 HTTP `/recommend`；`RecallAlgo` 支持 `protocol`、`brpc_endpoint`、`brpc_service_name`、`brpc_fallback_to_http`。新增 `configs/pairec_config.brpc.json`、`k8s/configmap-brpc.yaml` 和 `scripts/k8s_apply_pairec_brpc_hostpath.sh`，默认 HTTP 配置保持不变。验证：`go test -mod=vendor ./services/...` 通过，`go build -mod=vendor ./...` 通过，`bash -n scripts/k8s_apply_pairec_brpc_hostpath.sh`、`python -m json.tool configs/pairec_config.brpc.json`、`k8s/configmap-brpc.yaml` YAML parse 和 embedded RecallAlgo JSON parse 均通过。下一步需远端重建/替换 PaiRec server 镜像并跑 `/api/recommend` 端到端 brpc 验证** |
| **6/22** | **F14 真实 C++ TRT-LLM native brpc smoke 通过：用户对 `deployment/inference-brpc-trtllm` 执行 `scripts/test_brpc_native_inference_smoke.sh`，Health 返回 `health ok latency_ms=1 code=200 status=healthy`；Recommend 返回 `recommend ok index=1 latency_ms=224 code=200 user_id=brpc_smoke_1 items=5 inference_ms=223`，`summary ok=1 total=1 total_ms=224`。这条链路为 `brpc_recommend_client -> brpc_inference_server:18100 -> TensorRT-LLM C++ Executor -> semantic_id_map`，不经过 Flask HTTP，已证明 F14 真实 C++ brpc 推理服务可用。下一步进入 PaiRec Go 侧 brpc/TCP client 接入，并保留 HTTP fallback 方便灰度/回滚** |
| **6/22** | **F14 真实 C++ TRT-LLM brpc 后端远程启动成功：用户在 `inference-brpc-trtllm` Pod 日志中确认 `trtllm plugin preflight ok namespace=tensorrt_llm legacy_creators=36 v3_creators=3`，creator 列表包含 `GPTAttention/Gemm/GemmSwiglu` 等核心插件；TensorRT-LLM engine 反序列化成功，`Engine load time 313 ms`，容量参数生效为 `maxNumSequences=1/maxBatchSize=1/maxNumTokens=96`；KV cache 显示 `primaryBlocks=8 secondaryBlocks=28`，DataSystem C++ KV 初始化成功：`Create Datasystem class`、`Init KvCache Manager DataSystem success`；`brpc_inference_server` 完成初始化并监听 `0.0.0.0:18100`。这说明前序插件缺失、OOMKilled、tokenizer/config 路径问题均已越过；下一步是执行 brpc Health/Recommend smoke 验证真实推理响应** |
| **6/22** | **补强 TensorRT-LLM plugin registry 同类问题预检：在 `brpc_inference_server` 中新增 `--trt_plugin_preflight_only=1` 模式，只执行 `initTrtLlmPlugins()` 并列出 `libnvinfer_plugin_tensorrt_llm.so` 注册出的 plugin creators，不加载 engine、不启动 brpc server；正常 `trtllm_cpp` 启动路径也会先执行该预检并输出 `legacy_creators/v3_creators/creators` 日志。预检会 fail-fast 检查 Qwen engine 常见核心 creator：`Gemm`、`GPTAttention`、`GemmSwiglu`，避免只修 `Gemmtensorrt_llm` 后又等到 engine 反序列化阶段才发现另一个 creator 未注册。本机验证：brpc build 脚本 `bash -n`、`git diff --check`、TRT-LLM CMake 配置阶段均通过；真实 ARM 编译和 GPU Pod 预检仍需远程执行** |
| **6/22** | **F14 `backend=trtllm_cpp` 远程验证推进到 TensorRT-LLM plugin registry 阶段：worker1 K8s 日志显示 `maxNumSequences=1`、`maxBatchSize=1`、`maxNumTokens=96`，说明前序 Executor 容量参数已生效，OOMKilled 不再是当前主因；最新 CrashLoopBackOff 为 `Cannot find plugin: Gemmtensorrt_llm, version: 1, namespace:tensorrt_llm`，即 C++ brpc 进程在创建 TensorRT-LLM Executor 前没有显式注册 TensorRT-LLM 自定义插件。已修 `brpc_inference_server.cpp`：引入 `tensorrt_llm/plugins/api/tllmPlugin.h` 并在构造 Executor 前调用 `initTrtLlmPlugins()`；同步修 CMake/Docker/build 脚本，新增 `TRTLLM_PLUGIN_LIBRARY`，默认链接 `/TensorRT-LLM/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so` 并加入 install RPATH。本机验证：`bash -n` 三个 brpc build 脚本通过，`git diff --check` 通过，CMake TRT-LLM 配置阶段能识别新增 plugin lib 参数；远程下一步无需重新传 150G tar，优先用 dev binary overlay 覆盖 worker1 上 `/home/zcx/pairec-brpc-dev/bin/brpc_inference_server` 后 rollout restart** |
| **6/18** | **F14 `backend=trtllm_cpp` 第一版代码已落地，待远程 ARM/GPU runtime 验证：`cpp/brpc_gateway/brpc_inference_server.cpp` 新增 TensorRT-LLM C++ Executor 后端，启动时加载 `/app/trt_engines/qwen3_rec_v4`、`/app/exported/qwen3_rec/pairec_cpp_tokenizer.txt` 和 `semantic_id_map.json`；请求侧按当前 Python `TRTQwen3Backend` 的 prompt 结构构造 token ids，保留 `max_input_len=64`、`max_new_tokens=32`、`num_samples/top_k/temperature` 参数，生成后按 layer 收集 semantic special tokens、缺失层补 0、笛卡尔积组合并映射 item；trace 保留 `prompt_ms`、`runner_generate_ms`、`runner_calls`、`parse_combo_ms`、`map_item_ms`、`backend_total_ms`。新增 `PAIREC_ENABLE_TRTLLM_CPP` CMake 开关、TRT-LLM brpc 镜像构建脚本、`scripts/export_cpp_trt_tokenizer_config.py`、`k8s/deployment-inference-brpc-trtllm.yaml` 和 apply 脚本。本机验证：`python -m py_compile scripts/export_cpp_trt_tokenizer_config.py`、新增脚本 `bash -n`、`k8s/deployment-inference-brpc-trtllm.yaml` YAML parse、`protoc --cpp_out`、`git diff --check`、CMake TRT OFF 配置和 TRT ON 配置分支均通过；尚未在远程 ARM TensorRT-LLM runtime 内完成真实编译、镜像导入和 K8s smoke** |
| **6/18** | **F14 native C++ brpc inference K8s smoke 通过：`scripts/test_brpc_native_inference_smoke.sh` 输出 `health ok latency_ms=1 code=200 status=healthy`，`Recommend` 输出 `recommend ok index=1 latency_ms=1 code=200 user_id=brpc_smoke_1 items=5 inference_ms=0`。这条链路是 `brpc_recommend_client -> inference-brpc-native:18100 -> brpc_inference_server -> semantic_map backend`，不再经过 Flask HTTP 转发；当前验证的是协议、镜像、Service/Deployment 和 C++ brpc server 可用性，不代表真实模型推理时延。下一步进入 `backend=trtllm_cpp`，把 TensorRT-LLM C++ runner、tokenizer/prompt、semantic id 解析和 DataSystem/KV 运行时接入该 server** |
| **6/18** | **补强 brpc 镜像离线构建：master 本地已有 `zcx-pairec-image:v1.1` 与 `zcx-pairec-brpc-sdk:v1`，但使用 `docker.io/library/zcx-*` 作为 `BASE_IMAGE` 会触发 daocloud 元数据解析并因私有镜像不在白名单返回 403。已将 brpc build 脚本默认 base image 改成本地 tag；新增 `.dockerignore` 排除 `datasystem/uds` Unix socket，避免 legacy builder 打包 build context 时输出 `archive/tar: sockets not supported`；`docker/Dockerfile.brpc.gateway` 在 build/final stage 清空继承的 `LD_PRELOAD`，避免 DataSystem 预加载路径污染 brpc 镜像构建和 ldd 检查** |
| **6/18** | **统一镜像 tar 保存路径：后续 build/ship 脚本默认把导出的 `.tar` 或 `.tar.gz` 放到 `/home/zcx`，不再使用 `/tmp` 作为镜像中转目录；`ship_brpc_gateway_to_worker.sh` 与 `ship_brpc_inference_to_worker.sh` 会先创建本地 tar 目录，scp 到 worker 时也默认放到 `/home/zcx`，再由 worker 执行 `ctr -n k8s.io images import /home/zcx/*.tar`。已同步 `k8s/README.md` 示例和相关构建脚本输出** |
| **6/18** | **F14 调整方向：已用 `git revert` 生成提交撤销 PaiRec 侧 `brpc_http_proxy` sidecar 方案，因为该形态仍是 `HTTP -> brpc -> HTTP`，对时延优化意义不大；新增 `cpp/brpc_gateway/brpc_inference_server.cpp`，直接实现 `RecommendService`，不调用 Flask HTTP。当前 `backend=semantic_map` 会加载 `/app/data/tenrec/processed/semantic_id_map.json` 并返回确定性 mapped items，用于证明 native C++ brpc service、镜像、K8s 和客户端协议链路；`backend=trtllm_cpp` 当前 fail-fast，作为后续真实 TensorRT-LLM C++ runner 接入口。新增 `k8s/deployment-inference-brpc-native.yaml`、`scripts/build_brpc_inference_image.sh`、`scripts/ship_brpc_inference_to_worker.sh`、`scripts/k8s_apply_inference_brpc_native.sh`、`scripts/test_brpc_native_inference_smoke.sh`。下一步先远程构建/导入 `pairec-brpc-inference:k8s-arm64-v1` 并做 native smoke，再迁移 Python `TRTQwen3Backend` 的 prompt/tokenizer/runner.generate/parse/map trace 到 C++ backend** |
| **6/18** | **F14 brpc phase-1 K8s smoke 验证通过：基于 `docker.io/library/pairec-brpc-gateway:k8s-arm64-v1` sidecar，`scripts/test_brpc_gateway_smoke.sh` 输出 `health ok latency_ms=2 code=200 status=healthy`；`Recommend` 输出 `recommend ok index=1 latency_ms=267 code=200 user_id=brpc_smoke_1 items=5 inference_ms=264.52`，返回 5 个推荐 item，trace 显示 `backend=trt-qwen3`、`runner_calls=1`、`runner_generate_ms=212.35ms`、`total_ms=264.52ms`、`result_cache_source=disabled`。这证明 native brpc/`baidu_std` over TCP -> brpc-gateway sidecar -> localhost Flask `/recommend` -> TRT-LLM 的 phase-1 闭环已跑通；该结果暂不代表最终性能收益，因为 gateway 内部仍转发 HTTP，下一步需要同批 HTTP vs brpc 延迟和结果等价对比** |
| **6/16** | **补强 F14 brpc gateway 镜像构建防线：`scripts/build_brpc_gateway_image.sh` 在 docker build 后自动进入最终镜像，对 `/opt/pairec-brpc/bin/brpc_gateway` 与 `/opt/pairec-brpc/bin/brpc_recommend_client` 执行 `ldd`，若出现 `not found` 立即失败，避免再次出现编译成功但 K8s sidecar 运行时缺 `libbrpc.so`/Abseil/protobuf 等动态库导致 CrashLoopBackOff 的问题；`docs/F14_BRPC_GATEWAY_DESIGN.md` 已同步说明该校验。下一步重新构建 gateway 镜像时应先看到 ldd 全部解析成功，再执行 worker1 导入和 K8s smoke** |
| **6/16** | **F14 brpc 环境搭建流程已固化：新增 `scripts/build_brpc_sdk_image.sh`，基于 `zcx-pairec-image:v1.1` 构建 `docker.io/library/zcx-pairec-brpc-sdk:v1`，在专用 SDK base image 中安装编译依赖并构建/安装 Apache brpc headers/libs；新增 `scripts/ship_brpc_gateway_to_worker.sh`，固化 master 构建 gateway 镜像后的 `docker save`、`scp` 到 worker1、`ctr -n k8s.io images import` 或 `docker load` 导入流程；更新 `docs/F14_BRPC_GATEWAY_DESIGN.md` 为当前 master 构建、worker1 导入的实际 K8s 镜像流转方式。新增脚本本机 `bash -n` 通过，`feature_list.json` JSON parse 和 `git diff --check` 通过。当前远程 master 应使用本地 base tag：`BASE_IMAGE=zcx-pairec-image:v1.1 bash scripts/build_brpc_sdk_image.sh docker.io/library/zcx-pairec-brpc-sdk:v1`、`BASE_IMAGE=zcx-pairec-brpc-sdk:v1 bash scripts/build_brpc_gateway_image.sh`、`WORKER=root@141.61.91.188 bash scripts/ship_brpc_gateway_to_worker.sh`，再 apply/smoke** |
| **6/15** | **F14 brpc phase-1 PoC 设计与实现骨架完成：新增 `proto/recommend.proto` 定义 `RecommendService.Recommend/Health`，字段对齐当前 `/recommend` JSON 并保留 `raw_json` 便于对比；新增 `cpp/brpc_gateway/recommend_gateway.cpp`，作为 native brpc `baidu_std`/TCP server 监听 `18100`，内部通过 brpc HTTP channel 转发到 `127.0.0.1:18000/recommend`；新增 `cpp/brpc_gateway/recommend_client.cpp` 作为 C++ smoke client；新增 `docker/Dockerfile.brpc.gateway`、`k8s/deployment-inference-brpc-image.yaml` 和构建/部署/验证脚本。当前本机验证：`protoc --proto_path=proto --cpp_out=/tmp/pairec-brpc-proto-check proto/recommend.proto` 通过；`bash -n scripts/build_brpc_gateway_image.sh scripts/k8s_apply_inference_brpc_gateway.sh scripts/test_brpc_gateway_smoke.sh` 通过；`python -c "import yaml; list(yaml.safe_load_all(open('k8s/deployment-inference-brpc-image.yaml'))); print('yaml ok')"` 输出 `yaml ok`；`cmake -S cpp/brpc_gateway -B /tmp/pairec-brpc-cmake-check` 找到 Protobuf 3.12.4 后按预期失败于 `brpc SDK was not found`，说明本机缺 brpc headers/libs，需远程或专用 builder 镜像完成编译。已补 `scripts/build_brpc_sdk_image.sh` 固化 brpc SDK base image 构建，并补 `scripts/ship_brpc_gateway_to_worker.sh` 固化 master 构建、tar 导出、scp 到 worker1、containerd/docker 导入流程。F14 仍为 pending，下一步是远程构建 gateway 镜像并在 inference Pod 内做 `Health`/`Recommend` smoke，再输出 HTTP vs brpc 延迟对比** |
| **6/12** | **补充 E2E DataSystem 时延报告指标口径说明：在 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md` 新增“指标口径详细说明”，逐项解释 `client_e2e_ms`、PaiRec `total/user_feature/recall/filter/rank/merge/sort`、GenerativeRecall `cache/history/convert/http/items/http_overhead`、inference `tr_*`/TRT service stages，以及 C++ DataSystem per KV block `offload.create/d2h/set/total`、`onboard.get/h2d/total`。报告明确区分 request 级、服务内阶段级和 KV block 级指标，并标注 `tr_*` 与 inference trace 字段的别名关系，避免将 Python 结果缓存/DataSystem 指标与 TensorRT-LLM C++ KV block Set/Get 混淆** |
| **6/12** | **K8s C++ KV DataSystem offload/onboard 功能闭环验证通过：基于正式 `pairec-inference:k8s-arm64-ds-runtime-v1` 镜像运行 pressure/replay 压测，日志判定显示 KV pool `primaryBlocks=32 secondaryBlocks=28`、scheduler `MAX_UTILIZATION`、`reuse disabled warning seen=False`，新增结构化 DataSystem trace `offload/onboard=1313/3`，`Get Key=3`，说明 C++ 层已真实触发 GPU KV block offload 到 DataSystem 并从 DataSystem onboard 回 HBM。当前样本量仍不足以给 p9999 结论：offload count=1313、onboard count=3，低于脚本建议 minimum=10000/recommended>=100000；下一步只需扩大 replay/onboard 压测样本用于分位数报告，不再需要改 runtime 镜像或 engine** |
| **6/12** | **正式 K8s inference DS runtime 镜像启动验证通过：基于 `scripts/build_datasystem_runtime_base_image.sh` 固化的 patched TensorRT-LLM runtime 重建 `docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1` 后，inference Pod 启动日志显示 `[TRTQwen3Backend] Scheduler policy: max_utilization, max_kv_tokens=1024, host_cache_size=104857600`、KV pool `primaryBlocks=32 secondaryBlocks=28`、`[TensorRT-LLM][Datasystem] Create Datasystem class`、`Init KvCache Manager DataSystem. host = 141.61.91.188, ip = 18481` 与 `Init KvCache Manager DataSystem success`。Python DataSystem 按预期保持 `PYTHON_DATASYSTEM_ENABLED=0`，当前只验证 C++ KV path。下一步运行 `scripts/test_trt_cpp_kv_offload.py` 的 pressure/replay 压测，确认结构化 `[Datasystem][TRACE] op=offload/onboard` 出现并汇总 C++ Set/Get latency** |
| **6/12** | **确认 K8s C++ DataSystem 差异来自 runtime 镜像而非 engine：用户在历史跑通容器 `3d25ebe028d6` 内用同一 `qwen3_rec_v4` 启动服务，日志出现 `[TensorRT-LLM][Datasystem] Create Datasystem class`、`Init KvCache Manager DataSystem success`，且 `Blocks per window size` 显示 `primaryBlocks=32 secondaryBlocks=28`，说明该容器内 TensorRT-LLM C++ patch 与 `block_ds_consumer.so/stub_gpu.so/libabseil` 预加载链路有效。进一步确认 `tensorrt-llm` 是 editable project，实际位置为 `/TensorRT-LLM`，而该路径是宿主机 `/home/zcx/TensorRT-LLM` 的 bind mount，因此不能直接 `docker commit` 旧容器。已新增 `scripts/build_datasystem_runtime_base_image.sh`，从旧容器复制 `/TensorRT-LLM` 到新的 `docker.io/library/zcx-pairec-ds-runtime:v1` base image；正式 runtime Dockerfile 继续基于该 base 构建，并在镜像内编译 `/opt/pairec/lib/block_ds_consumer.so` 与 `/opt/pairec/lib/stub_gpu.so`；entrypoint 默认 `PYTHONPATH` 同时包含 `/home/TensorRT-LLM`、`/TensorRT-LLM` 与 cutlass python 路径，启动时清理旧 `LD_PRELOAD`，按顺序加载 block/stub/NVML/abseil；K8s manifest 默认 tag 为 `docker.io/library/pairec-inference:k8s-arm64-ds-runtime-v1`。下一步重建/导入新镜像后复验 C++ offload/onboard TRACE** |
| **6/12** | **修正 K8s C++ DataSystem KV 失败判断：用户在历史跑通容器中检查 `trt_engines/qwen3_rec_v4/config.json`，同样显示 `context_fmha=False`、`use_paged_context_fmha=False`、`kv_cache_type=PAGED`，说明不能把当前失败简单归因于 `qwen3_rec_v4` engine 配置。已将 `k8s/deployment-inference-image.yaml` 恢复为默认 `qwen3_rec_v4`，新增 `scripts/k8s_check_trtllm_datasystem_runtime.sh`，用于在当前 inference Pod 内同时打印 engine config、`ModelRunnerCpp.from_dir` 签名，并扫描 TensorRT-LLM `.so` 是否包含 `Create Datasystem class`、`Init KvCache Manager DataSystem`、`op=offload/onboard` 等 C++ DataSystem patch 字符串。新的判断顺序：先确认 runtime 镜像内 C++ DataSystem patch 是否存在并被加载；若不存在，优先重建/替换 TensorRT-LLM runtime；若存在但 upstream block reuse warning 仍阻断路径，再使用 `qwen3_rec_v4_paged_fmha` engine build job 作为后备方案** |
| **6/12** | **远程 verifier 显示 `reuse disabled warning seen=True`、offload/onboard/TRACE 均为 0；进一步检查当前 K8s 挂载的 `/app/trt_engines/qwen3_rec_v4/config.json`，确认 `build_config/plugin_config/context_fmha=False`、`use_paged_context_fmha=False`。该配置会触发 upstream TensorRT-LLM 关闭 `kv_cache_enable_block_reuse` 的 warning，但由于历史容器同样配置曾跑通 C++ DataSystem offload/onboard，当前不能仅凭 engine 配置下结论。为后备验证新增 `k8s/job-build-trt-engine-paged-fmha.yaml` 与 `scripts/k8s_build_trt_engine_paged_fmha.sh`，可在 worker1/GPU 上构建独立 `trt_engines/qwen3_rec_v4_paged_fmha`，构建参数包含 `--kv_cache_type paged --remove_input_padding enable --context_fmha enable --use_paged_context_fmha enable`；旧 `qwen3_rec_v4` 保留为默认 HTTP/TRT 基线 engine。当前本地 `bash -n` 与 YAML parse 通过；下一步优先诊断 inference 镜像内 TRT-LLM C++ DataSystem patch/动态库是否存在** |
| **6/11** | **为 C++ DataSystem KV 验证版 inference 镜像切换独立 tag：默认镜像从 `docker.io/library/pairec-inference:k8s-arm64-runtime` 改为 `docker.io/library/pairec-inference:k8s-arm64-ds-kv-v1`，当前导出 tar 统一保存在 `/home/zcx` 下；`k8s/deployment-inference-image.yaml`、`scripts/build_inference_runtime_image.sh` 与 `k8s/README.md` 已同步，避免覆盖已验证 HTTP/TRT 基线镜像，也避免 kubelet/containerd 因 `IfNotPresent` 使用旧 tag 缓存** |
| **6/11** | **修复 DataSystem K8s Pod 固定 worker 二进制路径错误：远程首次启动 `datasystem` Pod 报 `datasystem_worker not found or not executable: /usr/local/lib64/python3.11/site-packages/yr/datasystem/datasystem_worker`；已将 manifest 改为启动时自动探测 `datasystem_worker`（先 `command -v`，再搜索 `/usr/local`、`/usr`、`/opt`、`/root` 下 DataSystem 路径），找不到时打印 `dscli` 与候选 DataSystem 文件，避免继续猜固定路径。下一步远程重新 apply，若仍找不到 worker 二进制，则根据候选日志决定改用镜像内实际路径或改为 `dscli start` 包装方式** |
| **6/11** | **新增 K8s DataSystem hostNetwork 部署清单：`k8s/deployment-datasystem-hostnetwork.yaml` 固定调度到 worker1，使用 `docker.io/library/zcx-pairec-image:v1.1` 作为 runtime，挂载 worker1 已解压的 `/home/zcx/workspace/pairec4tigerllm/etcd-v3.5.10-linux-arm64`，在容器内以前台进程启动独立 etcd `141.61.91.188:12379/12380` 与 `datasystem_worker --worker_address=141.61.91.188:18481 --shared_memory_size_mb=5120`；为 DataSystem worker 挂载 6Gi memory `emptyDir` 到 `/dev/shm`，避免 K8s 默认 64MiB shm 导致 worker 启动失败；新增 `scripts/k8s_apply_datasystem_hostnetwork.sh`，本地 YAML parse 与 bash 语法检查通过。当前 18481/12379/12380 在 master/worker1 均未占用，下一步远程 apply DataSystem，再重建/apply inference 镜像验证 C++ offload/onboard TRACE** |
| **6/11** | **C++ DataSystem KV trace 验证链路已在 K8s inference runtime 配置侧补齐，待远程重建镜像复验：新增 `TRT_KV_CACHE_HOST_CACHE_SIZE` 传参，构建期 patch `ModelRunnerCpp.from_dir()` 可把 `host_cache_size/onboard_blocks` 转发进 `KvCacheConfig`，用于将 `secondaryBlocks=0` 改为可 offload 的二级池；新增 `PYTHON_DATASYSTEM_ENABLED=0`，允许容器继续向 TensorRT-LLM C++ 暴露非空 `DATASYSTEM_HOST/PORT`，但不启用 Python `yr.datasystem`；`k8s/deployment-inference-image.yaml` 已临时关闭 Python 结果缓存并按当前 `dscli start -w --worker_address ${HOST_IP}:18481` 设置 `DATASYSTEM_HOST=141.61.91.188`、`DATASYSTEM_PORT=18481`、`TRT_KV_CACHE_HOST_CACHE_SIZE=104857600`，下一步远程目标是日志出现 `secondaryBlocks>0`、`[TensorRT-LLM][Datasystem] Init KvCache Manager DataSystem` 和 `[Datasystem][TRACE] op=offload/onboard`** |
| **6/11** | **正式 inference runtime 镜像远程复验通过：重建并导入 `docker.io/library/pairec-inference:k8s-arm64-runtime` 后，`k8s/deployment-inference-image.yaml` 成功启动 TRT 服务；`GET /health` 返回 `status=healthy backend=trt-qwen3 trt_num_samples=1 datasystem=disabled`；直连 `POST /recommend` 返回 `code=200` 和 5 个 recommendations，trace 显示 `runner_calls=1`、`runner_generate_ms≈187ms`、`total_ms≈232ms`；随后 PaiRec 端到端 `GET /ping -> success`，`POST /api/recommend` uid=6312,size=10 返回 `code=200` 和 10 个 `generative_recall` item。正式 inference 镜像已替代启动时 patch/hostPath 代码依赖，当前仅模型/数据仍为 hostPath** |
| **6/11** | **正式 inference runtime 镜像首次远程验证发现镜像缺 `training.rqvae`：`server.py` 导入 `training.decoder.model` 时触发 `training/__init__.py` 导入 `RQVAE`，旧 Dockerfile 只复制 `training/decoder` 导致 `ModuleNotFoundError: No module named 'training.rqvae'`；已修 `docker/Dockerfile.inference.runtime` 为复制完整 `training/` 包，待远程重新构建镜像并复验** |
| **6/10** | **正式 inference runtime 镜像方案已新增：`docker/Dockerfile.inference.runtime` 基于已验证 `zcx-pairec-image:v1.1` runtime，复制 `/app` 业务代码并在构建期 patch TensorRT-LLM `ModelRunnerCpp.from_dir()` 支持 `scheduler_config`；新增 `docker/entrypoint-inference-runtime.sh` 统一设置 gcc-toolset-14 `LD_LIBRARY_PATH`、有效 NVML 和 DataSystem abseil `LD_PRELOAD`；新增 `k8s/deployment-inference-image.yaml` 与 `scripts/build_inference_runtime_image.sh`、`scripts/k8s_apply_inference_image.sh`。本机静态校验 `py_compile`、`bash -n`、YAML parse 通过；待远程 ARM 构建、导入 worker1 并复验 HTTP/TRT 基线** |
| **6/10** | **F13 K8s PaiRec 端到端闭环验证通过：基于 `go build -mod=vendor` 生成 arm64 静态 PaiRec 二进制并打包为 `docker.io/library/pairec-server:k8s-arm64-static`；修复 Alpine 动态链接导致的 `exec /app/pairec-server: no such file or directory`；修复 PaiRec `../data/...` 相对路径，通过 hostPath 将 worker1 `/home/zcx/workspace/pairec4tigerllm/data` 挂到 `/data`，`workingDir=/app`；首次 fallback JSON 触发 1Gi OOM 后将 PaiRec memory limit 提到 8Gi；远程验证 `GET /ping -> success`，`POST /api/recommend` uid=6312,size=10 返回 `code=200` 和 10 个 `generative_recall` item；新增 `k8s/deployment-pairec-hostpath.yaml` 与 `scripts/k8s_apply_pairec_hostpath.sh`** |
| **6/9** | **F13 K8s inference 单服务闭环跑通并固化 hostPath manifest：worker1 已注册 `nvidia.com/gpu=1`；`zcx-pairec-image:v1.1` 作为 TRT/DataSystem runtime，hostPath 挂 `/home/zcx/workspace/pairec4tigerllm`；修正 `LD_LIBRARY_PATH` 使用 gcc-toolset-14、`LD_PRELOAD` 使用有效 NVML + DataSystem abseil，并在启动时临时 patch `ModelRunnerCpp.from_dir()` 转发 `scheduler_config`；远程验证 `/health` 返回 `status=healthy backend=trt-qwen3 trt_num_samples=1`，`/recommend` 返回 `code=200` 和 5 个 item；新增 `k8s/deployment-inference-hostpath.yaml` 与 `scripts/k8s_apply_inference_hostpath.sh`。遗留：Python DataSystem 为 disabled，C++ `CacheTransceiver` disabled，PaiRec Go 镜像/Deployment 待接入** |
| **6/5** | **F13 K8s 最小闭环部署配置已成型：修正 inference `:18000`、PaiRec `:18080`、PaiRec `/ping` 探针、`http://inference:18000` 服务发现、TRT engine/Qwen3/checkpoint/DataSystem/LD_PRELOAD 环境变量、GPU 单副本 `Recreate` 策略；新增 `k8s/README.md` 执行步骤；PaiRec Dockerfile 改为 `-mod=vendor` 离线构建并本机验证 `go build -mod=vendor` 通过** |
| **6/5** | **远端 DataSystem Get 到本地测试完成并补充到 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md`：DS host=`141.61.91.188:18581`，client p50=1050.3ms、p99=1689.2ms；C++ onboard=1346，Get p50=315.572ms、p99=317.038ms；H2D p50=0.405ms，确认当前路径是远端 DS Get 到本机 host 后再本机 H2D，remote H2D 暂未测试** |
| **6/4** | **`docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md` 补充原始 benchmark stdout 附录，保留 pressure/replay、请求级阶段和 C++ DataSystem block 指标的原始输出，便于复核摘要表** |
| **6/4** | **最终 E2E DataSystem Set/Get 压测完成并生成 `docs/E2E_DATASYSTEM_FINAL_REPORT_2026-06-04.md`：12000 请求成功 11853，client p50=92.8ms、p99=103.2ms；C++ offload=29330，Set p50=0.746ms、p99=1.015ms、p9999=1.689ms；C++ onboard=13421，Get p50=0.623ms、p99=0.818ms、p9999=1.094ms** |
| **6/4** | **新增 `TRT_RESULT_CACHE_ENABLED=0`，可关闭 Python TRT 推荐结果缓存；`scripts/benchmark_e2e_latency.py` 新增 `--uid-file`、pressure/replay 两段流量和按阶段 DataSystem C++ 指标，支持在端到端报告里同时观察 request stages 与 C++ `offload.set_ms` / `onboard.get_ms`** |
| **6/4** | **输出 `docs/E2E_LATENCY_REPORT_2026-06-04.md`，汇总 TRT_NUM_SAMPLES=1 的 PaiRec E2E、缓存分组、当前 E2E 中 DS trace 缺失原因与下一轮充分测 DS Set/Get 的要求** |
| **6/4** | **`scripts/benchmark_e2e_latency.py` 合并 DataSystem C++ KV block 统计：同一份 E2E 报告同时输出请求级阶段耗时和 `offload.set_ms` / `onboard.get_ms` 等 per-block 指标** |
| **6/4** | **TRT_NUM_SAMPLES=1 PaiRec E2E 完成: 60 请求全成功，client p50=3.5ms、p95=90.5ms、p99=92.8ms；报告混入结果缓存命中，冷 miss 对应 p95/p99 尾部约 90ms** |
| **6/4** | **远程 runner 轮数 A/B 完成: 1/2/4/8 轮均 `full_topk=60/60`；HTTP p50 分别为 91.8/180.1/350.2/701.7ms；确认 runner 耗时近似线性缩放** |
| **6/4** | **修复 `scripts/benchmark_trt_runner_samples.py` 使用 `--stop-command 'pkill -f ...'` 时会匹配自身 `--server-cmd` 并被 `Terminated` 的问题；文档改为脚本外手动清理旧服务** |
| **6/3** | **新增 `scripts/benchmark_trt_runner_samples.py`，自动核验 `/health` 的 `trt_num_samples`、运行直连压测、解析 `runner_calls` 并输出 JSON/CSV 对比** |
| **6/2** | **TRT runner 优化准备: 定位每个 miss 请求串行执行 8 次 `runner.generate()`；增加 `TRT_NUM_SAMPLES` 参数和单轮 trace，支持 1/2/4/8 轮质量-时延 A/B** |
| **6/2** | **DataSystem onboard 专项扩样准备: 报告明确区分 host DataSystem API、host 计时同步 D2H/H2D 和总 wall-clock；新增 `--min-onboard-samples` 门槛** |
| **6/2** | **远程 replay 修正验证通过: offload 8237 次，onboard 177 次；Get p50=0.725ms、p99=1.709ms，Get+H2D p50=1.258ms、p99=2.291ms** |
| **6/2** | **远程应用 C++ trace patch 并完成首轮 DataSystem 实测: offload 4181 次，Set p99=1.093ms；onboard/Get 已观测到 1 次，需继续增加读样本** |
| **6/2** | **F12 C++ DataSystem 时延观测补齐: 新增 Create/D2H/Set、Get/H2D 结构化 trace patch；C++ 压测和 PaiRec E2E 汇总新增 p99/p9999/max** |
| **6/1** | **端到端阶段性收口: `dev` 固化为可回退基线；后续从专用分支开展 TRT-LLM C++ DataSystem 与原生 pinned DRAM 的端到端 A/B** |
| **6/1** | **开始 F12 推荐系统时延分析: 补齐 PaiRec 入口、GenerativeRecall、Python TRT 服务和 TRT runner 分阶段 trace；新增端到端压测汇总脚本** |
| **6/1** | **F08 PaiRec 对接完成: 远程复验 Kafka 实时特征用户 `130`、`2184` 均单次触发 TRT 推理并返回完整映射结果** |
| **6/1** | **修复 PaiRec 自定义 recall 启动 panic: 外部注册同步写入配置签名，框架二次加载时正确跳过内置工厂** |
| **6/1** | **修复远程 DNS 不可用: 将完整 Go `vendor/` 纳入仓库，PaiRec 启动固定使用 `-mod=vendor` 离线依赖** |
| **6/1** | **修复 PaiRec 远程启动: 将 `go.sum` 纳入版本控制；启动脚本正确导出 `CONFIG_PATH` 并移除无效 `--port` 参数** |
| **6/1** | **推荐链路对接 — PaiRec 端到端返回成功；修复 Go 客户端 500ms 超时导致的重复推理** |
| **5/30** | C++ KV cache offload/onboard 闭环验证通过; 任务切换到推荐链路工程化 |
| **5/29** | C++ offload/onboard 链路突破 (FMHA crash根因/C++源码绕过/重编.so) |
| **5/28** | KV Cache + offload/onboard 闭环 (TRT/PyTorch双路径/三层缓存全通) |
| **5/27** | 推理打通 (FMHA bf16 kernel SM89非法内存访问根因) |
| **5/23** | ARM 4090D推理部署 + TRT-LLM引擎构建 + 多轮采样去重 |
| **5/22** | 训练完成 epoch 20 (loss 2.49) |
| **5/21** | DDP三卡训练启动 (epoch 11) |

## 当前状态

- **训练**: ✅ epoch 20, loss 2.49
- **TRT-LLM 引擎**: ✅ bfloat16, 1.46 GB, 限制: max_input_len=64, max_new_tokens=32, max_seq_len=96
- **C++ KV offload/onboard**: ✅ 闭环验证通过
- **推理服务**: ✅ /recommend 可用
- **PaiRec 对接**: ✅ Kafka 实时特征、生成式召回、TRT 推理和 item 映射链路已打通
- **时延分析**: ✅ 第一版端到端 trace 和冷请求分解已验证；✅ C++ DataSystem Set/Get 阶段性统计已完成；✅ 关闭结果缓存后的 E2E pressure/replay 最终报告已生成；✅ 远端 DS Get 到本地中等样本已完成；🔄 remote H2D 与 pinned DRAM A/B 待补充
- **K8s 部署**: ✅ ARM worker1 HTTP/TRT 基线闭环已通过：正式 inference runtime 镜像 `/health` + `/recommend` 和 PaiRec `/ping` + `/api/recommend` 均已验证；🔄 仍需将模型/数据 hostPath 替换为 PVC，并单独修 DataSystem Python/C++ KV 路径
- **brpc 改造**: 🔄 F14 已回退 PaiRec proxy 方案，转向 C++ brpc inference service；`semantic_map` native smoke 已在 K8s 通过；`trtllm_cpp` 后端第一版已实现并通过本机静态/CMake 配置验证，待远程 ARM TRT-LLM runtime 构建、部署和真实 brpc smoke
- **vLLM-Ascend NPU 替换**: 🔄 F15 模型导出、910B Chip1/NUMA6 容器、原生 brpc、合法商品约束、并发2/4 smoke 和 PaiRec Docker 灰度 E2E 已验证；当前端到端稳态约 `307-370ms`
- **vLLM-Ascend DataSystem**: ⏳ 尚未接入；当前 `datasystem_kvc=disabled`、prefix cache关闭，默认128-token block下当前最多64-token输入预计不会产生Set/Get；下一步先完成SDK与connector连接并以真实指标确认调用数

## 6/1 探索：推理命中率优化 (5个bug修复)

### 问题链

PaiRec 调用推理服务 → 大部分用户返回 `code:299 "items size not enough"`。

根因是 **推理服务返回的推荐 item 太少**（1个或0个），远低于 PaiRec 期望的 size（10）。

### 修复清单

| # | 提交 | 问题 | 修复 |
|---|------|------|------|
| 1 | `f7beeb4` | 20条历史→117 tokens 超引擎 max_input_len=64 | prompt 自动截断，保留最近9条 |
| 2 | `fb1cb98` | `<s0_X><pad><s1_Y>` 因中间有非语义token被丢弃 | 先过滤有效token再匹配，跳过非语义token |
| 3 | `326e1bb` | 模型输出层序乱 (s2,s1,s0,s3) 连续递增模式找不到 | 改为按层收集+笛卡尔积组合，不要求顺序 |
| 4 | `5533a16` | 20 tokens输出太短+8轮各自为战 | 8轮token合并到一个池子统一组合 |
| 5 | `1544a3f` | max_new_tokens>32 C++层卡死不报错 | 硬限制≤32 (引擎构建时预留) |
| 6 | `b44dbc6` | layer 3永远为0 (1条历史的用户) | 缺失层用{0}填充，由semantic_id_map验证 |

### 引擎硬限制 (重要)

```
max_seq_len = 96
  ├─ max_input_len = 64   (超过报 RuntimeError)
  └─ max_new_tokens = 32  (超过 C++ 层卡死不报错!)
```

必须同时遵守两个限制。

### 核心机制

**组合模式**: 8轮采样 × 32 token = 256 token 池 → 按层收集所有有效语义token → 笛卡尔积组合 → semantic_id_map 验证

**layer 填充**: 当某层缺失时用 {0} 补位 → 组合出候选 → map 验证真假

## 6/1 验证：PaiRec 端到端功能打通

### 验证证据

| 用户 | history_len | PaiRec 响应 |
|------|-------------|-------------|
| `303` | 20 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `1201` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |
| `130` | 1 | `code=200`, `size=5`, 5 个 `generative_recall` item |

### 新发现：Go 客户端超时导致重复推理

单次 TRT miss 约 `667-812ms`，但 Go 客户端默认超时仅 `500ms` 且最多尝试 3 次。同一个 PaiRec 请求会并发触发多次 GPU 推理，再由后续重试命中 HBM 结果缓存。

已在本地修复：`RecallAlgo` 增加 `timeout_ms=3000`、`max_retries=1`，并同步修改默认配置。待远程部署后确认单次 PaiRec 请求只触发一次 `/recommend`。

### 远程启动修复

远程 `go run` 曾因仓库未跟踪 `go.sum` 报依赖校验缺失。已将 `go.sum` 纳入版本控制，并修复 `scripts/start_pairec.sh`：导出 `CONFIG_PATH` 供 `main.go` 首次加载，移除 PaiRec 未定义的 `--port` 参数。

远程环境随后因 DNS 解析失败无法访问 `mirrors.aliyun.com`。已将完整 `vendor/` 纳入版本控制，启动脚本固定使用 `go run -mod=vendor`，不再依赖在线下载。

首次 vendor 提交仍遗漏 128 个文件：根因是 `.gitignore` 中通用 `lib/` 规则误伤 vendor 内 ClickHouse、Apache Thrift 和 PostgreSQL 驱动目录。已增加 `!vendor/**` 例外并补齐文件。验证方式：从 Git 暂存区导出临时副本，在 `GOPROXY=off`、空 `GOMODCACHE` 和空 `GOCACHE` 下执行 `go test -mod=vendor ./services/...`，全部通过且模块缓存文件数为 0。

### 自定义 recall 注册修复

PaiRec 启动时会再次执行 `recall.Load()`。此前 `main.go` 手工注册 `GenerativeRecall` 时只写入实例，没有写入框架的配置签名；二次加载时框架尝试用内置工厂重建自定义类型并 panic：`recall empty, name:generative_recall`。

已在 vendored recall 包增加 `RegisterRecallWithConfig()`，同步写入实例和配置签名，`main.go` 改用该入口。启动级验证已越过注册阶段并输出 `server start`。

## 6/1 F12：推荐系统端到端时延分解

已按当前 Qwen3 TRT 链路补齐 trace：

- PaiRec 入口内部：`user_feature_ms`、`recall_ms`、`filter_ms`、`general_rank_ms`、`feature_ms`、`rank_ms`、`pipeline_wait_ms`、`merge_ms`、`sort_ms`
- GenerativeRecall：`history_ms`、`convert_ms`、`http_ms`、`items_ms`、`http_overhead_ms`
- Python TRT 服务：`prepare_input_ms`、`kv_lookup_ms`、结果缓存查询和异步写入提交耗时
- TRT 后端：`prompt_ms`、8 轮累计 `runner_generate_ms`、`parse_combo_ms`、`output_pad_ms`
- 新增 `scripts/benchmark_e2e_latency.py`：从 PaiRec 入口发请求，用 `request_id` 关联 PaiRec 和 TRT 日志，汇总 p50/p95/p99/p9999/max，并按 `miss` / `hbm_hit` / `ds_hit` 分组
- 修正小样本 percentile 插值：2 个样本的 p50 使用中位数，不再错误取最小值
- 修复 PaiRec trace 采集：`glog` 默认写独立文件，`scripts/start_pairec.sh` 现在默认传入 `--alsologtostderr=true`，保留文件日志并可由 `tee /tmp/pairec.log` 捕获结构化日志

远程冷请求分解已确认：

- fallback 用户 `142`、`211` 均为 TRT `miss`
- 首个 fallback 请求：PaiRec `3392ms`，其中 `history_ms=2573ms`；对应首次加载 `999447` 用户 fallback JSON
- 后续稳态冷请求：PaiRec `687ms`，其中 TRT 服务 `685.3ms`，PaiRec 额外开销约 `2ms`
- 两次 TRT 平均 `750.0ms`，其中 8 轮 `runner_generate` 平均 `686.2ms`，占比约 `91.5%`
- TRT 次要耗时：`output_pad_ms` 平均 `27.3ms`、`prompt_ms` 平均 `10.7ms`
- 结论：fallback JSON 应在启动阶段预加载，避免首请求抖动；稳态冷请求的首要优化目标是 8 轮 TRT runner

本机验证：

```text
python -m py_compile inference/trt_llm/server.py inference/trt_llm/trt_qwen3_backend.py scripts/benchmark_e2e_latency.py
# exit 0

go test -mod=vendor ./services/...
# services / config / feature / recall 均通过

python -c '<trace parser assertions>'
# trace parser OK
```

## 6/2 F12：C++ DataSystem Get/Set 时延与尾延迟指标

已新增 `trtllm-datasystem-latency-trace.patch`，针对外部 TensorRT-LLM
`cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp` 增加结构化日志：

```text
[TensorRT-LLM][Datasystem][TRACE] op=offload ... create_ms=... d2h_ms=... set_ms=... total_ms=...
[TensorRT-LLM][Datasystem][TRACE] op=onboard ... get_ms=... h2d_ms=... total_ms=...
```

统计脚本同步增强：

- `scripts/test_trt_cpp_kv_offload.py`：按 warmup / pressure / replay / overall 汇总 DataSystem C++ 指标，HTTP 和 C++ 指标均输出 `avg/p50/p95/p99/p9999/max`，支持 `--json-output`
- `scripts/benchmark_e2e_latency.py`：PaiRec E2E、各阶段和 TRT 推荐结果缓存分组新增 `p9999`，缓存分组补齐 `p99/p9999/max`
- `p9999` 使用线性插值；样本量不足时仅作方向性观察，正式结论需扩大请求量

本机验证：

```text
git -C /home/vivwimp/TensorRT-LLM apply --check \
  /home/vivwimp/pairec4tigerllm/trtllm-datasystem-latency-trace.patch
# exit 0

python -m py_compile \
  scripts/benchmark_e2e_latency.py scripts/test_trt_cpp_kv_offload.py
# exit 0

python - <<'PY'
# synthetic percentile + DataSystem TRACE parser assertions
PY
# datasystem trace parser OK
```

### 远程 DataSystem 首轮实测

DataSystem runtime 已应用 trace patch 并完成首轮 pressure / replay：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=4181
  create_ms p50=0.680 p99=1.025 max=2.171
  d2h_ms    p50=0.413 p99=0.691 max=0.732
  set_ms    p50=0.759 p99=1.093 max=1.381
  total_ms  p50=1.937 p99=2.553 max=4.006
onboard count=1
  get_ms=1.562 h2d_ms=0.345 total_ms=1.961
```

结论：

- DataSystem C++ offload 写路径稳定，单个 3.5 MiB block 的 `Create + D2H + Set`
  总耗时 p50 约 `1.94ms`，p99 约 `2.55ms`
- onboard/Get 已出现 1 次，证明读路径打通；样本不足，暂不能评价 Get 分位数
- 原压测 verdict 依赖 DEBUG 级 `KV cache block reuse is enabled`、`copyBlock entered`
  和 `Set Key` 日志，在 INFO 级日志下会误报失败；脚本已改为优先使用结构化 trace 判定

### 远程 DataSystem 第二轮实测与 replay 修正

扩大 pressure 后采集到：

```text
KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=10440
  create_ms p50=0.765 p99=1.117 p9999=1.783 max=2.311
  d2h_ms    p50=0.493 p99=0.700 p9999=1.475 max=2.087
  set_ms    p50=0.839 p99=1.120 p9999=1.481 max=10.906
  total_ms  p50=2.200 p99=2.796 p9999=4.198 max=12.334
onboard count=0
```

结论：

- `Set` 路径已有 `10440` 个样本，p99 仍约 `1.12ms`；出现一次
  `set_ms=10.906ms` 尾部尖峰，后续 A/B 需保留 max 和原始 JSON
- 原 replay 固定重放最早的 pressure 历史；secondary pool 只有 `28` 个 block，
  增大 pressure 反而会使这些前缀更早被回收，无法稳定触发 `Get`
- `scripts/test_trt_cpp_kv_offload.py` 新增 `--replay-source-count` 和
  `--replay-tail-offset`，默认循环最近一小组 pressure 历史；历史生成器改为
  可逆 32 位混合，避免每 `256` 个 seed 重复，可构造十万级不同请求

### 远程 DataSystem 第三轮实测：Get/onboard 样本补齐

使用近期 pressure 历史循环 replay 后，结构化 trace 判定通过：

```text
PASS: C++ KV offload and DataSystem onboard were both observed.

KV pool: primaryBlocks=32 secondaryBlocks=28
offload count=8237
  set_ms    p50=0.780 p99=1.202 p9999=10.793 max=10.848
  total_ms  p50=2.101 p99=2.760 p9999=12.028 max=12.173
onboard count=177
  get_ms    p50=0.725 p95=1.520 p99=1.709 p9999=1.960 max=1.963
  h2d_ms    p50=0.492 p95=0.607 p99=0.618 p9999=0.624 max=0.624
  total_ms  p50=1.258 p95=2.027 p99=2.291 p9999=2.471 max=2.471
```

其中 replay/onboard wave 单独贡献 `174` 次 onboard，说明新的 replay 策略可以
稳定进入 DataSystem `Get + H2D` 读路径。`Get` 的 p50/p95/p99 已具备阶段性参考
价值；读路径只有 `177` 个样本，p9999 仍接近 max，仅作为观察值。

### Host/device 指标边界与 onboard 专项扩样

已展开 TensorRT-LLM runtime 实现：

```text
create_ms  host steady_clock 包围 DataSystem Create API
set_ms     host steady_clock 包围 DataSystem Set API
get_ms     host steady_clock 包围 DataSystem Get API
d2h_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy DeviceToHost
h2d_ms     host steady_clock 包围同步 cudaMemcpySanitized/cudaMemcpy HostToDevice
total_ms   host steady_clock 包围完整 offload/onboard 流程
```

因此：

- `create_ms/set_ms/get_ms` 是 host 侧 DataSystem API wall-clock
- `d2h_ms/h2d_ms` 是 host 观察到的同步 CPU↔GPU 传输完成耗时，包含 PCIe/DMA
  等待，但不是 CUDA event 计出的纯 device 时间
- 当前没有采集 GPU kernel 或 CUDA-event device-only 指标

`scripts/test_trt_cpp_kv_offload.py` 已增加：

- `--min-onboard-samples`：结构化 onboard trace 少于目标数量时返回失败
- 报告开头逐项打印 metric scope
- 报告结尾提示 p9999 样本门槛：`10000` 是最低尾部观测门槛，稳定结论建议
  `>=100000`

远程下一轮先采集 `>=10000` 个 onboard 样本：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_v4.log \
  --warmup-requests 0 \
  --requests 360 --repeat-requests 10000 \
  --replay-source-count 4 --replay-tail-offset 1 \
  --min-onboard-samples 10000 \
  --json-output /tmp/cppkv-datasystem-onboard-10k.json
```

### 远程 DataSystem 第四轮实测：onboard 10k+

onboard 专项 replay 已完成，达到最低 p9999 尾部观测门槛：

```text
replay/onboard wave HTTP:
  count=10000 p50=694ms p95=727ms p99=749ms p9999=832ms max=1552ms

all measured phases:
offload count=166089
  create_ms p50=0.665 p99=0.990 p9999=1.637 max=10.910
  d2h_ms    p50=0.483 p99=0.675 p9999=0.836 max=1.150
  set_ms    p50=0.756 p99=1.084 p9999=1.816 max=11.055
  total_ms  p50=1.937 p99=2.582 p9999=11.877 max=12.513
onboard count=14208
  get_ms    p50=0.740 p99=1.037 p9999=1.570 max=10.801
  h2d_ms    p50=0.407 p99=0.599 p9999=0.804 max=1.021
  total_ms  p50=1.177 p99=1.546 p9999=2.121 max=11.272
```

结论：

- onboard 已有 `14208` 个样本，可观察 p9999；若需要稳定的正式 p9999
  结论，仍建议扩大到 `>=100000`
- onboard 极端 max `11.272ms` 主要来自 `Get max=10.801ms`，不是 H2D；
  `h2d_ms max=1.021ms`
- replay 请求 E2E p50 为 `694ms`，单 block onboard p50 为 `1.177ms`；
  DataSystem 单块读回不是当前推荐请求的主耗时，主耗时仍在 8 轮 TRT runner
- offload 已有 `166089` 个样本，写路径 p9999 具备更好的统计支撑

### TRT runner 耗时根因与减轮数 A/B

当前每个 TRT miss 请求会串行调用 `8` 次 `ModelRunnerCpp.generate()`，每轮最多
生成 `32` 个 token，再合并 token 池做语义四元组组合。端到端基线中：

```text
runner_generate_ms avg=675.2ms
runner calls per request=8
single runner.generate avg≈84.4ms
```

8 轮不是模型推理的硬要求，而是当前无约束解码下为提高候选覆盖率采用的召回
策略。减少轮数预计近似线性降低 runner 耗时，但可能使可映射 item 数下降。

已增加：

- 服务参数 `--trt_num_samples`，环境变量 `TRT_NUM_SAMPLES`，默认 `8`
- TRT trace：`runner_calls`、`runner_avg_ms`、`runner_max_ms`
- `scripts/benchmark_e2e_latency.py` 汇总上述字段
- `scripts/test_trt_cpp_kv_offload.py` 输出每个 phase 的 `full_topk`、item 数
  p50/p95/min/max，便于直连 TRT 做质量-时延 A/B

粗略预估，固定开销按约 `23ms` 计算：

```text
8 rounds: runner≈675ms, request≈698ms  当前基线
4 rounds: runner≈338ms, request≈361ms  待实测
2 rounds: runner≈169ms, request≈192ms  待实测
1 round:  runner≈ 84ms, request≈107ms  待实测
```

远程完整 A/B 已回传，`/health` 与服务 TRACE 均确认轮数生效：

```text
sample health kv_exit calls ok/topk http_p50 http_p99 runner_avg runner_max
1      True   2       1     60/60  91.8     150.7    85.8       177.7
2      True   2       2     60/60  180.1    247.7    85.4       178.6
4      True   2       4     60/60  350.2    423.0    84.8       178.4
8      True   2       8     60/60  701.7    795.6    83.9       178.8
```

结论：

- 对当前直连 TRT、`topk=5`、60 请求样本，1/2/4/8 轮均没有出现 top-k 返回不足
- 1 轮相对 8 轮 HTTP p50 从 `701.7ms` 降到 `91.8ms`，下降约 `87%`
- 单次 `runner.generate()` 平均稳定在 `83.9-85.8ms`，说明主耗时几乎完全随轮数线性缩放
- `kv_exit=2` 来自底层 C++ KV/DataSystem verifier 未通过；该组结果只作为 runner
  轮数 A/B，不作为 DataSystem offload/onboard 结论

远程按 `TRT_NUM_SAMPLES=1/2/4/8` 分别重启服务，再执行相同直连请求序列：

```text
python scripts/test_trt_cpp_kv_offload.py \
  --log /tmp/server_samplesN.log \
  --warmup-requests 0 --requests 60 --repeat-requests 0 \
  --topk 5 --json-output /tmp/runner-samplesN.json
```

脚本默认使用带时间戳的唯一 `user_id` 前缀，避免重复运行命中 Python 结果缓存。
优先比较 `full_topk` 比例和 HTTP p50/p99。默认值暂不下调，需根据远程 A/B
结果选择。

已补充自动编排脚本，推荐后续直接使用：

```text
python scripts/benchmark_trt_runner_samples.py \
  --samples 1,2,4,8 \
  --server-cmd 'python -m inference.trt_llm.server ...' \
  --requests 60 --repeat-requests 0 --topk 5
```

如果需要清理旧服务，先在脚本外单独执行 `pkill -f "inference.trt_llm.server" || true`。
不要把这条 `pkill -f` 放进 `--stop-command`：benchmark 进程自己的
`--server-cmd` 参数也包含 `inference.trt_llm.server`，会被误杀并显示
`Terminated`。

脚本会逐轮注入 `TRT_NUM_SAMPLES`，等待 `/health` 返回匹配的
`trt_num_samples`，运行 `scripts/test_trt_cpp_kv_offload.py`，解析服务 TRACE
中的 `runner_calls/runner_avg_ms/runner_max_ms`，最后输出
`/tmp/trt_runner_samples_ab.json` 和 `/tmp/trt_runner_samples_ab.csv`。
默认情况下，runner A/B 脚本不会因为底层 C++ KV/DataSystem verifier 返回非零而
失败；`kv_exit` 会保留在汇总中作为诊断字段。若需要严格验证 C++ KV/DataSystem，
显式加 `--fail-on-kv-verdict`。

### TRT_NUM_SAMPLES=1 PaiRec E2E 实测

从 PaiRec `:18080` 入口使用 `TRT_NUM_SAMPLES=1`、`size=5`、60 个请求完成
端到端压测：

```text
Client: avg=9.4ms p50=3.5ms p95=90.5ms p99=92.8ms max=94.6ms
PaiRec total: avg=8.4ms p50=3.0ms p95=89.0ms p99=91.6ms max=94.0ms
GenerativeRecall http_ms: avg=8.1ms p50=2.0ms p95=89.0ms p99=91.2ms
TRT trace tr_total_ms: avg=7.5ms p50=1.6ms p95=88.2ms p99=90.4ms
TRT trace tr_runner_ms: avg=5.4ms p50=0.0ms p95=80.0ms p99=81.6ms
ok=60 fail=0
```

该报告混合了推荐结果缓存命中和少量冷 miss。p50 主要代表缓存命中路径；p95/p99
代表仍需进入 TRT runner 的冷 miss，符合 1 轮 runner 约 `84ms` 的预期。由于 TRT
服务日志未按 `request_id` 关联上，`scripts/benchmark_e2e_latency.py` 已补充
item 完整率和基于 GenerativeRecall `tr_result_cache_source` 的缓存分组 fallback。

后续重跑同一个 E2E 脚本时，会追加：

```text
== DataSystem C++ KV block stages ==
  scope: per KV block, from TRT-LLM C++ [Datasystem][TRACE], host wall-clock
  offload.create_ms / offload.d2h_ms / offload.set_ms / offload.total_ms
  onboard.get_ms / onboard.h2d_ms / onboard.total_ms
```

该部分是 per-block C++ KV 传输指标，不是 per-request 请求指标；需要与上面的
请求级 E2E 阶段并排解读。

后续可实验将多轮 prompt 批量提交给 `ModelRunnerCpp.generate()`。当前本地
TensorRT-LLM API 对整批默认共用一个 sampling config，而现有逻辑依赖每轮不同
seed；批量化需要先验证候选多样性，不能直接替换串行循环。

## 下一步

`dev` 已作为端到端阶段性基线保留。后续在专用分支开展 C++ DataSystem A/B：

1. 分开采集 `TRT_NUM_SAMPLES=1` 的纯 cold/miss E2E 与缓存命中 E2E，避免一个报告里 p50/p99 语义混杂
2. 单独恢复 DataSystem C++ KV verifier：当前 runner A/B 中 `kv_exit=2`，不能作为 DataSystem 结论
3. 如需稳定的正式 onboard p9999 结论，再扩大到 `>=100000` 个 onboard 样本
4. 推理服务增加实验开关，关闭 TRT Python 结果缓存和无效的 Python KV Cache 查询，确保请求进入 C++ runner
5. TensorRT-LLM runtime 恢复 KV 配置参数化，确保两组使用相同 primary / secondary block 数
6. 基于同一份 engine 构建原生 pinned DRAM baseline，与当前 DataSystem runtime 对照
7. 后续独立优化 fallback JSON 启动预加载
