// stub — 让 PipelineRH2DQueueConsumer::ConsumerLoop 立即返回
// 编译: gcc -shared -fPIC -o block_ds_consumer.so stub_consumer.c
// 用法: LD_PRELOAD=./block_ds_consumer.so:libabseil_dll.so...

// C++ mangled: _ZN11OsXprtPipln25PipelineRH2DQueueConsumer12ConsumerLoopEv
// demangled:   OsXprtPipln::PipelineRH2DQueueConsumer::ConsumerLoop()

void _ZN11OsXprtPipln25PipelineRH2DQueueConsumer12ConsumerLoopEv(void) {
    // no-op: 线程立即返回，不访问共享内存
}
