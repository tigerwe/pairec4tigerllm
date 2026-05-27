// stub: 绕过 DataSystem SDK 0.7.7 ARM GPU ID 解析 bug
// 问题: OsXprtPipln::CudaRH2DDriver::SwitchToAndGetGpuId 对 ARM GPU 空标识做 substr(4, ...) 崩溃
// 绕过: 直接返回 GPU 0
// 编译: gcc -shared -fPIC -o stub_gpu.so stub_gpu.c
// 用法: 加入 LD_PRELOAD 链，在 block_ds_consumer.so 之后

// OsXprtPipln::CudaRH2DDriver::SwitchToAndGetGpuId(std::string const&) -> int
int _ZN11OsXprtPipln14CudaRH2DDriver19SwitchToAndGetGpuIdERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void) {
    return 0;
}

// OsXprtPipln::SwitchToAndGetGpuId(std::string const&) -> int
int _ZN11OsXprtPipln19SwitchToAndGetGpuIdERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(void) {
    return 0;
}
