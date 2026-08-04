#include "kvc_burst_shared.h"

#include <sys/mman.h>
#include <sys/wait.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <unistd.h>

namespace
{

using pairec::kvc_burst::SharedControl;

void Initialize(SharedControl* control, uint32_t concurrency)
{
    std::memset(control, 0, sizeof(*control));
    control->magic = pairec::kvc_burst::kMagic;
    control->version = pairec::kvc_burst::kVersion;
    control->struct_size = sizeof(*control);
    control->configured_concurrency = concurrency;
    control->pressure_lanes = concurrency - 1;
    control->barrier_timeout_ms = 100;
}

} // namespace

int main()
{
    auto memory = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    assert(memory != MAP_FAILED);
    auto* control = static_cast<SharedControl*>(memory);
    Initialize(control, 4);

    auto child = ::fork();
    assert(child >= 0);
    if (child == 0)
    {
        std::vector<std::thread> workers;
        for (int i = 0; i < 3; ++i)
        {
            workers.emplace_back([control]() {
                uint64_t waitUs = 0;
                assert(pairec::kvc_burst::ArriveAndWait(control, 1, 100, &waitUs));
            });
        }
        for (auto& worker : workers) worker.join();
        _exit(0);
    }

    while (pairec::kvc_burst::Load(&control->arrived_participants) < 3)
    {
        std::this_thread::yield();
    }
    uint64_t waitUs = 0;
    assert(pairec::kvc_burst::ArriveAndWait(control, 1, 100, &waitUs));
    int status = 0;
    assert(::waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(pairec::kvc_burst::Load(&control->release_generation) == 1);
    assert(pairec::kvc_burst::Load(&control->barrier_failed) == 0);

    Initialize(control, 2);
    assert(!pairec::kvc_burst::ArriveAndWait(control, 2, 2, &waitUs));
    assert(pairec::kvc_burst::Load(&control->barrier_failed) == 1);

    ::munmap(control, sizeof(SharedControl));
    std::cout << "KVC_BURST_SHARED_TEST_OK" << std::endl;
    return 0;
}
