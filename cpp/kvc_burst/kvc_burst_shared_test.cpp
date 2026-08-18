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
    control->object_size_bytes = 3670016;
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
    assert(pairec::kvc_burst::IsCompatible(*control));
    assert(std::string(pairec::kvc_burst::BusinessApiName(
               pairec::kvc_burst::BusinessApi::kParallelGet))
        == "parallel_get");

    Initialize(control, 2);
    assert(!pairec::kvc_burst::ArriveAndWait(control, 2, 2, &waitUs));
    assert(pairec::kvc_burst::Load(&control->barrier_failed) == 1);

    // BusinessSubmitRank: no business Get recorded -> 0.
    Initialize(control, 4);
    assert(pairec::kvc_burst::BusinessSubmitRank(*control) == 0);
    // Two of three pressure lanes started before the business Get -> rank 3.
    pairec::kvc_burst::Store(&control->pressure_start_ns[0], uint64_t{100});
    pairec::kvc_burst::Store(&control->pressure_start_ns[1], uint64_t{300});
    pairec::kvc_burst::Store(&control->pressure_start_ns[2], uint64_t{200});
    pairec::kvc_burst::Store(&control->business_start_ns, uint64_t{250});
    assert(pairec::kvc_burst::BusinessSubmitRank(*control) == 3);
    // Business Get first -> rank 1; lanes without a recorded start are ignored.
    pairec::kvc_burst::Store(&control->business_start_ns, uint64_t{50});
    pairec::kvc_burst::Store(&control->pressure_start_ns[2], uint64_t{0});
    assert(pairec::kvc_burst::BusinessSubmitRank(*control) == 1);
    // Business Get last -> rank lanes+1.
    pairec::kvc_burst::Store(&control->pressure_start_ns[2], uint64_t{400});
    pairec::kvc_burst::Store(&control->business_start_ns, uint64_t{500});
    assert(pairec::kvc_burst::BusinessSubmitRank(*control) == 4);

    // SustainedStopReason precedence: business done > loop budget > duration.
    auto loopStarted = pairec::kvc_burst::MonotonicNs();
    assert(pairec::kvc_burst::SustainedStopReason(*control, 7, loopStarted, 0, 1000, 100)
        == pairec::kvc_burst::SustainedStop::kNone);
    pairec::kvc_burst::Store(&control->business_done_generation, 7U);
    assert(pairec::kvc_burst::SustainedStopReason(*control, 7, loopStarted, 0, 1000, 100)
        == pairec::kvc_burst::SustainedStop::kBusinessDone);
    pairec::kvc_burst::Store(&control->business_done_generation, 0U);
    assert(pairec::kvc_burst::SustainedStopReason(*control, 7, loopStarted, 100, 1000, 100)
        == pairec::kvc_burst::SustainedStop::kMaxLoops);
    auto longAgo = loopStarted - 2000000000ULL;
    assert(pairec::kvc_burst::SustainedStopReason(*control, 7, longAgo, 0, 1000, 100)
        == pairec::kvc_burst::SustainedStop::kMaxDuration);
    // A different generation must not stop this loop.
    pairec::kvc_burst::Store(&control->business_done_generation, 8U);
    assert(pairec::kvc_burst::SustainedStopReason(*control, 7, loopStarted, 0, 1000, 100)
        == pairec::kvc_burst::SustainedStop::kNone);

    ::munmap(control, sizeof(SharedControl));
    std::cout << "KVC_BURST_SHARED_TEST_OK" << std::endl;
    return 0;
}
