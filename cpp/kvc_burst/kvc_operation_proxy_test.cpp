#include "kvc_operation_proxy.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

namespace
{

using pairec::kvc_burst::BusinessApi;
using pairec::kvc_burst::SharedControl;
using pairec::kvc_burst::State;
using pairec::kvc_burst::TriggerStatus;

} // namespace

int main()
{
    std::string path = "/tmp/kvc_operation_proxy_test." + std::to_string(::getpid());
    auto fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    assert(fd >= 0);
    assert(::ftruncate(fd, sizeof(SharedControl)) == 0);
    auto address = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(address != MAP_FAILED);
    auto* control = static_cast<SharedControl*>(address);
    std::memset(control, 0, sizeof(*control));
    control->magic = pairec::kvc_burst::kMagic;
    control->version = pairec::kvc_burst::kVersion;
    control->struct_size = sizeof(*control);
    control->configured_concurrency = 2;
    control->pressure_lanes = 1;
    control->barrier_timeout_ms = 100;
    control->prepared_generation = 1;
    control->object_size_bytes = 3670016;
    control->keys_verified = 1;
    control->clients_connected = 1;
    control->trigger_armed = 0;
    control->state = static_cast<uint32_t>(State::kReady);
    assert(::setenv("KVC_BURST_ENABLED", "1", 1) == 0);
    assert(::setenv("KVC_BURST_CONTROL_PATH", path.c_str(), 1) == 0);

    auto disarmed = pairec::kvc_burst::beginBusinessGet("request-disarmed", BusinessApi::kGet, 1);
    assert(!disarmed.triggered());
    assert(disarmed.status == TriggerStatus::kSkippedDisarmed);
    control->trigger_armed = 1;

    auto child = ::fork();
    assert(child >= 0);
    if (child == 0)
    {
        uint32_t generation = 0;
        assert(pairec::kvc_burst::WaitForGenerationChange(&control->run_generation, 0, &generation));
        uint64_t waitUs = 0;
        auto released = pairec::kvc_burst::ArriveAndWait(control, generation, 100, &waitUs);
        _exit(released ? 0 : 1);
    }

    auto token = pairec::kvc_burst::beginBusinessGet("request-1", BusinessApi::kMGet, 2);
    assert(token.triggered());
    assert(token.generation == 1);
    assert(token.bytes == 7340032);
    assert(token.barrierReleased);
    pairec::kvc_burst::finishBusinessGet(token, true);
    assert(control->business_api == static_cast<uint32_t>(BusinessApi::kMGet));
    assert(control->business_key_count == 2);
    assert(control->business_done_generation == 1);
    assert(control->business_success == 1);

    int status = 0;
    assert(::waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    control->state = static_cast<uint32_t>(State::kReady);
    auto duplicate = pairec::kvc_burst::beginBusinessGet("request-1", BusinessApi::kGet, 1);
    assert(!duplicate.triggered());
    assert(duplicate.status == TriggerStatus::kSkippedAlreadyTriggered);

    control->state = static_cast<uint32_t>(State::kRunning);
    auto busy = pairec::kvc_burst::beginBusinessGet("request-2", BusinessApi::kGet, 1);
    assert(!busy.triggered());
    assert(busy.status == TriggerStatus::kSkippedBusy);

    control->state = static_cast<uint32_t>(State::kReady);
    control->prepared_generation = 2;
    control->arrived_participants = 0;
    control->release_generation = 0;
    control->barrier_failed = 0;
    std::thread triggering([&] {
        auto waiting = pairec::kvc_burst::beginBusinessGet("request-3", BusinessApi::kGet, 1);
        assert(waiting.triggered());
        assert(!waiting.barrierReleased);
        pairec::kvc_burst::finishBusinessGet(waiting, true);
    });
    while (pairec::kvc_burst::Load(&control->state) != static_cast<uint32_t>(State::kRunning))
    {
        std::this_thread::yield();
    }
    auto started = std::chrono::steady_clock::now();
    auto bypass = pairec::kvc_burst::beginBusinessGet("request-4", BusinessApi::kGet, 1);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    assert(bypass.status == TriggerStatus::kSkippedBusy);
    assert(elapsed.count() < 20);
    triggering.join();

    assert(::munmap(control, sizeof(*control)) == 0);
    assert(::close(fd) == 0);
    assert(::unlink(path.c_str()) == 0);
    std::cout << "KVC_OPERATION_PROXY_TEST_OK" << std::endl;
    return 0;
}
