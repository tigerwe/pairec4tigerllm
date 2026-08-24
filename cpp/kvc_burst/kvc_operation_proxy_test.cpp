#include "kvc_operation_proxy.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include <cassert>
#include <atomic>
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
        pairec::kvc_burst::Store(&control->pressure_start_ns[0],
            pairec::kvc_burst::MonotonicNs());
        pairec::kvc_burst::FetchAdd(&control->pressure_started_lanes, 1U);
        pairec::kvc_burst::FutexWake(&control->pressure_started_lanes);
        while (pairec::kvc_burst::Load(&control->business_release_generation) != generation)
        {
            std::this_thread::yield();
        }
        _exit(released ? 0 : 1);
    }

    auto token = pairec::kvc_burst::beginBusinessGet("request-1", BusinessApi::kMGet, 2);
    assert(token.triggered());
    assert(token.generation == 1);
    assert(token.bytes == 7340032);
    assert(token.barrierReleased);
    assert(token.pressureEstablished);
    assert(control->business_release_generation == 1);
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
    control->pressure_started_lanes = 0;
    control->business_release_generation = 0;
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

    assert(::setenv("PAIREC_KVC_INPROCESS_BURST", "1", 1) == 0);
    control->configured_concurrency = 32;
    control->pressure_lanes = 31;
    control->pressure_key_count = 4;
    control->keys_verified = 31;
    control->clients_connected = 0;
    control->object_size_bytes = 3670016;
    control->prepared_generation = 3;
    control->run_generation = 0;
    control->state = static_cast<uint32_t>(State::kReady);
    control->arrived_participants = 0;
    control->release_generation = 0;
    control->barrier_failed = 0;
    control->pressure_started_lanes = 0;
    control->completed_pressure_lanes = 0;
    control->pressure_success = 0;
    control->pressure_errors = 0;
    control->pressure_first_failed = 0;
    control->business_done_generation = 0;
    control->business_get_started_count = 0;
    control->business_get_completed_count = 0;
    control->business_get_success_count = 0;
    control->pressure_command_generation = 0;
    control->pressure_established_generation = 0;
    control->pressure_stop_generation = 0;
    control->pressure_stopped_generation = 0;
    control->pressure_active_gets = 0;
    control->pressure_completed_gets = 0;
    control->pressure_completions_after_stop = 0;
    control->pressure_max_duration_ms = 5000;
    control->pressure_max_loops = 1000;
    control->business_start_ns = 0;
    control->business_end_ns = 0;
    for (uint32_t lane = 0; lane < 31; ++lane)
    {
        control->pressure_start_ns[lane] = 0;
        control->pressure_end_ns[lane] = 0;
        control->pressure_ok[lane] = 0;
    }
    std::atomic<uint32_t> fakeGets{0};
    control->trigger_armed = 0;
    auto registration = pairec::kvc_burst::beginBusinessGet(
        "request-registration", BusinessApi::kGet, 1);
    assert(!registration.triggered());
    auto pressure = pairec::kvc_burst::beginInProcessPressure(registration,
        [&fakeGets](uint32_t lane, std::string const& key) {
            assert(key == "PairecKvcBurstV2_g3_pressure_" + std::to_string(lane % 4));
            fakeGets.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return true;
        });
    assert(!pressure.active());
    assert(control->pressure_client_registered == 1);
    control->trigger_armed = 1;
    auto firstGet = pairec::kvc_burst::beginBusinessGet(
        "request-inprocess", BusinessApi::kGet, 1);
    assert(firstGet.triggered());
    assert(firstGet.inProcessPressure);
    assert(firstGet.businessGetOrdinal == 1);
    assert(control->pressure_established_generation == 3);
    assert(control->pressure_started_lanes == 31);
    assert(control->pressure_active_gets >= 30);
    assert(control->completed_pressure_lanes == 0);
    assert(control->pressure_first_failed == 0);
    assert(control->business_start_ns > 0);
    pairec::kvc_burst::finishBusinessGet(
        firstGet, true, pairec::kvc_burst::MonotonicNs());
    assert(control->business_done_generation == 0);

    auto secondGet = pairec::kvc_burst::beginBusinessGet(
        "request-inprocess", BusinessApi::kGet, 1);
    assert(secondGet.triggered());
    assert(secondGet.businessGetOrdinal == 2);
    auto const stopStarted = std::chrono::steady_clock::now();
    pairec::kvc_burst::finishBusinessGet(
        secondGet, true, pairec::kvc_burst::MonotonicNs());
    auto const stopElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - stopStarted);
    assert(stopElapsed.count() < 5);
    assert(control->pressure_stop_generation == 3);
    assert(control->pressure_active_at_stop >= 30);
    assert(control->business_get_completed_count == 2);
    assert(control->business_get_success_count == 2);
    assert(control->business_done_generation == 3);

    pairec::kvc_burst::observeAddTokenStart("request-inprocess");
    pairec::kvc_burst::observeAddTokenEnd("request-inprocess");
    assert(control->add_token_observation_count == 1);
    assert(control->pressure_active_at_first_add_token > 0);
    pairec::kvc_burst::observeBusinessRequestComplete("request-inprocess");
    assert(control->business_lifecycle_done_generation == 3);
    assert(control->business_lifecycle_done_ns > 0);

    auto const stoppedDeadline = pairec::kvc_burst::DeadlineNs(1000);
    while (control->pressure_stopped_generation != 3
        && pairec::kvc_burst::MonotonicNs() < stoppedDeadline)
    {
        std::this_thread::yield();
    }
    assert(control->pressure_stopped_generation == 3);
    assert(fakeGets.load(std::memory_order_relaxed) >= 31);
    assert(control->completed_pressure_lanes == 31);
    assert(control->pressure_success == 31);
    assert(control->pressure_errors == 0);
    assert(control->pressure_completions_after_stop > 0);
    assert(control->pressure_last_end_ns >= control->pressure_stop_ns);

    assert(::munmap(control, sizeof(*control)) == 0);
    assert(::close(fd) == 0);
    assert(::unlink(path.c_str()) == 0);
    std::cout << "KVC_OPERATION_PROXY_TEST_OK" << std::endl;
    return 0;
}
