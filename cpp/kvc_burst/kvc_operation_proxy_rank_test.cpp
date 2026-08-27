#include "kvc_operation_proxy.h"

#include <fcntl.h>
#include <sys/mman.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

int main()
{
    using namespace pairec::kvc_burst;

    std::string path = "/tmp/kvc_operation_proxy_rank_test." + std::to_string(::getpid());
    auto fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    assert(fd >= 0);
    assert(::ftruncate(fd, sizeof(SharedControl)) == 0);
    auto address = ::mmap(nullptr, sizeof(SharedControl), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(address != MAP_FAILED);
    auto* control = static_cast<SharedControl*>(address);
    std::memset(control, 0, sizeof(*control));
    control->magic = kMagic;
    control->version = kVersion;
    control->struct_size = sizeof(*control);
    control->configured_concurrency = 32;
    control->pressure_lanes = 31;
    control->pressure_key_count = 4;
    control->barrier_timeout_ms = 100;
    control->prepared_generation = 1;
    control->object_size_bytes = 8388608;
    control->keys_verified = 31;
    control->pressure_max_duration_ms = 2000;
    control->pressure_max_loops = 1;
    control->trigger_armed = 1;
    control->state = static_cast<uint32_t>(State::kReady);

    assert(::setenv("KVC_BURST_ENABLED", "1", 1) == 0);
    assert(::setenv("KVC_BURST_CONTROL_PATH", path.c_str(), 1) == 0);
    assert(::setenv("KVC_BURST_EXPECTED_BUSINESS_GETS", "1", 1) == 0);
    assert(::setenv("KVC_BURST_EXPECTED_OBJECT_BYTES", "8388608", 1) == 0);
    assert(::setenv("PAIREC_KVC_INPROCESS_BURST", "1", 1) == 0);
    assert(::setenv("PAIREC_KVC_INPROCESS_PRESSURE_PREFIX", "PairecRankKvcBurstV1", 1) == 0);

    std::atomic<uint32_t> pressureGets{0};
    assert(registerInProcessPressureClient(
        [&pressureGets](uint32_t lane, std::string const& key) {
            assert(key == "PairecRankKvcBurstV1_g1_pressure_" + std::to_string(lane % 4));
            pressureGets.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return true;
        }));

    auto business = beginBusinessGet("rank-request-1", BusinessApi::kGet, 1);
    assert(business.triggered());
    assert(business.inProcessPressure);
    assert(business.businessGetOrdinal == 1);
    assert(business.bytes == 8388608);
    assert(business.pressureEstablished);
    assert(control->pressure_started_lanes == 31);
    assert(control->pressure_active_gets == 31);
    assert(control->business_pressure_active_snapshot == 31);

    finishBusinessGet(business, true, MonotonicNs());
    observeBusinessRequestComplete("rank-request-1");
    assert(control->business_get_completed_count == 1);
    assert(control->business_get_success_count == 1);
    assert(control->business_success == 1);
    assert(control->business_done_generation == 1);
    assert(control->business_lifecycle_done_generation == 1);
    assert(control->pressure_active_at_stop == 31);

    auto deadline = DeadlineNs(1000);
    while (control->pressure_stopped_generation != 1 && MonotonicNs() < deadline)
    {
        std::this_thread::yield();
    }
    assert(control->pressure_stopped_generation == 1);
    assert(control->completed_pressure_lanes == 31);
    assert(control->pressure_success == 31);
    assert(control->pressure_errors == 0);
    assert(pressureGets.load(std::memory_order_relaxed) == 31);

    shutdownInProcessPressureClient();
    assert(::munmap(control, sizeof(*control)) == 0);
    assert(::close(fd) == 0);
    assert(::unlink(path.c_str()) == 0);
    std::cout << "KVC_OPERATION_PROXY_RANK_TEST_OK" << std::endl;
    return 0;
}
