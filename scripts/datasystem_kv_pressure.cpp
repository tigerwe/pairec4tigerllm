#include <datasystem/kv_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace
{

std::atomic<bool> gStop{false};

void HandleSignal(int)
{
    gStop.store(true, std::memory_order_relaxed);
}

struct Config
{
    std::string mode{"mixed"};
    std::string host{"127.0.0.1"};
    int port{18482};
    size_t objectSize{3584U * 1024U};
    size_t keyCount{256};
    size_t getClients{4};
    size_t setClients{6};
    int durationSeconds{0};
    int reportIntervalSeconds{1};
    std::string prefix{"KvcPressure"};
    std::string readyFile;
    std::string statsFile;
    bool cleanupKeys{true};
};

struct Counters
{
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> success{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> latencyUs{0};
    std::atomic<uint64_t> maxLatencyUs{0};
    std::atomic<uint64_t> activeCalls{0};
    std::atomic<uint64_t> maxActiveCalls{0};
};

void UpdateMaximum(std::atomic<uint64_t>& target, uint64_t value)
{
    auto current = target.load(std::memory_order_relaxed);
    while (current < value
        && !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed))
    {
    }
}

bool ParseSize(const std::string& value, size_t* output)
{
    try
    {
        size_t used = 0;
        auto parsed = std::stoull(value, &used);
        if (used != value.size() || parsed == 0)
        {
            return false;
        }
        *output = static_cast<size_t>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ParseInt(const std::string& value, int* output)
{
    try
    {
        size_t used = 0;
        auto parsed = std::stoi(value, &used);
        if (used != value.size())
        {
            return false;
        }
        *output = parsed;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ParseBool(const std::string& value, bool* output)
{
    if (value == "1" || value == "true")
    {
        *output = true;
        return true;
    }
    if (value == "0" || value == "false")
    {
        *output = false;
        return true;
    }
    return false;
}

bool ParseArgs(int argc, char** argv, Config* config)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        auto split = arg.find('=');
        if (split == std::string::npos || arg.rfind("--", 0) != 0)
        {
            std::cerr << "invalid argument: " << arg << std::endl;
            return false;
        }
        auto name = arg.substr(2, split - 2);
        auto value = arg.substr(split + 1);
        if (name == "mode")
        {
            config->mode = value;
        }
        else if (name == "host")
        {
            config->host = value;
        }
        else if (name == "port")
        {
            if (!ParseInt(value, &config->port)) return false;
        }
        else if (name == "object_size")
        {
            if (!ParseSize(value, &config->objectSize)) return false;
        }
        else if (name == "key_count")
        {
            if (!ParseSize(value, &config->keyCount)) return false;
        }
        else if (name == "get_clients")
        {
            if (!ParseSize(value, &config->getClients)) return false;
        }
        else if (name == "set_clients")
        {
            if (!ParseSize(value, &config->setClients)) return false;
        }
        else if (name == "duration_seconds")
        {
            if (!ParseInt(value, &config->durationSeconds)) return false;
        }
        else if (name == "report_interval_seconds")
        {
            if (!ParseInt(value, &config->reportIntervalSeconds)) return false;
        }
        else if (name == "prefix")
        {
            config->prefix = value;
        }
        else if (name == "ready_file")
        {
            config->readyFile = value;
        }
        else if (name == "stats_file")
        {
            config->statsFile = value;
        }
        else if (name == "cleanup_keys")
        {
            if (!ParseBool(value, &config->cleanupKeys)) return false;
        }
        else
        {
            std::cerr << "unknown argument: " << name << std::endl;
            return false;
        }
    }
    if (config->mode != "get" && config->mode != "set" && config->mode != "mixed")
    {
        std::cerr << "mode must be get, set, or mixed" << std::endl;
        return false;
    }
    if (config->host.empty() || config->port <= 0 || config->port > 65535 || config->keyCount == 0
        || config->objectSize == 0 || config->durationSeconds < 0 || config->reportIntervalSeconds <= 0)
    {
        std::cerr << "invalid host/port/count/size/duration/report interval" << std::endl;
        return false;
    }
    size_t workers = (config->mode == "get" ? config->getClients
                                              : config->mode == "set" ? config->setClients
                                                                      : config->getClients + config->setClients);
    if (workers == 0 || workers > 128)
    {
        std::cerr << "total clients must be between 1 and 128" << std::endl;
        return false;
    }
    return true;
}

std::unique_ptr<datasystem::KVClient> CreateClient(const Config& config, bool exclusive)
{
    datasystem::ConnectOptions options;
    options.host = config.host;
    options.port = config.port;
    options.enableCrossNodeConnection = true;
    options.enableExclusiveConnection = exclusive;
    auto client = std::make_unique<datasystem::KVClient>(options);
    auto status = client->Init();
    if (status.IsError())
    {
        std::cerr << "KVClient Init failed: " << status.ToString() << std::endl;
        return nullptr;
    }
    return client;
}

std::vector<std::string> BuildKeys(const Config& config, const std::string& kind)
{
    std::vector<std::string> keys;
    keys.reserve(config.keyCount);
    for (size_t i = 0; i < config.keyCount; ++i)
    {
        keys.emplace_back(config.prefix + "_" + kind + "_" + std::to_string(i));
    }
    return keys;
}

bool Prefill(datasystem::KVClient& client, const std::vector<std::string>& keys, const std::string& value)
{
    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    for (const auto& key : keys)
    {
        auto status = client.Set(key, datasystem::StringView(value), param);
        if (status.IsError())
        {
            std::cerr << "prefill failed key=" << key << " detail=" << status.ToString() << std::endl;
            return false;
        }
    }
    return true;
}

void RecordCall(Counters& counters, bool ok, uint64_t bytes, uint64_t latencyUs)
{
    counters.calls.fetch_add(1, std::memory_order_relaxed);
    (ok ? counters.success : counters.errors).fetch_add(1, std::memory_order_relaxed);
    if (ok)
    {
        counters.bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    counters.latencyUs.fetch_add(latencyUs, std::memory_order_relaxed);
    UpdateMaximum(counters.maxLatencyUs, latencyUs);
}

void RunGetWorker(const Config& config, size_t workerIndex, datasystem::KVClient& client,
    const std::vector<std::string>& keys, std::atomic<bool>& start, std::atomic<size_t>& initialized, Counters& counters)
{
    initialized.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire) && !gStop.load(std::memory_order_relaxed))
    {
        std::this_thread::yield();
    }
    size_t index = workerIndex % keys.size();
    while (!gStop.load(std::memory_order_relaxed))
    {
        auto active = counters.activeCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        UpdateMaximum(counters.maxActiveCalls, active);
        auto begin = std::chrono::steady_clock::now();
        datasystem::Optional<datasystem::Buffer> buffer;
        auto status = client.Get(keys[index], buffer, 0);
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
        counters.activeCalls.fetch_sub(1, std::memory_order_relaxed);
        bool ok = !status.IsError() && static_cast<bool>(buffer);
        RecordCall(counters, ok, ok ? config.objectSize : 0, static_cast<uint64_t>(elapsed));
        index = (index + config.getClients) % keys.size();
    }
}

void RunSetWorker(const Config& config, size_t workerIndex, datasystem::KVClient& client,
    const std::vector<std::string>& keys, const std::string& value, std::atomic<bool>& start,
    std::atomic<size_t>& initialized, Counters& counters)
{
    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    initialized.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire) && !gStop.load(std::memory_order_relaxed))
    {
        std::this_thread::yield();
    }
    size_t index = workerIndex % keys.size();
    while (!gStop.load(std::memory_order_relaxed))
    {
        auto active = counters.activeCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        UpdateMaximum(counters.maxActiveCalls, active);
        auto begin = std::chrono::steady_clock::now();
        auto status = client.Set(keys[index], datasystem::StringView(value), param);
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
        counters.activeCalls.fetch_sub(1, std::memory_order_relaxed);
        bool ok = !status.IsError();
        RecordCall(counters, ok, ok ? config.objectSize : 0, static_cast<uint64_t>(elapsed));
        index = (index + config.setClients) % keys.size();
    }
}

void WriteStats(const Config& config, const Counters& counters,
    std::chrono::steady_clock::time_point started, bool final)
{
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    auto calls = counters.calls.load(std::memory_order_relaxed);
    auto success = counters.success.load(std::memory_order_relaxed);
    auto errors = counters.errors.load(std::memory_order_relaxed);
    auto bytes = counters.bytes.load(std::memory_order_relaxed);
    auto latencyUs = counters.latencyUs.load(std::memory_order_relaxed);
    double qps = elapsed > 0.0 ? static_cast<double>(calls) / elapsed : 0.0;
    double gbps = elapsed > 0.0 ? static_cast<double>(bytes) * 8.0 / elapsed / 1.0e9 : 0.0;
    double avgMs = calls > 0 ? static_cast<double>(latencyUs) / static_cast<double>(calls) / 1000.0 : 0.0;
    std::string line = "{\"event\":\"kvc_pressure_stats\",\"mode\":\"" + config.mode
        + "\",\"final\":" + (final ? "true" : "false") + ",\"elapsed_s\":" + std::to_string(elapsed)
        + ",\"calls\":" + std::to_string(calls) + ",\"success\":" + std::to_string(success)
        + ",\"errors\":" + std::to_string(errors) + ",\"bytes\":" + std::to_string(bytes)
        + ",\"qps\":" + std::to_string(qps) + ",\"gbps\":" + std::to_string(gbps)
        + ",\"avg_ms\":" + std::to_string(avgMs) + ",\"max_ms\":"
        + std::to_string(static_cast<double>(counters.maxLatencyUs.load(std::memory_order_relaxed)) / 1000.0)
        + ",\"active_calls\":" + std::to_string(counters.activeCalls.load(std::memory_order_relaxed))
        + ",\"max_active_calls\":" + std::to_string(counters.maxActiveCalls.load(std::memory_order_relaxed))
        + "}";
    std::cout << line << std::endl;
    if (!config.statsFile.empty())
    {
        std::ofstream output(config.statsFile, std::ios::app);
        output << line << '\n';
    }
}

void DeleteKeys(datasystem::KVClient& client, const std::vector<std::string>& keys)
{
    std::vector<std::string> failed;
    auto status = client.Del(keys, failed);
    if (status.IsError())
    {
        std::cerr << "cleanup delete failed: " << status.ToString() << std::endl;
    }
}

} // namespace

int main(int argc, char** argv)
{
    Config config;
    if (!ParseArgs(argc, argv, &config))
    {
        return 2;
    }
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    auto getKeys = BuildKeys(config, "get");
    auto setKeys = BuildKeys(config, "set");
    std::string value(config.objectSize, 'a');
    auto controlClient = CreateClient(config, false);
    if (!controlClient)
    {
        return 1;
    }
    if ((config.mode == "get" || config.mode == "mixed") && !Prefill(*controlClient, getKeys, value))
    {
        return 1;
    }
    if ((config.mode == "set" || config.mode == "mixed") && !Prefill(*controlClient, setKeys, value))
    {
        return 1;
    }

    size_t getWorkers = config.mode == "set" ? 0 : config.getClients;
    size_t setWorkers = config.mode == "get" ? 0 : config.setClients;
    std::vector<std::unique_ptr<datasystem::KVClient>> clients;
    clients.reserve(getWorkers + setWorkers);
    for (size_t i = 0; i < getWorkers + setWorkers; ++i)
    {
        auto client = CreateClient(config, true);
        if (!client)
        {
            return 1;
        }
        clients.emplace_back(std::move(client));
    }

    std::atomic<bool> start{false};
    std::atomic<size_t> initialized{0};
    Counters counters;
    std::vector<std::thread> workers;
    workers.reserve(clients.size());
    for (size_t i = 0; i < getWorkers; ++i)
    {
        workers.emplace_back(RunGetWorker, std::cref(config), i, std::ref(*clients[i]), std::cref(getKeys),
            std::ref(start), std::ref(initialized), std::ref(counters));
    }
    for (size_t i = 0; i < setWorkers; ++i)
    {
        workers.emplace_back(RunSetWorker, std::cref(config), i, std::ref(*clients[getWorkers + i]),
            std::cref(setKeys), std::cref(value), std::ref(start), std::ref(initialized), std::ref(counters));
    }
    while (initialized.load(std::memory_order_acquire) != workers.size())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto started = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    auto readyDeadline = started + std::chrono::seconds(30);
    while (!gStop.load(std::memory_order_relaxed)
        && (counters.success.load(std::memory_order_relaxed) < workers.size()
            || counters.maxActiveCalls.load(std::memory_order_relaxed) < workers.size())
        && std::chrono::steady_clock::now() < readyDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (counters.maxActiveCalls.load(std::memory_order_relaxed) < workers.size())
    {
        std::cerr << "failed to reach requested concurrent calls: requested=" << workers.size()
                  << " observed=" << counters.maxActiveCalls.load(std::memory_order_relaxed) << std::endl;
        gStop.store(true, std::memory_order_relaxed);
        for (auto& worker : workers)
        {
            worker.join();
        }
        return 1;
    }
    if (!config.readyFile.empty())
    {
        std::ofstream ready(config.readyFile);
        ready << "pid=" << ::getpid() << " workers=" << workers.size()
              << " max_active_calls=" << counters.maxActiveCalls.load(std::memory_order_relaxed) << '\n';
    }
    std::cout << "{\"event\":\"kvc_pressure_ready\",\"mode\":\"" << config.mode
              << "\",\"workers\":" << workers.size() << ",\"max_active_calls\":"
              << counters.maxActiveCalls.load(std::memory_order_relaxed) << "}" << std::endl;

    auto deadline = config.durationSeconds > 0
        ? started + std::chrono::seconds(config.durationSeconds)
        : std::chrono::steady_clock::time_point::max();
    while (!gStop.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::seconds(config.reportIntervalSeconds));
        WriteStats(config, counters, started, false);
    }
    gStop.store(true, std::memory_order_relaxed);
    for (auto& worker : workers)
    {
        worker.join();
    }
    WriteStats(config, counters, started, true);

    if (!config.readyFile.empty())
    {
        std::remove(config.readyFile.c_str());
    }
    if (config.cleanupKeys)
    {
        if (getWorkers > 0) DeleteKeys(*controlClient, getKeys);
        if (setWorkers > 0) DeleteKeys(*controlClient, setKeys);
    }
    return counters.errors.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}
