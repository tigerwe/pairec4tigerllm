#include <datasystem/kv_client.h>

#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

constexpr uint64_t kDefaultObjectSize = 3670016ULL;
constexpr uint64_t kDefaultSeed = 0x4b56435542504f43ULL;

struct Config
{
    std::string host;
    int port{0};
    uint64_t objectSize{kDefaultObjectSize};
    uint32_t iterations{1};
    uint64_t seed{kDefaultSeed};
    std::string prefix{"PairecKvcUbIntegrity"};
    int32_t connectTimeoutMs{30000};
    int32_t requestTimeoutMs{30000};
    bool cleanup{true};
    bool selfTest{false};
};

void PrintUsage(const char* program)
{
    std::cout << "Usage: " << program << " --host HOST --port PORT [options]\n"
              << "  --object_size BYTES       default 3670016 (3.5 MiB)\n"
              << "  --iterations N           default 1\n"
              << "  --seed N                 deterministic payload seed\n"
              << "  --prefix KEY_PREFIX      default PairecKvcUbIntegrity\n"
              << "  --connect_timeout_ms N   default 30000\n"
              << "  --request_timeout_ms N   default 30000\n"
              << "  --cleanup true|false     delete each key after validation\n"
              << "  --self_test              test payload and SHA-256 without DataSystem\n";
}

bool ParseUnsigned(const std::string& text, uint64_t* output)
{
    try
    {
        size_t used = 0;
        auto value = std::stoull(text, &used, 0);
        if (used != text.size()) return false;
        *output = value;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ParseBool(const std::string& text, bool* output)
{
    if (text == "1" || text == "true")
    {
        *output = true;
        return true;
    }
    if (text == "0" || text == "false")
    {
        *output = false;
        return true;
    }
    return false;
}

bool IsValidPrefix(const std::string& prefix)
{
    if (prefix.empty() || prefix.size() > 180) return false;
    return std::all_of(prefix.begin(), prefix.end(), [](unsigned char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
    });
}

bool ParseArgs(int argc, char** argv, Config* config)
{
    for (int index = 1; index < argc; ++index)
    {
        std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h")
        {
            PrintUsage(argv[0]);
            return false;
        }
        if (argument == "--self_test")
        {
            config->selfTest = true;
            continue;
        }
        if (argument.rfind("--", 0) != 0)
        {
            std::cerr << "invalid argument: " << argument << std::endl;
            return false;
        }

        std::string name;
        std::string value;
        auto split = argument.find('=');
        if (split != std::string::npos)
        {
            name = argument.substr(2, split - 2);
            value = argument.substr(split + 1);
        }
        else
        {
            name = argument.substr(2);
            if (index + 1 >= argc)
            {
                std::cerr << "missing value for --" << name << std::endl;
                return false;
            }
            value = argv[++index];
        }

        uint64_t parsed = 0;
        if (name == "host") config->host = value;
        else if (name == "port")
        {
            if (!ParseUnsigned(value, &parsed) || parsed == 0 || parsed > 65535) return false;
            config->port = static_cast<int>(parsed);
        }
        else if (name == "object_size")
        {
            if (!ParseUnsigned(value, &config->objectSize) || config->objectSize == 0) return false;
        }
        else if (name == "iterations")
        {
            if (!ParseUnsigned(value, &parsed) || parsed == 0
                || parsed > std::numeric_limits<uint32_t>::max()) return false;
            config->iterations = static_cast<uint32_t>(parsed);
        }
        else if (name == "seed")
        {
            if (!ParseUnsigned(value, &config->seed)) return false;
        }
        else if (name == "prefix") config->prefix = value;
        else if (name == "connect_timeout_ms")
        {
            if (!ParseUnsigned(value, &parsed) || parsed == 0
                || parsed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return false;
            config->connectTimeoutMs = static_cast<int32_t>(parsed);
        }
        else if (name == "request_timeout_ms")
        {
            if (!ParseUnsigned(value, &parsed) || parsed == 0
                || parsed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return false;
            config->requestTimeoutMs = static_cast<int32_t>(parsed);
        }
        else if (name == "cleanup")
        {
            if (!ParseBool(value, &config->cleanup)) return false;
        }
        else
        {
            std::cerr << "unknown argument: --" << name << std::endl;
            return false;
        }
    }

    if (config->selfTest) return true;
    if (config->host.empty() || config->port == 0 || !IsValidPrefix(config->prefix))
    {
        std::cerr << "host, port, and a valid prefix are required" << std::endl;
        return false;
    }
    return true;
}

uint64_t NextWord(uint64_t* state)
{
    *state += 0x9e3779b97f4a7c15ULL;
    uint64_t value = *state;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::string DeterministicPayload(uint64_t size, uint64_t seed)
{
    std::string payload(size, '\0');
    uint64_t state = seed;
    for (uint64_t offset = 0; offset < size; offset += sizeof(uint64_t))
    {
        auto word = NextWord(&state);
        auto count = static_cast<size_t>(std::min<uint64_t>(sizeof(word), size - offset));
        std::memcpy(payload.data() + offset, &word, count);
    }
    return payload;
}

std::string Sha256(const void* data, size_t size)
{
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char*>(data), size, digest);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : digest) output << std::setw(2) << static_cast<unsigned int>(byte);
    return output.str();
}

std::string KeyFor(const Config& config, uint32_t iteration)
{
    std::ostringstream key;
    key << config.prefix << '_' << std::setw(6) << std::setfill('0') << iteration;
    return key.str();
}

double ElapsedMs(std::chrono::steady_clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

double Percentile99(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    auto index = static_cast<size_t>((values.size() - 1) * 0.99);
    return values[index];
}

double Average(const std::vector<double>& values)
{
    double total = 0.0;
    for (double value : values) total += value;
    return values.empty() ? 0.0 : total / static_cast<double>(values.size());
}

void DeleteKey(datasystem::KVClient& client, const std::string& key)
{
    std::vector<std::string> failedKeys;
    auto status = client.Del(std::vector<std::string>{key}, failedKeys);
    if (status.IsError() || !failedKeys.empty())
    {
        std::cerr << "cleanup failed key=" << key << " status=" << status.ToString() << std::endl;
    }
}

bool RunSelfTest()
{
    const std::string abc = "abc";
    const std::string expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    auto first = DeterministicPayload(4099, 1234);
    auto second = DeterministicPayload(4099, 1234);
    auto different = DeterministicPayload(4099, 1235);
    bool valid = Sha256(abc.data(), abc.size()) == expected && first == second && first != different
        && Sha256(first.data(), first.size()) == Sha256(second.data(), second.size());
    std::cout << (valid ? "KVC_UB_INTEGRITY_SELF_TEST_OK" : "KVC_UB_INTEGRITY_SELF_TEST_FAILED")
              << std::endl;
    return valid;
}

} // namespace

int main(int argc, char** argv)
{
    Config config;
    if (!ParseArgs(argc, argv, &config)) return 2;
    if (config.selfTest) return RunSelfTest() ? 0 : 1;

    datasystem::ConnectOptions options;
    options.host = config.host;
    options.port = config.port;
    options.connectTimeoutMs = config.connectTimeoutMs;
    options.requestTimeoutMs = config.requestTimeoutMs;
    options.enableCrossNodeConnection = true;
    options.enableExclusiveConnection = true;

    datasystem::KVClient client(options);
    auto initStatus = client.Init();
    if (initStatus.IsError())
    {
        std::cerr << "KVClient Init failed: " << initStatus.ToString() << std::endl;
        return 1;
    }

    datasystem::SetParam setParam;
    setParam.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    std::vector<double> setLatencyMs;
    std::vector<double> getLatencyMs;

    for (uint32_t iteration = 0; iteration < config.iterations; ++iteration)
    {
        auto key = KeyFor(config, iteration);
        auto payload = DeterministicPayload(config.objectSize, config.seed + iteration);
        auto expectedSha = Sha256(payload.data(), payload.size());

        auto started = std::chrono::steady_clock::now();
        auto setStatus = client.Set(key, datasystem::StringView(payload), setParam);
        auto setMs = ElapsedMs(started);
        if (setStatus.IsError())
        {
            std::cerr << "Set failed key=" << key << " status=" << setStatus.ToString() << std::endl;
            return 1;
        }

        datasystem::Optional<datasystem::Buffer> buffer;
        started = std::chrono::steady_clock::now();
        auto getStatus = client.Get(key, buffer, 0);
        auto getMs = ElapsedMs(started);
        if (getStatus.IsError() || !buffer)
        {
            std::cerr << "Get failed key=" << key << " status=" << getStatus.ToString() << std::endl;
            if (config.cleanup) DeleteKey(client, key);
            return 1;
        }

        auto actualSize = static_cast<uint64_t>(buffer->GetSize());
        const void* actualData = buffer->ImmutableData();
        auto actualSha = actualData == nullptr ? std::string() : Sha256(actualData, buffer->GetSize());
        bool equal = actualSize == config.objectSize && actualData != nullptr
            && std::memcmp(payload.data(), actualData, payload.size()) == 0 && actualSha == expectedSha;
        if (!equal)
        {
            std::cerr << "integrity mismatch key=" << key << " expected_size=" << config.objectSize
                      << " actual_size=" << actualSize << " expected_sha256=" << expectedSha
                      << " actual_sha256=" << (actualSha.empty() ? "null" : actualSha) << std::endl;
            if (config.cleanup) DeleteKey(client, key);
            return 1;
        }

        setLatencyMs.push_back(setMs);
        getLatencyMs.push_back(getMs);
        std::cout << "{\"event\":\"kvc_ub_integrity_iteration\",\"iteration\":" << iteration
                  << ",\"key\":\"" << key << "\",\"bytes\":" << actualSize
                  << ",\"sha256\":\"" << actualSha << "\",\"set_ms\":" << std::fixed
                  << std::setprecision(3) << setMs << ",\"get_ms\":" << getMs
                  << ",\"valid\":true}" << std::endl;

        if (config.cleanup) DeleteKey(client, key);
    }

    std::cout << "{\"event\":\"kvc_ub_integrity_summary\",\"host\":\"" << config.host
              << "\",\"port\":" << config.port << ",\"prefix\":\"" << config.prefix
              << "\",\"iterations\":" << config.iterations << ",\"object_size\":"
              << config.objectSize << ",\"set_avg_ms\":" << std::fixed << std::setprecision(3)
              << Average(setLatencyMs) << ",\"set_p99_ms\":" << Percentile99(setLatencyMs)
              << ",\"get_avg_ms\":" << Average(getLatencyMs) << ",\"get_p99_ms\":"
              << Percentile99(getLatencyMs) << ",\"valid\":true}" << std::endl;
    std::cout << "KVC_UB_INTEGRITY_PASS" << std::endl;
    return 0;
}
