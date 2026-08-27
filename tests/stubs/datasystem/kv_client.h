#pragma once

#include <optional>
#include <string>
#include <vector>

namespace datasystem
{

struct Status
{
    bool IsError() const { return false; }
    std::string ToString() const { return "ok"; }
};

struct Buffer
{
    size_t GetSize() const { return 0; }
    const void* ImmutableData() const { return nullptr; }
};

template <typename T>
using Optional = std::optional<T>;

struct ConnectOptions
{
    std::string host;
    int port{0};
    bool enableCrossNodeConnection{false};
    bool enableExclusiveConnection{false};
};

enum class WriteMode
{
    NONE_L2_CACHE_EVICT,
};

struct SetParam
{
    WriteMode writeMode{WriteMode::NONE_L2_CACHE_EVICT};
    uint32_t ttlSecond{0};
};

class StringView
{
public:
    explicit StringView(const std::string&)
    {
    }
};

class KVClient
{
public:
    explicit KVClient(const ConnectOptions&)
    {
    }

    Status Init() { return {}; }
    Status Get(const std::string&, Optional<Buffer>&, int) { return {}; }
    Status Set(const std::string&, const StringView&, const SetParam&) { return {}; }
    Status Del(const std::vector<std::string>&, std::vector<std::string>&) { return {}; }
};

} // namespace datasystem
