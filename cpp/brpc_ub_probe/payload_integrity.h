#pragma once

#include <openssl/evp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pairec::brpc_ub_probe
{

inline uint64_t NextPayloadWord(uint64_t* state)
{
    *state += 0x9e3779b97f4a7c15ULL;
    uint64_t value = *state;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

inline std::string BuildDeterministicPayload(size_t size, uint64_t seed)
{
    std::string payload(size, '\0');
    uint64_t state = seed;
    for (size_t offset = 0; offset < size; offset += sizeof(uint64_t))
    {
        const uint64_t word = NextPayloadWord(&state);
        const size_t bytes = std::min(sizeof(word), size - offset);
        std::memcpy(payload.data() + offset, &word, bytes);
    }
    return payload;
}

inline std::string Sha256Hex(const std::string& payload)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digestSize = 0;
    if (EVP_Digest(payload.data(), payload.size(), digest, &digestSize, EVP_sha256(), nullptr) != 1)
    {
        throw std::runtime_error("EVP_Digest(SHA-256) failed");
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digestSize; ++i)
    {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
}

inline std::string BuildPayloadMetadata(
    const std::string& method, size_t payloadSize, const std::string& sha256)
{
    std::ostringstream output;
    output << "{\"method\":\"" << method << "\",\"payload_bytes\":" << payloadSize
           << ",\"payload_sha256\":\"" << sha256 << "\"}";
    return output.str();
}

} // namespace pairec::brpc_ub_probe
