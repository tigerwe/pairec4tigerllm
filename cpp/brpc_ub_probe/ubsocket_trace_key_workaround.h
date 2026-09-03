#pragma once

#include <iostream>

#if defined(BRPC_WITH_URMA)
#include <bthread/bthread.h>
#endif

namespace pairec::brpc_ub_probe
{

// The integrated bRPC tree declares the UBSocket trace keys but does not
// allocate them. Register process-lifetime keys before any UB Channel/Server
// operation so Channel::CallMethod does not use the hard-coded placeholders.
inline bool InitializeUBSocketTraceKeys(
    const char* readyMarker = "MINIMAL_RECOMMEND_UB_TRACE_KEYS_READY")
{
#if defined(BRPC_WITH_URMA)
    const int rpcidResult = bthread_key_create(&ubsocket_trace_rpcid_key, nullptr);
    if (rpcidResult != 0)
    {
        std::cerr << "Failed to create UBSocket RPC ID trace key: " << rpcidResult << std::endl;
        return false;
    }

    const int timestampResult =
        bthread_key_create(&ubsocket_trace_call_timestamp, nullptr);
    if (timestampResult != 0)
    {
        (void)bthread_key_delete(ubsocket_trace_rpcid_key);
        std::cerr << "Failed to create UBSocket call timestamp trace key: " << timestampResult
                  << std::endl;
        return false;
    }

    std::cout << readyMarker << std::endl;
#endif
    return true;
}

} // namespace pairec::brpc_ub_probe
