#pragma once

namespace pairec::kvc_burst
{

template <typename Options>
auto SetExclusiveConnectionIfSupported(Options& options, bool enabled, int)
    -> decltype(options.enableExclusiveConnection = enabled, void())
{
    options.enableExclusiveConnection = enabled;
}

template <typename Options>
void SetExclusiveConnectionIfSupported(Options&, bool, long)
{
}

template <typename Options>
void SetExclusiveConnectionIfSupported(Options& options, bool enabled)
{
    SetExclusiveConnectionIfSupported(options, enabled, 0);
}

} // namespace pairec::kvc_burst
