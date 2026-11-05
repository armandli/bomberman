#include "random_agent.hpp"

namespace pommerman {

RandomAgent::RandomAgent(uint32_t seed)
    : _rng(seed == 0 ? std::random_device{}() : seed) {}

Action RandomAgent::act(const Observation& /*obs*/) {
    return static_cast<Action>(_dist(_rng));
}

} // namespace pommerman
