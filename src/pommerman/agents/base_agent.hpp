#pragma once
#include "../types.hpp"

namespace pommerman {

class BaseAgent {
public:
    virtual ~BaseAgent() = default;

    // Called once before each episode begins.
    virtual void episode_start(int agent_id) { _agent_id = agent_id; }

    // Choose an action given the current observation.
    virtual Action act(const Observation& obs) = 0;

    // Called once when the episode ends.
    virtual void episode_end(double /*reward*/) {}

    int agent_id() const noexcept { return _agent_id; }

protected:
    int _agent_id = -1;
};

} // namespace pommerman
