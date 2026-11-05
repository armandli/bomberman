#pragma once
#include <random>
#include "base_agent.hpp"

namespace pommerman {

class RandomAgent final : public BaseAgent {
public:
    explicit RandomAgent(uint32_t seed = 0);
    Action act(const Observation& obs) override;

private:
    std::mt19937 _rng;
    std::uniform_int_distribution<int> _dist{0, 5}; // Action::Stop..Action::Bomb
};

} // namespace pommerman
