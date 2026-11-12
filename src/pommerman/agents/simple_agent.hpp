#pragma once
#include <array>
#include <optional>
#include <random>
#include "base_agent.hpp"

namespace pommerman {

// Rule-based baseline opponent (in the spirit of the classic Pommerman
// SimpleAgent). Priorities, in order:
//   1. If standing in a cell a bomb will hit (or on a flame), step towards the
//      nearest cell that is safe in time.
//   2. Lay a bomb when an enemy is in blast range or wood is adjacent, but only
//      if an escape route exists after placing it.
//   3. Walk (avoiding danger) to the nearest powerup, else to a cell next to
//      wood, else towards the nearest enemy.
//   4. Otherwise stop.
// Fully observable boards only (fog cells are treated as walls).
class SimpleAgent final : public BaseAgent {
public:
    explicit SimpleAgent(uint32_t seed = 0);
    Action act(const Observation& obs) override;

    // Steps until each cell is hit by a blast (INT_MAX = never). Flames count
    // as 0. Takes chain reactions into account. Exposed for testing.
    using DangerMap = std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE>;
    static DangerMap danger_map(const Observation& obs,
                                std::optional<Position> extra_bomb = std::nullopt,
                                int extra_strength = 0);

private:
    std::mt19937 _rng;
};

} // namespace pommerman
