#pragma once
#include <array>
#include <optional>
#include <vector>
#include "constants.hpp"

namespace pommerman {

struct Position {
    int row = 0;
    int col = 0;

    bool operator==(const Position&) const = default;
    bool operator<(const Position& o) const noexcept {
        return row < o.row || (row == o.row && col < o.col);
    }
};

using Board = std::array<std::array<Item, BOARD_SIZE>, BOARD_SIZE>;

struct Observation {
    Board board{};
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> bomb_blast_strength{};
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> bomb_life{};
    // 0 = not moving, else maps to Direction int value
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> bomb_moving_direction{};
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> flame_life{};
    Position position;
    int ammo                = 0;
    int blast_strength      = DEFAULT_BLAST_STRENGTH;
    bool can_kick           = false;
    int agent_id            = -1;
    std::vector<int> alive_agents;
    std::optional<int> teammate;
    std::array<int, 3> enemies{-1, -1, -1};
};

struct GameInfo {
    Result result = Result::Incomplete;
    std::vector<int> winners;
};

} // namespace pommerman
