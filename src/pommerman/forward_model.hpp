#pragma once
#include <map>
#include <span>
#include <vector>
#include "characters.hpp"
#include "types.hpp"

namespace pommerman {

class ForwardModel {
public:
    // Advances game state one tick in-place.
    // static_board: contains only Rigid/Wood/Passage/revealed powerups;
    //               modified when wood is destroyed.
    // hidden_items: (row*BOARD_SIZE+col) → powerup hidden under wood;
    //               entries removed as wood burns.
    // board:        fully assembled board, rebuilt at the end of step.
    void step(std::span<const Action> actions,
              Board& board,
              std::vector<Agent>& agents,
              std::vector<Bomb>& bombs,
              std::vector<Flame>& flames,
              Board& static_board,
              std::map<int, Item>& hidden_items);

    bool    is_done(std::span<const Agent> agents, int step_count,
                    GameType game_type) const noexcept;

    std::vector<double> get_rewards(std::span<const Agent> agents,
                                    GameType game_type,
                                    bool done, int step_count) const;

    GameInfo get_info(std::span<const Agent> agents,
                      GameType game_type, int step_count) const;
};

} // namespace pommerman
