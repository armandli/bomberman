#pragma once
#include <map>
#include <random>
#include <span>
#include <tuple>
#include <vector>
#include "characters.hpp"
#include "forward_model.hpp"
#include "types.hpp"

namespace pommerman {

class Environment {
public:
    explicit Environment(GameType game_type         = GameType::FFA,
                         bool    partially_observable = false,
                         int     board_size           = BOARD_SIZE,
                         uint32_t seed               = 0);

    // Reset to a fresh game, return initial per-agent observations.
    std::vector<Observation> reset();

    // Advance one tick with the given actions (one per agent, indexed 0-3).
    // Returns: (observations, rewards, done, info).
    std::tuple<std::vector<Observation>, std::vector<double>, bool, GameInfo>
    step(std::span<const Action> actions);

    std::vector<Observation> get_observations() const;

    // ── State accessors for rendering ────────────────────────────────────────
    const Board&              board()      const noexcept { return _board; }
    const std::vector<Agent>& agents()     const noexcept { return _agents; }
    const std::vector<Bomb>&  bombs()      const noexcept { return _bombs; }
    const std::vector<Flame>& flames()     const noexcept { return _flames; }
    int                       step_count() const noexcept { return _step_count; }
    bool                      is_done()    const noexcept { return _done; }
    const GameInfo&           last_info()  const noexcept { return _last_info; }
    GameType                  game_type()  const noexcept { return _game_type; }

private:
    Board              _board{};
    Board              _static_board{};   // rigid + wood + revealed items
    std::vector<Agent> _agents;
    std::vector<Bomb>  _bombs;
    std::vector<Flame> _flames;
    std::map<int,Item> _hidden_items;     // pos_key → powerup under wood
    int                _step_count      = 0;
    GameType           _game_type;
    bool               _partially_observable;
    int                _board_size;
    bool               _done            = false;
    GameInfo           _last_info;
    std::mt19937       _rng;
    ForwardModel       _fm;

    Observation _build_observation(const Agent& agent) const;
};

} // namespace pommerman
