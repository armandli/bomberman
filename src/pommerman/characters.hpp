#pragma once
#include <array>
#include <optional>
#include <vector>
#include "types.hpp"

namespace pommerman {

struct Bomb {
    int agent_id       = -1;
    Position position  = {};
    int life           = DEFAULT_BOMB_LIFE;
    int blast_strength = DEFAULT_BLAST_STRENGTH;
    std::optional<Direction> moving_direction;

    // Returns true when life reaches zero
    bool tick() noexcept { return --life <= 0; }
    bool exploded() const noexcept { return life <= 0; }
    bool is_moving() const noexcept { return moving_direction.has_value(); }
    void stop() noexcept { moving_direction = std::nullopt; }
    void fire() noexcept { life = 0; }

    // Blast positions in all 4 cardinal directions, stopping at board edge.
    // Does NOT filter by wall type — caller handles wall blocking.
    std::vector<Position> raw_blast_positions(int board_size = BOARD_SIZE) const;
};

struct Flame {
    Position position = {};
    int life          = FLAME_LIFE;

    bool tick() noexcept { return --life <= 0; }
    bool is_dead() const noexcept { return life <= 0; }
};

struct Agent {
    int agent_id       = -1;
    Position position  = {};
    int ammo           = 1;
    bool is_alive      = true;
    int blast_strength = DEFAULT_BLAST_STRENGTH;
    bool can_kick      = false;
    GameType game_type = GameType::FFA;
    int teammate_id    = -1;
    std::array<int, 3> enemy_ids{-1, -1, -1};

    void set_agent_id(int id, GameType gtype) noexcept;
    // Creates a bomb if ammo > 0, decrements ammo, returns the bomb
    std::optional<Bomb> maybe_lay_bomb() noexcept;
    void pick_up(Item item) noexcept;
    void die() noexcept { is_alive = false; }
    void reset() noexcept;
    void incr_ammo() noexcept { if (ammo < MAX_AMMO) ++ammo; }
};

} // namespace pommerman
