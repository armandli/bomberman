#include "characters.hpp"
#include <array>

namespace pommerman {

std::vector<Position> Bomb::raw_blast_positions(int board_size) const {
    std::vector<Position> result;
    result.reserve(1 + 4 * blast_strength);
    result.push_back(position);

    constexpr std::array<std::pair<int,int>, 4> dirs{{{-1,0},{1,0},{0,-1},{0,1}}};
    for (auto [dr, dc] : dirs) {
        for (int s = 1; s <= blast_strength; ++s) {
            int r = position.row + dr * s;
            int c = position.col + dc * s;
            if (r < 0 || r >= board_size || c < 0 || c >= board_size) break;
            result.push_back({r, c});
        }
    }
    return result;
}

void Agent::set_agent_id(int id, GameType gtype) noexcept {
    agent_id  = id;
    game_type = gtype;

    if (gtype == GameType::FFA || gtype == GameType::OneVsOne) {
        teammate_id = -1;
        // All other agents are enemies
        int ei = 0;
        for (int i = 0; i < NUM_AGENTS && ei < 3; ++i) {
            if (i != id) enemy_ids[ei++] = i;
        }
    } else {
        // Team / TeamRadio: agents 0&2 vs 1&3
        teammate_id = (id % 2 == 0) ? id + 1 : id - 1;
        int ei = 0;
        for (int i = 0; i < NUM_AGENTS && ei < 3; ++i) {
            if (i != id && i != teammate_id) enemy_ids[ei++] = i;
        }
    }
}

std::optional<Bomb> Agent::maybe_lay_bomb() noexcept {
    if (ammo <= 0) return std::nullopt;
    --ammo;
    return Bomb{agent_id, position, DEFAULT_BOMB_LIFE, blast_strength, std::nullopt};
}

void Agent::pick_up(Item item) noexcept {
    switch (item) {
        case Item::ExtraBomb: incr_ammo(); break;
        case Item::IncrRange: ++blast_strength; break;
        case Item::Kick:      can_kick = true; break;
        default: break;
    }
}

void Agent::reset() noexcept {
    ammo           = 1;
    is_alive       = true;
    blast_strength = DEFAULT_BLAST_STRENGTH;
    can_kick       = false;
}

} // namespace pommerman
