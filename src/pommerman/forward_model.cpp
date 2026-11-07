#include "forward_model.hpp"
#include "utility.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <set>

namespace pommerman {

// ── helpers local to this translation unit ───────────────────────────────────

static int pos_key(Position p) noexcept { return p.row * BOARD_SIZE + p.col; }

// Blast positions accounting for static walls: stops before Rigid, includes
// (and stops at) the first Wood tile. Does not modify the board, so every bomb
// exploding in the same tick sees the same pre-explosion walls.
static std::vector<Position> compute_blast(const Bomb& bomb, const Board& static_board) {
    std::vector<Position> result;
    result.push_back(bomb.position);

    constexpr std::array<std::pair<int,int>, 4> dirs{{{-1,0},{1,0},{0,-1},{0,1}}};
    for (auto [dr, dc] : dirs) {
        for (int s = 1; s <= bomb.blast_strength; ++s) {
            Position p{bomb.position.row + dr*s, bomb.position.col + dc*s};
            if (!position_on_board(p)) break;
            if (static_board[p.row][p.col] == Item::Rigid) break;
            result.push_back(p);
            if (static_board[p.row][p.col] == Item::Wood) break; // blast does not pass through wood
        }
    }
    return result;
}

static bool blocks_movement(const Board& static_board, Position p) noexcept {
    return !position_on_board(p) ||
           static_board[p.row][p.col] == Item::Rigid ||
           static_board[p.row][p.col] == Item::Wood;
}

// ── ForwardModel::step ───────────────────────────────────────────────────────

void ForwardModel::step(std::span<const Action> actions,
                        Board& board,
                        std::vector<Agent>& agents,
                        std::vector<Bomb>& bombs,
                        std::vector<Flame>& flames,
                        Board& static_board,
                        std::map<int, Item>& hidden_items) {

    const int nagents = static_cast<int>(agents.size());
    const int nbombs  = static_cast<int>(bombs.size());

    // ── 1. Tick flames ────────────────────────────────────────────────────────
    {
        std::vector<Flame> alive;
        alive.reserve(flames.size());
        for (auto& f : flames) {
            if (!f.tick()) {
                alive.push_back(f);
            } else {
                // Flame expired: restore cell from static board (already passage/item)
                // static_board already has the right value; nothing extra to do here.
            }
        }
        flames = std::move(alive);
    }

    // ── 2. Compute desired agent positions ────────────────────────────────────
    std::vector<Position> curr(nagents);
    std::vector<Position> desired(nagents);
    std::vector<bool> laying_bomb(nagents, false);

    for (int i = 0; i < nagents; ++i) {
        curr[i] = agents[i].position;
        if (!agents[i].is_alive) { desired[i] = curr[i]; continue; }

        Action act = (i < (int)actions.size()) ? actions[i] : Action::Stop;
        if (act == Action::Bomb) {
            laying_bomb[i] = true;
            desired[i] = curr[i];
        } else {
            Direction dir = action_to_direction(act);
            Position  nxt = get_next_position(curr[i], dir);
            if (blocks_movement(static_board, nxt)) {
                desired[i] = curr[i];
            } else {
                desired[i] = nxt;
            }
        }
    }

    // ── 3. Compute desired bomb positions (moving bombs) ──────────────────────
    std::vector<Position> bcurr(nbombs), bdesired(nbombs);
    for (int b = 0; b < nbombs; ++b) {
        bcurr[b] = bombs[b].position;
        if (bombs[b].is_moving()) {
            Position nxt = get_next_position(bcurr[b], *bombs[b].moving_direction);
            if (blocks_movement(static_board, nxt)) {
                bombs[b].stop();
                bdesired[b] = bcurr[b];
            } else {
                bdesired[b] = nxt;
            }
        } else {
            bdesired[b] = bcurr[b];
        }
    }

    // ── 4. Crossing detection: agents swapping positions → both revert ────────
    for (int i = 0; i < nagents; ++i) {
        if (!agents[i].is_alive) continue;
        for (int j = i+1; j < nagents; ++j) {
            if (!agents[j].is_alive) continue;
            if (desired[i] == curr[j] && desired[j] == curr[i]) {
                desired[i] = curr[i];
                desired[j] = curr[j];
            }
        }
    }

    // Crossing detection: moving bombs swapping → both stop
    for (int i = 0; i < nbombs; ++i) {
        for (int j = i+1; j < nbombs; ++j) {
            if (bdesired[i] == bcurr[j] && bdesired[j] == bcurr[i]) {
                bombs[i].stop(); bdesired[i] = bcurr[i];
                bombs[j].stop(); bdesired[j] = bcurr[j];
            }
        }
    }

    // ── 5. Kick logic ─────────────────────────────────────────────────────────
    for (int i = 0; i < nagents; ++i) {
        if (!agents[i].is_alive || desired[i] == curr[i]) continue;

        // Does the desired cell currently contain a stationary bomb?
        for (int b = 0; b < nbombs; ++b) {
            if (bdesired[b] != bcurr[b]) continue; // already moving
            if (desired[i] != bcurr[b])  continue;

            if (agents[i].can_kick) {
                Direction kick_dir = get_direction(curr[i], desired[i]);
                Position  kick_dest = get_next_position(bcurr[b], kick_dir);
                bool blocked = blocks_movement(static_board, kick_dest);
                if (!blocked) {
                    // Check no other bomb already at kick_dest
                    for (int bb = 0; bb < nbombs; ++bb) {
                        if (bb != b && bcurr[bb] == kick_dest) { blocked = true; break; }
                    }
                }
                if (!blocked) {
                    bombs[b].moving_direction = kick_dir;
                    bdesired[b] = kick_dest;
                } else {
                    bombs[b].stop();
                    desired[i] = curr[i]; // agent blocked too
                }
            } else {
                desired[i] = curr[i]; // no kick ability
            }
            break;
        }
    }

    // ── 6. Occupancy resolution (iterative) ───────────────────────────────────
    // Reverting one mover can create a new conflict at its old cell, so repeat
    // until stable. Every pass only reverts, so this terminates.
    //  - alive agents targeting the same cell all revert
    //  - bombs targeting the same cell: the moving ones revert and stop
    //  - agent and bomb targeting the same cell: whichever moved reverts
    //    (a moving bomb stops when it runs into an agent; an agent cannot
    //    walk onto a bomb that did not get out of the way)
    bool changed = true;
    while (changed) {
        changed = false;
        std::map<Position, std::vector<int>> pos_map;
        for (int i = 0; i < nagents; ++i) {
            if (agents[i].is_alive) pos_map[desired[i]].push_back(i);
        }
        for (auto& [pos, group] : pos_map) {
            if (group.size() > 1) {
                for (int i : group) {
                    if (desired[i] != curr[i]) { desired[i] = curr[i]; changed = true; }
                }
            }
        }

        for (int a = 0; a < nbombs; ++a) {
            for (int b = a + 1; b < nbombs; ++b) {
                if (bdesired[a] != bdesired[b]) continue;
                for (int x : {a, b}) {
                    if (bdesired[x] != bcurr[x]) {
                        bdesired[x] = bcurr[x];
                        bombs[x].stop();
                        changed = true;
                    }
                }
            }
        }

        for (int i = 0; i < nagents; ++i) {
            if (!agents[i].is_alive) continue;
            for (int b = 0; b < nbombs; ++b) {
                if (desired[i] != bdesired[b]) continue;
                if (desired[i] != curr[i]) { desired[i] = curr[i]; changed = true; }
                if (bdesired[b] != bcurr[b]) {
                    bdesired[b] = bcurr[b];
                    bombs[b].stop();
                    changed = true;
                }
            }
        }
    }

    // ── 7. Apply movements ────────────────────────────────────────────────────
    for (int i = 0; i < nagents; ++i) {
        if (agents[i].is_alive) agents[i].position = desired[i];
    }
    for (int b = 0; b < nbombs; ++b) {
        bombs[b].position = bdesired[b];
    }

    // ── 8. Powerup pickup ─────────────────────────────────────────────────────
    for (auto& agent : agents) {
        if (!agent.is_alive) continue;
        Item cell = static_board[agent.position.row][agent.position.col];
        if (cell == Item::ExtraBomb || cell == Item::IncrRange || cell == Item::Kick) {
            agent.pick_up(cell);
            static_board[agent.position.row][agent.position.col] = Item::Passage;
        }
    }

    // ── 9. Lay bombs ──────────────────────────────────────────────────────────
    for (int i = 0; i < nagents; ++i) {
        if (!agents[i].is_alive || !laying_bomb[i]) continue;
        // Don't place if a bomb already occupies this cell
        bool occupied = false;
        for (auto& bomb : bombs) {
            if (bomb.position == agents[i].position) { occupied = true; break; }
        }
        if (!occupied) {
            if (auto b = agents[i].maybe_lay_bomb()) bombs.push_back(*b);
        }
    }

    // ── 10. Tick bombs ────────────────────────────────────────────────────────
    for (auto& bomb : bombs) bomb.tick();

    // ── 11. Chain explosion ───────────────────────────────────────────────────
    // A bomb explodes when its timer runs out, when it sits in a still-burning
    // flame (e.g. it was kicked into one), or when another exploding bomb's
    // wall-aware blast reaches it.
    std::set<std::pair<int,int>> flame_set;
    for (auto& f : flames) flame_set.insert({f.position.row, f.position.col});

    const int nbombs2 = static_cast<int>(bombs.size());
    std::vector<bool> exploding(nbombs2, false);
    for (int b = 0; b < nbombs2; ++b) {
        if (bombs[b].exploded() ||
            flame_set.contains({bombs[b].position.row, bombs[b].position.col}))
            exploding[b] = true;
    }

    std::vector<std::vector<Position>> blasts(nbombs2);
    std::vector<bool> blast_done(nbombs2, false);
    bool chain = true;
    while (chain) {
        chain = false;
        for (int i = 0; i < nbombs2; ++i) {
            if (!exploding[i] || blast_done[i]) continue;
            blasts[i]     = compute_blast(bombs[i], static_board);
            blast_done[i] = true;
            const auto& blast = blasts[i];
            for (int j = 0; j < nbombs2; ++j) {
                if (exploding[j]) continue;
                for (auto& p : blast) {
                    if (p == bombs[j].position) {
                        exploding[j] = true;
                        chain = true;
                        break;
                    }
                }
            }
        }
    }

    // ── 12. Generate flames + return ammo + destroy wood ──────────────────────
    for (int b = 0; b < nbombs2; ++b) {
        if (!exploding[b]) continue;
        // Return ammo to bomber
        int bid = bombs[b].agent_id;
        if (bid >= 0 && bid < nagents) agents[bid].incr_ammo();

        for (auto& p : blasts[b]) {
            if (flame_set.insert({p.row, p.col}).second) {
                flames.push_back(Flame{p, FLAME_LIFE});
            } else {
                // Re-ignited cell: refresh the flame's lifetime
                for (auto& f : flames) if (f.position == p) f.life = FLAME_LIFE;
            }
        }
    }

    // Destroy wood hit by any blast (after all blasts were computed, so two
    // bombs hitting the same wood cannot pass through it in one tick)
    for (int b = 0; b < nbombs2; ++b) {
        if (!exploding[b]) continue;
        for (auto& p : blasts[b]) {
            if (static_board[p.row][p.col] != Item::Wood) continue;
            if (auto it = hidden_items.find(pos_key(p)); it != hidden_items.end()) {
                static_board[p.row][p.col] = it->second;
                hidden_items.erase(it);
            } else {
                static_board[p.row][p.col] = Item::Passage;
            }
        }
    }

    // Remove exploded bombs
    std::vector<Bomb> alive_bombs;
    alive_bombs.reserve(bombs.size());
    for (int b = 0; b < nbombs2; ++b) {
        if (!exploding[b]) alive_bombs.push_back(bombs[b]);
    }
    bombs = std::move(alive_bombs);

    // ── 13. Kill agents on flames ─────────────────────────────────────────────
    for (auto& agent : agents) {
        if (!agent.is_alive) continue;
        for (auto& f : flames) {
            if (agent.position == f.position) { agent.die(); break; }
        }
    }

    // ── 14. Rebuild board ─────────────────────────────────────────────────────
    rebuild_board(board, static_board, agents, bombs, flames);
}

// ── Termination / reward / info ───────────────────────────────────────────────

bool ForwardModel::is_done(std::span<const Agent> agents,
                           int step_count, GameType game_type) const noexcept {
    if (step_count >= MAX_STEPS) return true;

    int alive_count = 0;
    for (auto& a : agents) if (a.is_alive) ++alive_count;

    if (game_type == GameType::FFA) {
        return alive_count <= 1;
    }
    if (game_type == GameType::OneVsOne) {
        return alive_count <= 1;
    }
    // Team / TeamRadio: check if one whole team is eliminated
    bool t0 = agents[0].is_alive || agents[2].is_alive;
    bool t1 = agents[1].is_alive || agents[3].is_alive;
    return !t0 || !t1;
}

std::vector<double> ForwardModel::get_rewards(std::span<const Agent> agents,
                                               GameType game_type,
                                               bool done, int step_count) const {
    int n = static_cast<int>(agents.size());
    std::vector<double> rewards(n, 0.0);

    if (!done) return rewards;

    const bool team_game = game_type == GameType::Team || game_type == GameType::TeamRadio;

    // Timeout with the game still contested → soft tie for survivors, loss for the dead.
    if (step_count >= MAX_STEPS) {
        bool contested;
        if (team_game) {
            bool t0 = agents[0].is_alive || agents[2].is_alive;
            bool t1 = agents[1].is_alive || agents[3].is_alive;
            contested = t0 && t1;
        } else {
            int alive = 0;
            for (auto& a : agents) if (a.is_alive) ++alive;
            contested = alive > 1;
        }
        if (contested) {
            for (int i = 0; i < n; ++i) rewards[i] = agents[i].is_alive ? -0.5 : -1.0;
            return rewards;
        }
    }

    if (team_game) {
        // Team 0 = agents 0&2, team 1 = agents 1&3; the whole surviving team wins
        // (including a teammate who died along the way).
        bool t0 = agents[0].is_alive || agents[2].is_alive;
        bool t1 = agents[1].is_alive || agents[3].is_alive;
        for (int i = 0; i < n; ++i) {
            bool my_team_alive = (i % 2 == 0) ? t0 : t1;
            rewards[i] = (my_team_alive && !(t0 && t1)) ? 1.0 : -1.0;
        }
        return rewards;
    }

    // FFA / OneVsOne: the survivor (if any) wins; everyone else loses.
    for (int i = 0; i < n; ++i) rewards[i] = agents[i].is_alive ? 1.0 : -1.0;
    return rewards;
}

GameInfo ForwardModel::get_info(std::span<const Agent> agents,
                                GameType game_type, int step_count) const {
    GameInfo info;

    if (!is_done(agents, step_count, game_type)) {
        info.result = Result::Incomplete;
        return info;
    }

    auto rewards = get_rewards(agents, game_type, true, step_count);
    for (int i = 0; i < static_cast<int>(rewards.size()); ++i) {
        if (rewards[i] > 0.0) info.winners.push_back(agents[i].agent_id);
    }
    info.result = info.winners.empty() ? Result::Tie : Result::Win;
    return info;
}

} // namespace pommerman
