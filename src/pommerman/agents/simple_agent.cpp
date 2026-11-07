#include "simple_agent.hpp"
#include <algorithm>
#include <climits>
#include <deque>
#include <vector>
#include "../utility.hpp"

namespace pommerman {

namespace {

constexpr int NEVER = INT_MAX;
constexpr std::array<Direction, 4> DIRS{Direction::Up, Direction::Down,
                                        Direction::Left, Direction::Right};

bool is_agent_item(Item it) {
    int v = static_cast<int>(it);
    return v >= static_cast<int>(Item::Agent0) && v <= static_cast<int>(Item::Agent3);
}

bool is_powerup_item(Item it) {
    return it == Item::ExtraBomb || it == Item::IncrRange || it == Item::Kick;
}

// Cells an agent can walk into this tick (ignores danger).
bool walkable(const Observation& o, Position p) {
    if (!position_on_board(p)) return false;
    Item it = o.board[p.row][p.col];
    if (it == Item::Rigid || it == Item::Wood || it == Item::Bomb ||
        it == Item::Fog   || it == Item::Flames) return false;
    if (is_agent_item(it)) return false;
    return o.bomb_life[p.row][p.col] == 0;  // a bomb may be hidden under an agent
}

Action dir_to_action(Direction d) {
    switch (d) {
        case Direction::Up:    return Action::Up;
        case Direction::Down:  return Action::Down;
        case Direction::Left:  return Action::Left;
        case Direction::Right: return Action::Right;
        default:               return Action::Stop;
    }
}

struct Bfs {
    std::array<std::array<int, BOARD_SIZE>, BOARD_SIZE> dist;
    std::array<std::array<Direction, BOARD_SIZE>, BOARD_SIZE> first;  // first step from start
};

// BFS over walkable cells; a cell is entered only if it is still safe on
// arrival (arrival time < danger time). `dirs` order breaks ties.
Bfs bfs(const Observation& o, const SimpleAgent::DangerMap& danger,
        const std::array<Direction, 4>& dirs) {
    Bfs b;
    for (auto& row : b.dist) row.fill(-1);
    for (auto& row : b.first) row.fill(Direction::Stop);
    std::deque<Position> q;
    b.dist[o.position.row][o.position.col] = 0;
    q.push_back(o.position);
    while (!q.empty()) {
        Position p = q.front(); q.pop_front();
        int d = b.dist[p.row][p.col];
        for (Direction dir : dirs) {
            Position n = get_next_position(p, dir);
            if (!walkable(o, n) || b.dist[n.row][n.col] >= 0) continue;
            if (danger[n.row][n.col] <= d + 1) continue;  // would be standing in a blast
            b.dist[n.row][n.col]  = d + 1;
            b.first[n.row][n.col] = (d == 0) ? dir : b.first[p.row][p.col];
            q.push_back(n);
        }
    }
    return b;
}

// Nearest reachable cell satisfying `pred`; returns the first step towards it.
template <typename Pred>
std::optional<Direction> step_towards(const Bfs& b, Pred pred) {
    int best = INT_MAX;
    std::optional<Direction> step;
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int d = b.dist[r][c];
            if (d <= 0 || d >= best) continue;
            if (!pred(Position{r, c})) continue;
            best = d;
            step = b.first[r][c];
        }
    return step;
}

// Is there a cell that never gets hit, reachable in time?
bool can_escape(const Observation& o, const SimpleAgent::DangerMap& danger,
                const std::array<Direction, 4>& dirs) {
    if (danger[o.position.row][o.position.col] == NEVER) return true;
    auto b = bfs(o, danger, dirs);
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c)
            if (b.dist[r][c] > 0 && danger[r][c] == NEVER) return true;
    return false;
}

// Cells within `strength` of p along open lines (walls stop the scan; the wall
// cell itself is included).
std::vector<Position> blast_cells(const Observation& o, Position p, int strength) {
    std::vector<Position> cells{p};
    for (Direction dir : DIRS) {
        Position q = p;
        for (int s = 1; s <= strength; ++s) {
            q = get_next_position(q, dir);
            if (!position_on_board(q)) break;
            Item it = o.board[q.row][q.col];
            if (it == Item::Rigid || it == Item::Fog) break;
            cells.push_back(q);
            if (it == Item::Wood) break;
        }
    }
    return cells;
}

} // namespace

SimpleAgent::SimpleAgent(uint32_t seed)
    : _rng(seed == 0 ? std::random_device{}() : seed) {}

SimpleAgent::DangerMap SimpleAgent::danger_map(const Observation& o,
                                               std::optional<Position> extra_bomb,
                                               int extra_strength) {
    struct B { Position p; int life; int strength; };
    std::vector<B> bombs;
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c)
            if (o.bomb_life[r][c] > 0)
                bombs.push_back({{r, c}, o.bomb_life[r][c], o.bomb_blast_strength[r][c]});
    // A bomb laid now has already ticked once by the agent's next decision.
    if (extra_bomb) bombs.push_back({*extra_bomb, DEFAULT_BOMB_LIFE - 1, extra_strength});

    // Chain reactions: a bomb inside another's blast explodes no later than it.
    std::vector<std::vector<Position>> blasts;
    for (auto& b : bombs) blasts.push_back(blast_cells(o, b.p, b.strength));
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 0; i < bombs.size(); ++i)
            for (size_t j = 0; j < bombs.size(); ++j) {
                if (i == j || bombs[j].life <= bombs[i].life) continue;
                if (std::find(blasts[i].begin(), blasts[i].end(), bombs[j].p) != blasts[i].end()) {
                    bombs[j].life = bombs[i].life;
                    changed = true;
                }
            }
    }

    DangerMap d;
    for (auto& row : d) row.fill(NEVER);
    for (size_t i = 0; i < bombs.size(); ++i)
        for (auto& p : blasts[i])
            d[p.row][p.col] = std::min(d[p.row][p.col], bombs[i].life);
    for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c)
            if (o.flame_life[r][c] > 0) d[r][c] = 0;
    return d;
}

Action SimpleAgent::act(const Observation& o) {
    auto dirs = DIRS;
    std::shuffle(dirs.begin(), dirs.end(), _rng);

    const Position me = o.position;
    const auto danger = danger_map(o);

    // 1. Escape
    if (danger[me.row][me.col] != NEVER) {
        auto b = bfs(o, danger, dirs);
        if (auto s = step_towards(b, [&](Position p) { return danger[p.row][p.col] == NEVER; }))
            return dir_to_action(*s);
        // No fully safe cell: move to the cell that is hit latest
        int best_t = danger[me.row][me.col];
        Direction best = Direction::Stop;
        for (Direction dir : dirs) {
            Position n = get_next_position(me, dir);
            if (walkable(o, n) && danger[n.row][n.col] > best_t) {
                best_t = danger[n.row][n.col];
                best = dir;
            }
        }
        return dir_to_action(best);
    }

    auto is_enemy = [&](Item it) {
        if (!is_agent_item(it)) return false;
        int id = static_cast<int>(it) - static_cast<int>(Item::Agent0);
        return std::find(o.enemies.begin(), o.enemies.end(), id) != o.enemies.end();
    };

    // 2. Bomb if it would hit an enemy or wood and we can get away
    if (o.ammo > 0 && o.bomb_life[me.row][me.col] == 0) {
        bool worth = false;
        for (auto& p : blast_cells(o, me, o.blast_strength)) {
            Item it = o.board[p.row][p.col];
            if (is_enemy(it) || (it == Item::Wood &&
                                 std::abs(p.row - me.row) + std::abs(p.col - me.col) == 1))
                worth = true;
        }
        if (worth) {
            auto with_bomb = danger_map(o, me, o.blast_strength);
            if (can_escape(o, with_bomb, dirs)) return Action::Bomb;
        }
    }

    // 3. Move with purpose, never into a cell that is going to be hit
    SimpleAgent::DangerMap strict = danger;  // treat every threatened cell as blocked
    for (auto& row : strict)
        for (int& v : row)
            if (v != NEVER) v = 0;
    auto b = bfs(o, strict, dirs);

    if (auto s = step_towards(b, [&](Position p) { return is_powerup_item(o.board[p.row][p.col]); }))
        return dir_to_action(*s);

    if (o.ammo > 0) {
        auto next_to_wood = [&](Position p) {
            for (Direction dir : DIRS) {
                Position n = get_next_position(p, dir);
                if (position_on_board(n) && o.board[n.row][n.col] == Item::Wood) return true;
            }
            return false;
        };
        if (auto s = step_towards(b, next_to_wood)) return dir_to_action(*s);

        auto near_enemy = [&](Position p) {
            for (auto& q : blast_cells(o, p, o.blast_strength))
                if (is_enemy(o.board[q.row][q.col])) return true;
            return false;
        };
        if (auto s = step_towards(b, near_enemy)) return dir_to_action(*s);
    }

    return Action::Stop;
}

} // namespace pommerman
