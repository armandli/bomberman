#include "utility.hpp"
#include <algorithm>
#include <set>

namespace pommerman {

Direction action_to_direction(Action action) noexcept {
    switch (action) {
        case Action::Up:    return Direction::Up;
        case Action::Down:  return Direction::Down;
        case Action::Left:  return Direction::Left;
        case Action::Right: return Direction::Right;
        default:            return Direction::Stop;
    }
}

Position get_next_position(Position pos, Direction dir) noexcept {
    switch (dir) {
        case Direction::Up:    return {pos.row - 1, pos.col};
        case Direction::Down:  return {pos.row + 1, pos.col};
        case Direction::Left:  return {pos.row,     pos.col - 1};
        case Direction::Right: return {pos.row,     pos.col + 1};
        default:               return pos;
    }
}

Direction get_direction(Position from, Position to) noexcept {
    int dr = to.row - from.row;
    int dc = to.col - from.col;
    if (dr == -1) return Direction::Up;
    if (dr ==  1) return Direction::Down;
    if (dc == -1) return Direction::Left;
    if (dc ==  1) return Direction::Right;
    return Direction::Stop;
}

bool position_on_board(Position pos, int board_size) noexcept {
    return pos.row >= 0 && pos.row < board_size &&
           pos.col >= 0 && pos.col < board_size;
}

bool position_is_passage(const Board& b, Position p) noexcept {
    return b[p.row][p.col] == Item::Passage;
}
bool position_is_rigid(const Board& b, Position p) noexcept {
    return b[p.row][p.col] == Item::Rigid;
}
bool position_is_wood(const Board& b, Position p) noexcept {
    return b[p.row][p.col] == Item::Wood;
}
bool position_is_bomb(const Board& b, Position p) noexcept {
    return b[p.row][p.col] == Item::Bomb;
}
bool position_is_flames(const Board& b, Position p) noexcept {
    return b[p.row][p.col] == Item::Flames;
}
bool position_is_agent(const Board& b, Position p) noexcept {
    int v = static_cast<int>(b[p.row][p.col]);
    return v >= static_cast<int>(Item::Agent0) && v <= static_cast<int>(Item::Agent3);
}
bool position_is_powerup(const Board& b, Position p) noexcept {
    Item it = b[p.row][p.col];
    return it == Item::ExtraBomb || it == Item::IncrRange || it == Item::Kick;
}
bool position_is_wall(const Board& b, Position p) noexcept {
    return position_is_rigid(b, p) || position_is_wood(b, p);
}
bool position_is_passable(const Board& b, Position p) noexcept {
    return !position_is_wall(b, p) && b[p.row][p.col] != Item::Bomb;
}

// ── Board generation ─────────────────────────────────────────────────────────

std::array<Position, NUM_AGENTS> agent_start_positions(int board_size) noexcept {
    int e = board_size - 2; // end index
    return {{ {1, 1}, {1, e}, {e, 1}, {e, e} }};
}

Board make_board(std::mt19937& rng, int board_size, int num_wood) {
    Board board;
    for (auto& row : board) row.fill(Item::Passage);

    // Fixed rigid pillars: even×even positions (classic Bomberman pattern)
    for (int r = 0; r < board_size; r += 2) {
        for (int c = 0; c < board_size; c += 2) {
            board[r][c] = Item::Rigid;
        }
    }

    // Build safe-zone set: agent positions + 2 steps along each axis
    auto starts = agent_start_positions(board_size);
    std::set<std::pair<int,int>> safe;
    constexpr std::array<std::pair<int,int>, 4> dirs{{{0,1},{0,-1},{1,0},{-1,0}}};
    for (auto [ar, ac] : starts) {
        for (int dist = 0; dist <= 2; ++dist) {
            for (auto [dr, dc] : dirs) {
                int r = ar + dr * dist, c = ac + dc * dist;
                if (r >= 0 && r < board_size && c >= 0 && c < board_size)
                    safe.insert({r, c});
            }
        }
        safe.insert({ar, ac});
    }

    // Collect candidate positions for symmetric wood placement (top-left quadrant only)
    int half = board_size / 2;
    std::vector<Position> q_cands;
    for (int r = 1; r <= half; ++r) {
        for (int c = 1; c <= half; ++c) {
            if (board[r][c] != Item::Passage) continue;
            // Check all 4 symmetric counterparts
            int mr = board_size - 1 - r, mc = board_size - 1 - c;
            if (board[r][mc] != Item::Passage || board[mr][c] != Item::Passage ||
                board[mr][mc] != Item::Passage) continue;
            if (safe.contains({r,c}) || safe.contains({r,mc}) ||
                safe.contains({mr,c}) || safe.contains({mr,mc})) continue;
            q_cands.push_back({r, c});
        }
    }

    std::shuffle(q_cands.begin(), q_cands.end(), rng);

    int wood_per_quadrant = num_wood / 4;
    int placed = 0;
    for (auto [r, c] : q_cands) {
        if (placed >= wood_per_quadrant) break;
        int mr = board_size - 1 - r, mc = board_size - 1 - c;
        board[r][c]   = Item::Wood;
        board[r][mc]  = Item::Wood;
        board[mr][c]  = Item::Wood;
        board[mr][mc] = Item::Wood;
        ++placed;
    }

    return board;
}

std::map<int, Item> make_items(const Board& board, std::mt19937& rng, int num_items) {
    // Collect all wood positions
    std::vector<int> wood_keys;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (board[r][c] == Item::Wood)
                wood_keys.push_back(r * BOARD_SIZE + c);
        }
    }

    std::shuffle(wood_keys.begin(), wood_keys.end(), rng);

    constexpr std::array<Item, 3> powerups{Item::ExtraBomb, Item::IncrRange, Item::Kick};
    std::uniform_int_distribution<int> pick(0, 2);

    std::map<int, Item> items;
    int count = std::min(num_items, (int)wood_keys.size());
    for (int i = 0; i < count; ++i) {
        items[wood_keys[i]] = powerups[pick(rng)];
    }
    return items;
}

void rebuild_board(Board& board,
                   const Board& static_board,
                   const std::vector<Agent>& agents,
                   const std::vector<Bomb>& bombs,
                   const std::vector<Flame>& flames) {
    board = static_board;

    for (auto& bomb : bombs)
        board[bomb.position.row][bomb.position.col] = Item::Bomb;

    // Flames override bombs
    for (auto& flame : flames)
        board[flame.position.row][flame.position.col] = Item::Flames;

    // Alive agents override flames (visually; death check already done)
    for (auto& agent : agents) {
        if (agent.is_alive) {
            board[agent.position.row][agent.position.col] =
                static_cast<Item>(static_cast<int>(Item::Agent0) + agent.agent_id);
        }
    }
}

} // namespace pommerman
