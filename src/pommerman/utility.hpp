#pragma once
#include <map>
#include <random>
#include "characters.hpp"

namespace pommerman {

// ── Direction / position helpers ─────────────────────────────────────────────

Direction action_to_direction(Action action) noexcept;
Position  get_next_position(Position pos, Direction dir) noexcept;
// Returns the direction from `from` to `to` (must be adjacent); Stop if same.
Direction get_direction(Position from, Position to) noexcept;

bool position_on_board(Position pos, int board_size = BOARD_SIZE) noexcept;

bool position_is_passage(const Board& b, Position p) noexcept;
bool position_is_rigid(const Board& b, Position p) noexcept;
bool position_is_wood(const Board& b, Position p) noexcept;
bool position_is_bomb(const Board& b, Position p) noexcept;
bool position_is_flames(const Board& b, Position p) noexcept;
bool position_is_agent(const Board& b, Position p) noexcept;
bool position_is_powerup(const Board& b, Position p) noexcept;
// Rigid or Wood
bool position_is_wall(const Board& b, Position p) noexcept;
// Passage, powerup, agent, or (optionally) bomb
bool position_is_passable(const Board& b, Position p) noexcept;

// ── Board generation ─────────────────────────────────────────────────────────

// Builds initial board: rigid pillars at even×even positions,
// random wood walls (symmetric about centre), agent safe-zones cleared.
Board make_board(std::mt19937& rng,
                 int board_size = BOARD_SIZE,
                 int num_wood   = NUM_WOOD);

// Hides powerup items under random Wood tiles; returns a map
// (row*BOARD_SIZE+col) → item that the forward model reveals on wood destruction.
std::map<int, Item> make_items(const Board& board, std::mt19937& rng,
                               int num_items = NUM_ITEMS);

// Rebuilds the dynamic board from static (rigid/wood/items) + game objects.
void rebuild_board(Board& board,
                   const Board& static_board,
                   const std::vector<Agent>& agents,
                   const std::vector<Bomb>& bombs,
                   const std::vector<Flame>& flames);

// Initial agent placements for a given board_size.
std::array<Position, NUM_AGENTS> agent_start_positions(int board_size = BOARD_SIZE) noexcept;

} // namespace pommerman
