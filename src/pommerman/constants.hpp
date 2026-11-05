#pragma once

namespace pommerman {

constexpr int BOARD_SIZE           = 11;
constexpr int NUM_RIGID            = 36;
constexpr int NUM_WOOD             = 36;
constexpr int NUM_ITEMS            = 20;
constexpr int AGENT_VIEW_SIZE      = 4;
constexpr int DEFAULT_BLAST_STRENGTH = 2;
constexpr int DEFAULT_BOMB_LIFE    = 9;
constexpr int FLAME_LIFE           = 2;
constexpr int FIRST_COLLAPSE       = 500;
constexpr int MAX_STEPS            = 800;
constexpr int MAX_AMMO             = 10;
constexpr int NUM_AGENTS           = 4;

enum class Item : int {
    Passage   = 0,
    Rigid     = 1,
    Wood      = 2,
    Bomb      = 3,
    Flames    = 4,
    Fog       = 5,
    ExtraBomb = 6,
    IncrRange = 7,
    Kick      = 8,
    AgentDummy = 9,
    Agent0    = 10,
    Agent1    = 11,
    Agent2    = 12,
    Agent3    = 13,
};

enum class GameType : int {
    FFA       = 1,
    Team      = 2,
    TeamRadio = 3,
    OneVsOne  = 4,
};

enum class Action : int {
    Stop  = 0,
    Up    = 1,
    Down  = 2,
    Left  = 3,
    Right = 4,
    Bomb  = 5,
};

enum class Direction : int {
    Stop  = 0,
    Up    = 1,
    Down  = 2,
    Left  = 3,
    Right = 4,
};

enum class Result : int {
    Win        = 0,
    Loss       = 1,
    Tie        = 2,
    Incomplete = 3,
};

} // namespace pommerman
