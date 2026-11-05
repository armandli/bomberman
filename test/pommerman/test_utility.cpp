#include <catch2/catch_test_macros.hpp>
#include "pommerman/utility.hpp"

using namespace pommerman;

TEST_CASE("action_to_direction", "[utility]") {
    CHECK(action_to_direction(Action::Stop)  == Direction::Stop);
    CHECK(action_to_direction(Action::Up)    == Direction::Up);
    CHECK(action_to_direction(Action::Down)  == Direction::Down);
    CHECK(action_to_direction(Action::Left)  == Direction::Left);
    CHECK(action_to_direction(Action::Right) == Direction::Right);
    CHECK(action_to_direction(Action::Bomb)  == Direction::Stop);
}

TEST_CASE("get_next_position", "[utility]") {
    Position p{5, 5};
    CHECK(get_next_position(p, Direction::Up)    == Position{4, 5});
    CHECK(get_next_position(p, Direction::Down)  == Position{6, 5});
    CHECK(get_next_position(p, Direction::Left)  == Position{5, 4});
    CHECK(get_next_position(p, Direction::Right) == Position{5, 6});
    CHECK(get_next_position(p, Direction::Stop)  == p);
}

TEST_CASE("get_direction", "[utility]") {
    CHECK(get_direction({5,5}, {4,5}) == Direction::Up);
    CHECK(get_direction({5,5}, {6,5}) == Direction::Down);
    CHECK(get_direction({5,5}, {5,4}) == Direction::Left);
    CHECK(get_direction({5,5}, {5,6}) == Direction::Right);
    CHECK(get_direction({5,5}, {5,5}) == Direction::Stop);
}

TEST_CASE("position_on_board", "[utility]") {
    CHECK(position_on_board({0, 0}));
    CHECK(position_on_board({10, 10}));
    CHECK(!position_on_board({-1, 0}));
    CHECK(!position_on_board({0, 11}));
    CHECK(!position_on_board({11, 11}));
}

TEST_CASE("make_board: rigid pillars at even×even positions", "[utility]") {
    std::mt19937 rng(42);
    Board b = make_board(rng);

    // Every even×even position should be Rigid
    for (int r = 0; r < BOARD_SIZE; r += 2) {
        for (int c = 0; c < BOARD_SIZE; c += 2) {
            INFO("r=" << r << " c=" << c);
            CHECK(b[r][c] == Item::Rigid);
        }
    }
}

TEST_CASE("make_board: agent start positions are Passage", "[utility]") {
    std::mt19937 rng(42);
    Board b = make_board(rng);
    for (auto [r, c] : agent_start_positions()) {
        INFO("r=" << r << " c=" << c);
        CHECK(b[r][c] == Item::Passage);
    }
}

TEST_CASE("make_board: safe zones around agents are Passage", "[utility]") {
    std::mt19937 rng(42);
    Board b = make_board(rng);
    // First two steps orthogonally from each agent must be clear of Wood
    auto starts = agent_start_positions();
    constexpr std::array<std::pair<int,int>, 4> dirs{{{0,1},{0,-1},{1,0},{-1,0}}};
    for (auto [ar, ac] : starts) {
        for (auto [dr, dc] : dirs) {
            for (int dist = 1; dist <= 2; ++dist) {
                int r = ar + dr*dist, c = ac + dc*dist;
                if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                    INFO("r=" << r << " c=" << c);
                    CHECK(b[r][c] != Item::Wood);
                }
            }
        }
    }
}

TEST_CASE("make_items: items placed only under Wood", "[utility]") {
    std::mt19937 rng(42);
    Board b = make_board(rng);
    auto items = make_items(b, rng);
    for (auto& [key, item] : items) {
        int r = key / BOARD_SIZE, c = key % BOARD_SIZE;
        INFO("r=" << r << " c=" << c);
        CHECK(b[r][c] == Item::Wood);
        bool is_powerup = (item == Item::ExtraBomb ||
                           item == Item::IncrRange  ||
                           item == Item::Kick);
        CHECK(is_powerup);
    }
}

TEST_CASE("rebuild_board overlays objects correctly", "[utility]") {
    std::mt19937 rng(42);
    Board static_b = make_board(rng);
    Board board{};
    std::vector<Agent> agents(1);
    agents[0].agent_id = 0;
    agents[0].position = {1, 1};
    agents[0].is_alive = true;

    std::vector<Bomb>  bombs;
    std::vector<Flame> flames;

    rebuild_board(board, static_b, agents, bombs, flames);
    CHECK(board[1][1] == Item::Agent0);

    // Add a bomb
    bombs.push_back(Bomb{0, {3, 3}, 5, 2, std::nullopt});
    rebuild_board(board, static_b, agents, bombs, flames);
    CHECK(board[3][3] == Item::Bomb);

    // Add a flame (overrides bomb visually)
    flames.push_back(Flame{{3, 3}, 2});
    rebuild_board(board, static_b, agents, bombs, flames);
    CHECK(board[3][3] == Item::Flames);
}
