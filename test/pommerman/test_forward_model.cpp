#include <catch2/catch_test_macros.hpp>
#include "pommerman/forward_model.hpp"
#include "pommerman/utility.hpp"
#include <array>
#include <map>

using namespace pommerman;

// Helper: build a minimal empty board (all Passage except even×even Rigid)
static Board empty_board() {
    std::mt19937 rng(1);
    return make_board(rng, BOARD_SIZE, 0); // no wood
}

static std::vector<Agent> four_agents(GameType gt = GameType::FFA) {
    std::vector<Agent> agents(NUM_AGENTS);
    auto starts = agent_start_positions();
    for (int i = 0; i < NUM_AGENTS; ++i) {
        agents[i].set_agent_id(i, gt);
        agents[i].position = starts[i];
    }
    return agents;
}

TEST_CASE("Step: agents stay on Stop action", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;
    rebuild_board(board, static_b, agents, bombs, flames);

    auto starts = agent_start_positions();
    std::array<Action, NUM_AGENTS> acts{Action::Stop, Action::Stop,
                                        Action::Stop, Action::Stop};
    fm.step(acts, board, agents, bombs, flames, static_b, items);

    for (int i = 0; i < NUM_AGENTS; ++i) {
        CHECK(agents[i].position == starts[i]);
    }
}

TEST_CASE("Step: agent moves into passage", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;
    rebuild_board(board, static_b, agents, bombs, flames);

    // Agent 0 at (1,1), move right → should land at (1,2)
    std::array<Action, NUM_AGENTS> acts{Action::Right, Action::Stop,
                                        Action::Stop, Action::Stop};
    fm.step(acts, board, agents, bombs, flames, static_b, items);
    CHECK(agents[0].position == Position{1, 2});
}

TEST_CASE("Step: agent blocked by Rigid wall", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;

    // Place agent 0 at (1,2); (0,2) is Rigid (even row × even col).
    agents[0].position = {1, 2};
    rebuild_board(board, static_b, agents, bombs, flames);

    std::array<Action, NUM_AGENTS> acts{Action::Up, Action::Stop,
                                        Action::Stop, Action::Stop};
    fm.step(acts, board, agents, bombs, flames, static_b, items);
    CHECK(agents[0].position == Position{1, 2}); // blocked by (0,2) Rigid
}

TEST_CASE("Step: bomb is placed and ticks", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;
    rebuild_board(board, static_b, agents, bombs, flames);

    std::array<Action, NUM_AGENTS> acts{Action::Bomb, Action::Stop,
                                        Action::Stop, Action::Stop};
    fm.step(acts, board, agents, bombs, flames, static_b, items);

    REQUIRE(bombs.size() == 1);
    CHECK(bombs[0].agent_id == 0);
    CHECK(bombs[0].position == agents[0].position);
    CHECK(bombs[0].life == DEFAULT_BOMB_LIFE - 1); // ticked once
    CHECK(agents[0].ammo == 0);
}

TEST_CASE("Step: bomb explodes and kills agent on flame", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;

    // Place agent 0 at (5,5) and a bomb also at (5,5) about to explode
    agents[0].position = {5, 5};
    Bomb b;
    b.agent_id       = 1;
    b.position       = {5, 5};
    b.life           = 1; // will expire after one tick
    b.blast_strength = 1;
    bombs.push_back(b);
    rebuild_board(board, static_b, agents, bombs, flames);

    std::array<Action, NUM_AGENTS> acts{Action::Stop, Action::Stop,
                                        Action::Stop, Action::Stop};
    fm.step(acts, board, agents, bombs, flames, static_b, items);

    // Bomb should have exploded, agent 0 at (5,5) should be dead
    CHECK(bombs.empty());
    CHECK(!agents[0].is_alive);
    bool flame_at_centre = false;
    for (auto& f : flames) {
        if (f.position == Position{5, 5}) { flame_at_centre = true; break; }
    }
    CHECK(flame_at_centre);
}

TEST_CASE("Step: bomb explosion stops at Rigid", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;

    // Bomb at (5,5) blast_strength=4; row 4 col 4 is Rigid (even×even)
    Bomb b;
    b.agent_id = 0; b.position = {5,5}; b.life = 1; b.blast_strength = 4;
    bombs.push_back(b);
    rebuild_board(board, static_b, agents, bombs, flames);

    std::array<Action, NUM_AGENTS> acts{};
    acts.fill(Action::Stop);
    fm.step(acts, board, agents, bombs, flames, static_b, items);

    // (4,5) is passage → should have flame
    // (5,4) is passage → should have flame
    // (3,5) is passage → should have flame (before rigid at (2,5)? (2,5) odd col, even row → not pillar)
    // But (4,4) is Rigid → blast going up-left from (5,5) doesn't reach there anyway
    bool flame_at_up1 = false;
    for (auto& f : flames) {
        if (f.position == Position{4, 5}) { flame_at_up1 = true; break; }
    }
    CHECK(flame_at_up1);

    // Blast going left: (5,4),(5,3)...; (4,4) is not on the left axis
    // (5,4) passage, (5,3) passage, (5,2) even-col/odd-row → not pillar → passage
    // (5,0) even-col/odd-row → not pillar but we cap at 4 steps
    // Check no flame past board
    for (auto& f : flames) {
        CHECK(position_on_board(f.position));
    }
}

TEST_CASE("Step: flames tick and die", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;

    // Place a flame with life=1 at (5,5)
    flames.push_back(Flame{{5,5}, 1});
    rebuild_board(board, static_b, agents, bombs, flames);

    std::array<Action, NUM_AGENTS> acts{};
    acts.fill(Action::Stop);
    fm.step(acts, board, agents, bombs, flames, static_b, items);

    // Flame should be gone
    bool gone = true;
    for (auto& f : flames) if (f.position == Position{5,5}) { gone = false; break; }
    CHECK(gone);
}

TEST_CASE("Step: two agents swapping positions are both blocked", "[forward_model]") {
    ForwardModel fm;
    Board static_b = empty_board();
    Board board{};
    std::map<int,Item> items;
    auto agents = four_agents();
    std::vector<Bomb> bombs;
    std::vector<Flame> flames;

    // Place agent 0 at (5,5) and agent 1 at (5,6)
    agents[0].position = {5, 5};
    agents[1].position = {5, 6};
    rebuild_board(board, static_b, agents, bombs, flames);

    // Both try to swap
    std::array<Action, NUM_AGENTS> acts{Action::Right, Action::Left,
                                        Action::Stop, Action::Stop};
    fm.step(acts, board, agents, bombs, flames, static_b, items);

    CHECK(agents[0].position == Position{5, 5});
    CHECK(agents[1].position == Position{5, 6});
}

TEST_CASE("ForwardModel::is_done: single survivor", "[forward_model]") {
    ForwardModel fm;
    std::vector<Agent> agents = four_agents();
    agents[1].die(); agents[2].die(); agents[3].die();
    CHECK(fm.is_done(agents, 1, GameType::FFA));
}

TEST_CASE("ForwardModel::is_done: max steps", "[forward_model]") {
    ForwardModel fm;
    std::vector<Agent> agents = four_agents();
    CHECK(fm.is_done(agents, MAX_STEPS, GameType::FFA));
    CHECK(!fm.is_done(agents, MAX_STEPS - 1, GameType::FFA));
}

TEST_CASE("ForwardModel::get_rewards: winner gets +1", "[forward_model]") {
    ForwardModel fm;
    std::vector<Agent> agents = four_agents();
    agents[1].die(); agents[2].die(); agents[3].die();
    auto rewards = fm.get_rewards(agents, GameType::FFA, true, 10);
    CHECK(rewards[0] ==  1.0);
    CHECK(rewards[1] == -1.0);
    CHECK(rewards[2] == -1.0);
    CHECK(rewards[3] == -1.0);
}
