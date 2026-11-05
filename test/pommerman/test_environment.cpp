#include <catch2/catch_test_macros.hpp>
#include "pommerman/environment.hpp"
#include "pommerman/utility.hpp"

using namespace pommerman;

TEST_CASE("Environment reset returns 4 observations", "[environment]") {
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    auto obs = env.reset();
    REQUIRE(obs.size() == NUM_AGENTS);
    for (int i = 0; i < NUM_AGENTS; ++i) {
        CHECK(obs[i].agent_id == i);
        CHECK(obs[i].ammo == 1);
        CHECK(obs[i].blast_strength == DEFAULT_BLAST_STRENGTH);
        CHECK(!obs[i].can_kick);
    }
}

TEST_CASE("Environment reset places agents at corners", "[environment]") {
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    env.reset();
    auto starts = agent_start_positions();
    for (int i = 0; i < NUM_AGENTS; ++i) {
        CHECK(env.agents()[i].position == starts[i]);
    }
}

TEST_CASE("Environment step_count increments", "[environment]") {
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    env.reset();
    CHECK(env.step_count() == 0);
    std::array<Action, NUM_AGENTS> acts{};
    acts.fill(Action::Stop);
    env.step(acts);
    CHECK(env.step_count() == 1);
}

TEST_CASE("Environment step returns correct number of rewards", "[environment]") {
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    env.reset();
    std::array<Action, NUM_AGENTS> acts{};
    acts.fill(Action::Stop);
    auto [obs, rewards, done, info] = env.step(acts);
    CHECK(rewards.size() == NUM_AGENTS);
    CHECK(obs.size() == NUM_AGENTS);
}

TEST_CASE("Environment not done after one step (all Stop)", "[environment]") {
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    env.reset();
    std::array<Action, NUM_AGENTS> acts{};
    acts.fill(Action::Stop);
    auto [obs, rewards, done, info] = env.step(acts);
    CHECK(!done);
    CHECK(info.result == Result::Incomplete);
}

TEST_CASE("Environment partial observability uses Fog", "[environment]") {
    Environment env(GameType::FFA, true, BOARD_SIZE, 42);
    auto obs = env.reset();
    // Agent 0 at (1,1); cells far away should be Fog
    bool has_fog = false;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (obs[0].board[r][c] == Item::Fog) { has_fog = true; break; }
        }
        if (has_fog) break;
    }
    CHECK(has_fog);
}

TEST_CASE("Environment bomb observation fields populated", "[environment]") {
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    env.reset();

    // Force agent 0 to lay a bomb
    std::array<Action, NUM_AGENTS> acts{Action::Bomb, Action::Stop,
                                        Action::Stop, Action::Stop};
    auto [obs, rewards, done, info] = env.step(acts);

    auto pos = env.agents()[0].position;
    CHECK(obs[0].bomb_blast_strength[pos.row][pos.col] > 0);
    CHECK(obs[0].bomb_life[pos.row][pos.col] > 0);
}

TEST_CASE("Environment game ends when only one agent alive", "[environment]") {
    // Run many steps; with all Stop actions nobody dies from bombs, so
    // game should reach MAX_STEPS and return done.
    Environment env(GameType::FFA, false, BOARD_SIZE, 42);
    env.reset();
    std::array<Action, NUM_AGENTS> acts{};
    acts.fill(Action::Stop);
    bool done = false;
    for (int s = 0; s < MAX_STEPS && !done; ++s) {
        auto [obs, rewards, d, info] = env.step(acts);
        done = d;
    }
    CHECK(done);
}
