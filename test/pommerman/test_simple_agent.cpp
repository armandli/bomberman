#include <catch2/catch_test_macros.hpp>
#include <climits>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include "pommerman/agents/random_agent.hpp"
#include "pommerman/agents/simple_agent.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/utility.hpp"

using namespace pommerman;

namespace {

Observation open_obs(Position me) {
    std::mt19937 rng(1);
    Observation o;
    o.board    = make_board(rng, BOARD_SIZE, 0);  // rigid pillars only
    o.position = me;
    o.agent_id = 0;
    o.ammo     = 1;
    o.enemies  = {1, 2, 3};
    o.alive_agents = {0, 1, 2, 3};
    o.board[me.row][me.col] = Item::Agent0;
    return o;
}

struct Outcome { int wins = 0, ties = 0, losses = 0; double mean_len = 0; std::array<int, NUM_AGENTS> seat_wins{}; };

// Learner in seat 0 vs three opponents.
template <typename MakeLearner, typename MakeOpp>
Outcome play(int games, MakeLearner make_learner, MakeOpp make_opp) {
    Environment env{GameType::FFA, false, BOARD_SIZE, 7};
    Outcome out;
    for (int g = 0; g < games; ++g) {
        std::vector<std::unique_ptr<BaseAgent>> agents;
        agents.push_back(make_learner(g));
        for (int i = 1; i < NUM_AGENTS; ++i) agents.push_back(make_opp(g * 4 + i));
        auto obs = env.reset();
        for (int i = 0; i < NUM_AGENTS; ++i) agents[i]->episode_start(i);
        bool done = false;
        GameInfo info;
        while (!done) {
            std::vector<Action> acts;
            for (int i = 0; i < NUM_AGENTS; ++i) acts.push_back(agents[i]->act(obs[i]));
            auto [o, r, d, inf] = env.step(acts);
            obs = std::move(o); done = d; info = inf;
        }
        out.mean_len += env.step_count();
        if (info.result == Result::Win)
            for (int w : info.winners) ++out.seat_wins[w];
        if (info.result == Result::Win && info.winners == std::vector<int>{0}) ++out.wins;
        else if (info.result == Result::Tie && env.agents()[0].is_alive)      ++out.ties;
        else                                                                   ++out.losses;
    }
    out.mean_len /= games;
    return out;
}

std::string str(const Outcome& o) {
    return "W" + std::to_string(o.wins) + "/T" + std::to_string(o.ties) + "/L" +
           std::to_string(o.losses) + " len=" + std::to_string(static_cast<int>(o.mean_len)) +
           " seat_wins=" + std::to_string(o.seat_wins[0]) + "," + std::to_string(o.seat_wins[1]) +
           "," + std::to_string(o.seat_wins[2]) + "," + std::to_string(o.seat_wins[3]);
}

struct StopAgent final : BaseAgent {
    Action act(const Observation&) override { return Action::Stop; }
};

} // namespace

TEST_CASE("SimpleAgent danger map marks blast cells with bomb life", "[simple_agent]") {
    auto o = open_obs({5, 5});
    o.bomb_life[5][3] = 4;
    o.bomb_blast_strength[5][3] = 2;
    auto d = SimpleAgent::danger_map(o);
    CHECK(d[5][3] == 4);
    CHECK(d[5][5] == 4);          // two cells right of the bomb
    CHECK(d[5][6] == INT_MAX);    // out of range
    CHECK(d[4][3] == 4);
}

TEST_CASE("SimpleAgent steps out of a bomb's blast", "[simple_agent]") {
    auto o = open_obs({5, 5});
    o.bomb_life[5][5] = 3;        // bomb under the agent
    o.bomb_blast_strength[5][5] = 2;
    SimpleAgent agent{1};
    Action a = agent.act(o);
    CHECK(a != Action::Stop);
    CHECK(a != Action::Bomb);
}

TEST_CASE("SimpleAgent lays a bomb next to wood when it can escape", "[simple_agent]") {
    auto o = open_obs({5, 5});
    o.board[5][6] = Item::Wood;
    SimpleAgent agent{1};
    CHECK(agent.act(o) == Action::Bomb);
}

TEST_CASE("SimpleAgent is a stronger baseline than random play", "[simple_agent][benchmark]") {
    constexpr int G = 60;
    auto simple = [](int s) { return std::make_unique<SimpleAgent>(static_cast<uint32_t>(s + 1)); };
    auto random = [](int s) { return std::make_unique<RandomAgent>(static_cast<uint32_t>(s + 1)); };
    auto stop   = [](int)   { return std::make_unique<StopAgent>(); };

    auto simple_vs_random = play(G, simple, random);
    auto stop_vs_simple   = play(G, stop,   simple);
    auto random_vs_simple = play(G, random, simple);
    auto simple_vs_simple = play(G, simple, simple);
    WARN(G << " games, seat 0 vs 3 opponents:"
         << "\n  simple vs random: " << str(simple_vs_random)
         << "\n  stop   vs simple: " << str(stop_vs_simple)
         << "\n  random vs simple: " << str(random_vs_simple)
         << "\n  simple vs simple: " << str(simple_vs_simple));

    CHECK(simple_vs_random.wins >= G * 6 / 10);   // dominates random agents
    CHECK(stop_vs_simple.wins   <= G / 10);       // standing still no longer wins
    CHECK(random_vs_simple.wins <= G / 10);
}

TEST_CASE("Seat balance probe: four SimpleAgents (run with [.probe])", "[.probe]") {
    Environment env{GameType::FFA, false, BOARD_SIZE, 11};
    std::array<int, NUM_AGENTS> survived{}, wins{};
    int timeouts = 0, alldead = 0;
    constexpr int G = 200;
    for (int g = 0; g < G; ++g) {
        std::vector<std::unique_ptr<BaseAgent>> agents;
        for (int i = 0; i < NUM_AGENTS; ++i) agents.push_back(std::make_unique<SimpleAgent>(g * 4 + i + 1));
        auto obs = env.reset();
        bool done = false; GameInfo info;
        while (!done) {
            std::vector<Action> acts;
            for (int i = 0; i < NUM_AGENTS; ++i) acts.push_back(agents[i]->act(obs[i]));
            auto [o, r, d, inf] = env.step(acts); obs = std::move(o); done = d; info = inf;
        }
        for (int i = 0; i < NUM_AGENTS; ++i) survived[i] += env.agents()[i].is_alive;
        for (int w : info.winners) ++wins[w];
        if (env.step_count() >= MAX_STEPS) ++timeouts;
        bool any = false; for (auto& a : env.agents()) any |= a.is_alive;
        if (!any) ++alldead;
    }
    WARN("survived " << survived[0] << "," << survived[1] << "," << survived[2] << "," << survived[3]
         << " wins " << wins[0] << "," << wins[1] << "," << wins[2] << "," << wins[3]
         << " timeouts " << timeouts << " alldead " << alldead);
}
