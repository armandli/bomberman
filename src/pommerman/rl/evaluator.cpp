#include "pommerman/rl/evaluator.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <vector>
#include "pommerman/agents/random_agent.hpp"
#include "pommerman/agents/simple_agent.hpp"
#include "pommerman/environment.hpp"

namespace pommerman::rl {

EvalResult evaluate(const PolicyFn& policy, Opponent opponent, int num_games, uint32_t seed) {
    EvalResult res;
    Environment env{GameType::FFA, false, BOARD_SIZE, seed == 0 ? 1u : seed};

    std::vector<std::unique_ptr<BaseAgent>> opponents;
    for (int i = 0; i < NUM_AGENTS; ++i) {
        uint32_t s = seed * 4 + static_cast<uint32_t>(i) + 1;
        if (opponent == Opponent::Simple) opponents.push_back(std::make_unique<SimpleAgent>(s));
        else                              opponents.push_back(std::make_unique<RandomAgent>(s));
    }

    double survival_sum = 0.0, len_sum = 0.0, bombs_sum = 0.0;

    for (int g = 0; g < num_games; ++g) {
        const int seat = g % NUM_AGENTS;
        auto obs = env.reset();
        for (int i = 0; i < NUM_AGENTS; ++i) opponents[i]->episode_start(i);
        int  died_at = -1;
        int  bombs   = 0;
        bool done    = false;
        GameInfo info;

        while (!done) {
            std::array<Action, NUM_AGENTS> acts{};
            for (int i = 0; i < NUM_AGENTS; ++i) {
                if (i == seat) {
                    acts[i] = policy(obs[i]);
                    if (acts[i] == Action::Bomb && obs[i].ammo > 0) ++bombs;
                } else {
                    acts[i] = opponents[i]->act(obs[i]);
                }
            }
            auto [next_obs, rewards, d, step_info] = env.step(acts);
            obs  = std::move(next_obs);
            done = d;
            info = step_info;
            if (died_at < 0 && !env.agents()[seat].is_alive) died_at = env.step_count();
        }

        const int len = env.step_count();
        survival_sum += died_at < 0 ? len : died_at;
        len_sum      += len;
        bombs_sum    += bombs;

        ++res.games;
        // Tie only if the learner was still in the game at the end: alive at
        // timeout, or dying on the same step as the last opponents.
        bool won  = std::find(info.winners.begin(), info.winners.end(), seat) != info.winners.end();
        bool tied = info.result == Result::Tie && (died_at < 0 || died_at == len);
        if (won)       ++res.wins;
        else if (tied) ++res.ties;
        else           ++res.losses;
    }

    if (res.games > 0) {
        res.mean_survival = survival_sum / res.games;
        res.mean_game_len = len_sum      / res.games;
        res.mean_bombs    = bombs_sum    / res.games;
    }
    return res;
}

} // namespace pommerman::rl
