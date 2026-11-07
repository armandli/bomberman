#pragma once
#include <cstdint>
#include <functional>
#include "pommerman/types.hpp"

namespace pommerman::rl {

// Maps one agent's observation to an action (greedy policy of a trained model).
using PolicyFn = std::function<Action(const Observation&)>;

enum class Opponent { Random, Simple };

struct EvalResult {
    int    games          = 0;
    int    wins           = 0;
    int    ties           = 0;
    int    losses         = 0;
    double mean_survival  = 0.0;   // steps the learner stayed alive
    double mean_game_len  = 0.0;
    double mean_bombs     = 0.0;   // bombs laid by the learner per game

    double win_rate()  const { return games ? static_cast<double>(wins)   / games : 0.0; }
    double tie_rate()  const { return games ? static_cast<double>(ties)   / games : 0.0; }
    double loss_rate() const { return games ? static_cast<double>(losses) / games : 0.0; }
};

// Plays `num_games` FFA games of the learner against 3 opponents of the given
// kind on a fresh environment (the training environment is not touched). The
// learner's seat rotates over 0..3 so no start corner is favoured.
// Deterministic for a given seed.
EvalResult evaluate(const PolicyFn& policy, Opponent opponent, int num_games, uint32_t seed);

} // namespace pommerman::rl

#include <iostream>
#include "pommerman/rl/metrics_logger.hpp"

namespace pommerman::rl {

// Standard evaluation used by all trainers: `games` vs SimpleAgents (the main
// benchmark) plus `games` vs RandomAgents (sanity check). Logs one "eval"
// record. `score` blends outcome and survival vs SimpleAgents so it moves
// before the first wins appear:
//   score = win_rate + 0.5 * tie_rate + 0.25 * mean_survival / MAX_STEPS
inline void run_and_log_eval(MetricsLogger& log, const PolicyFn& policy,
                             int games, uint32_t seed, int iter) {
    if (games <= 0) return;
    auto s = evaluate(policy, Opponent::Simple, games, seed);
    auto r = evaluate(policy, Opponent::Random, games, seed + 7919u);
    double score = s.win_rate() + 0.5 * s.tie_rate() + 0.25 * s.mean_survival / MAX_STEPS;
    std::cout << "Eval @" << iter << "  vs simple: win=" << s.win_rate() << " tie=" << s.tie_rate()
              << " loss=" << s.loss_rate() << " survive=" << s.mean_survival
              << "  | vs random: win=" << r.win_rate() << "  score=" << score << "\n";
    log.log("eval", {{"iter", iter}, {"games", s.games}, {"score", score},
                     {"win_rate", s.win_rate()}, {"tie_rate", s.tie_rate()},
                     {"loss_rate", s.loss_rate()}, {"mean_survival", s.mean_survival},
                     {"mean_game_len", s.mean_game_len}, {"mean_bombs", s.mean_bombs},
                     {"rand_win_rate", r.win_rate()}, {"rand_survival", r.mean_survival},
                     {"rand_bombs", r.mean_bombs}});
}

} // namespace pommerman::rl
