#include "pommerman/rl/muzero/game_buffer.hpp"
#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

namespace pommerman::rl::muzero {

// ── GameSegment ───────────────────────────────────────────────────────────────

void GameSegment::compute_value_targets(float discount, int td_steps) {
    int T = static_cast<int>(rewards.size());
    value_targets.assign(T, 0.f);

    for (int t = 0; t < T; ++t) {
        float G = 0.f;
        float gamma_k = 1.f;
        for (int k = 0; k < td_steps && t + k < T; ++k) {
            G += gamma_k * rewards[t + k];
            gamma_k *= discount;
        }
        int boot = t + td_steps;
        if (boot < T && boot < static_cast<int>(root_values.size()))
            G += gamma_k * root_values[boot];
        value_targets[t] = G;
    }
}

// ── GameBuffer ────────────────────────────────────────────────────────────────

GameBuffer::GameBuffer(const MuZeroConfig& cfg) : cfg_(cfg) {}

void GameBuffer::add_game(GameSegment segment) {
    games_.push_back(std::move(segment));
    ++total_added_;
    while (static_cast<int>(games_.size()) > cfg_.buffer_capacity)
        games_.pop_front();
}

int GameBuffer::size() const { return static_cast<int>(games_.size()); }

bool GameBuffer::ready(int min_games) const { return size() >= min_games; }

TrainingBatch GameBuffer::sample_batch(torch::Device device) const {
    static thread_local std::mt19937 rng{std::random_device{}()};

    int B = cfg_.batch_size;
    int K = cfg_.unroll_steps;

    // Collect all valid (game_idx, pos) pairs
    std::vector<std::pair<int,int>> positions;
    positions.reserve(games_.size() * 32);
    for (int g = 0; g < static_cast<int>(games_.size()); ++g) {
        int T = games_[g].length();
        for (int t = 0; t < T; ++t)
            positions.emplace_back(g, t);
    }

    if (static_cast<int>(positions.size()) < B)
        throw std::runtime_error("GameBuffer: not enough positions to sample");

    // Uniform sample without replacement
    std::vector<std::pair<int,int>> chosen(B);
    std::sample(positions.begin(), positions.end(), chosen.begin(), B, rng);

    // Allocate output tensors (CPU first, move to device at end)
    auto item_idx_t     = torch::zeros({B, 121},          torch::kLong);
    auto sp_cont_t      = torch::zeros({B, 121, 8},        torch::kFloat);
    auto ag_feat_t      = torch::zeros({B, 13},            torch::kFloat);
    auto actions_t      = torch::zeros({B, K},             torch::kLong);
    auto rewards_t      = torch::zeros({B, K},             torch::kFloat);
    auto mcts_pol_t     = torch::zeros({B, K, NUM_ACTIONS},torch::kFloat);
    auto value_tgt_t    = torch::zeros({B, K},             torch::kFloat);
    auto next_item_t    = torch::zeros({B, K, 121},        torch::kLong);
    auto next_sp_t      = torch::zeros({B, K, 121, 8},     torch::kFloat);
    auto next_ag_t      = torch::zeros({B, K, 13},         torch::kFloat);
    auto next_mask_t    = torch::zeros({B, K},             torch::kFloat);
    std::uniform_int_distribution<int> rand_action(0, NUM_ACTIONS - 1);

    for (int i = 0; i < B; ++i) {
        auto [g, t] = chosen[i];
        const GameSegment& seg = games_[g];
        int T = seg.length();

        item_idx_t[i]  = seg.item_idxs[t];
        sp_cont_t[i]   = seg.sp_conts[t];
        ag_feat_t[i]   = seg.ag_feats[t];

        for (int k = 0; k < K; ++k) {
            int idx = t + k;
            if (idx < T) {
                actions_t[i][k]   = seg.actions[idx];
                rewards_t[i][k]   = seg.rewards[idx];
                mcts_pol_t[i][k]  = seg.mcts_policies[idx];
                value_tgt_t[i][k] = seg.value_targets[idx];
            } else {
                // Past the terminal step: absorbing state with zero reward/value,
                // no policy target, arbitrary action.
                actions_t[i][k] = rand_action(rng);
            }
            if (idx + 1 < T) {
                next_item_t[i][k] = seg.item_idxs[idx + 1];
                next_sp_t[i][k]   = seg.sp_conts[idx + 1];
                next_ag_t[i][k]   = seg.ag_feats[idx + 1];
                next_mask_t[i][k] = 1.f;
            }
        }
    }

    return TrainingBatch{
        item_idx_t.to(device),
        sp_cont_t.to(device),
        ag_feat_t.to(device),
        actions_t.to(device),
        rewards_t.to(device),
        mcts_pol_t.to(device),
        value_tgt_t.to(device),
        next_item_t.to(device),
        next_sp_t.to(device),
        next_ag_t.to(device),
        next_mask_t.to(device)
    };
}

} // namespace pommerman::rl::muzero
