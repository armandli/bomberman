#include "pommerman/rl/muzero/game_buffer.hpp"
#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

namespace pommerman::rl::muzero {

// ── GameSegment ───────────────────────────────────────────────────────────────

void GameSegment::compute_value_targets(float discount, int td_steps) {
    int T = static_cast<int>(rewards.size());
    value_targets.resize(T, 0.f);

    for (int t = 0; t < T; ++t) {
        // n-step discounted return
        float G = 0.f;
        float gamma_k = 1.f;
        for (int k = 0; k < td_steps && t + k < T; ++k) {
            G += gamma_k * rewards[t + k];
            gamma_k *= discount;
        }
        // Bootstrap from MCTS value at t+td_steps (use mcts value_targets computed
        // from the visit-count weighted value: mean Q of root node)
        // We store the root value alongside mcts_policies in value_targets during
        // self-play.  Here we just use the already-stored bootstrap value if available.
        // The bootstrap is added by the trainer when it pushes segments; at this call
        // the terminal tail has zero bootstrap by convention.
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

    for (int i = 0; i < B; ++i) {
        auto [g, t] = chosen[i];
        const GameSegment& seg = games_[g];
        int T = seg.length();

        item_idx_t[i]  = seg.item_idxs[t];
        sp_cont_t[i]   = seg.sp_conts[t];
        ag_feat_t[i]   = seg.ag_feats[t];

        for (int k = 0; k < K; ++k) {
            int idx = std::min(t + k, T - 1);
            actions_t[i][k]      = seg.actions[idx];
            rewards_t[i][k]      = seg.rewards[idx];
            mcts_pol_t[i][k]     = seg.mcts_policies[idx];
            value_tgt_t[i][k]    = seg.value_targets[idx];
        }
    }

    return TrainingBatch{
        item_idx_t.to(device),
        sp_cont_t.to(device),
        ag_feat_t.to(device),
        actions_t.to(device),
        rewards_t.to(device),
        mcts_pol_t.to(device),
        value_tgt_t.to(device)
    };
}

} // namespace pommerman::rl::muzero
