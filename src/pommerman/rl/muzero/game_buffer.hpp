#pragma once
#include <deque>
#include <vector>
#include <torch/torch.h>
#include "pommerman/rl/obs_encoder.hpp"
#include "pommerman/rl/muzero/muzero_config.hpp"

namespace pommerman::rl::muzero {

// ── GameSegment ───────────────────────────────────────────────────────────────
// Stores one agent's episode trajectory (observations, actions, rewards, MCTS targets).
struct GameSegment {
    // Observations stored as raw tensors (CPU)
    std::vector<torch::Tensor> item_idxs;   // each [121] int64
    std::vector<torch::Tensor> sp_conts;    // each [121, 8] float
    std::vector<torch::Tensor> ag_feats;    // each [13] float

    // Actions, rewards, MCTS policy targets, value targets
    std::vector<int>   actions;
    std::vector<float> rewards;
    std::vector<torch::Tensor> mcts_policies;  // each [NUM_ACTIONS] float
    std::vector<float> root_values;             // MCTS root value at each step
    std::vector<float> value_targets;           // n-step bootstrap (computed)

    int length() const { return static_cast<int>(actions.size()); }

    // Compute n-step value targets in-place (call once after episode ends):
    //   z_t = sum_{k<n} γ^k r_{t+k} + γ^n v_root(t+n)   (bootstrap only if t+n < T;
    //   the segment ends at the agent's death or game end, which are terminal)
    void compute_value_targets(float discount, int td_steps);
};

// ── TrainingBatch ─────────────────────────────────────────────────────────────
struct TrainingBatch {
    // Initial observation for representation network
    torch::Tensor item_idx;      // [B, 121] int64
    torch::Tensor sp_cont;       // [B, 121, 8]
    torch::Tensor ag_feat;       // [B, 13]

    // Unrolled K steps
    torch::Tensor actions;        // [B, K] int64
    torch::Tensor rewards;        // [B, K]
    torch::Tensor mcts_policies;  // [B, K, NUM_ACTIONS] (zeros past episode end)
    torch::Tensor value_targets;  // [B, K]              (zeros past episode end)

    // Observation reached after unroll step k (t+k+1), for the consistency loss
    torch::Tensor next_item_idx;  // [B, K, 121] int64
    torch::Tensor next_sp_cont;   // [B, K, 121, 8]
    torch::Tensor next_ag_feat;   // [B, K, 13]
    torch::Tensor next_mask;      // [B, K] 1 if obs t+k+1 exists
};

// ── GameBuffer ────────────────────────────────────────────────────────────────
class GameBuffer {
public:
    explicit GameBuffer(const MuZeroConfig& cfg);

    void add_game(GameSegment segment);

    int  size() const;
    bool ready(int min_games) const;

    // Sample a training batch (uniform over (game, position) pairs)
    TrainingBatch sample_batch(torch::Device device) const;

private:
    MuZeroConfig             cfg_;
    std::deque<GameSegment>  games_;
    int                      total_added_ = 0;
};

} // namespace pommerman::rl::muzero
