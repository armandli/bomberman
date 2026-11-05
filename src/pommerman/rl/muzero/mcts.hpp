#pragma once
#include <array>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "pommerman/rl/obs_encoder.hpp"      // NUM_ACTIONS
#include "pommerman/rl/muzero/muzero_config.hpp"
#include "pommerman/rl/muzero/muzero_model.hpp"

namespace pommerman::rl::muzero {

// ── MinMaxStats ───────────────────────────────────────────────────────────────
// Tracks min/max Q across the current search tree to normalise UCB scores.
struct MinMaxStats {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();

    void update(float val);
    float normalize(float val) const;
};

// ── MCTSNode ──────────────────────────────────────────────────────────────────
struct MCTSNode {
    // State
    torch::Tensor hidden;      // [1, D_MODEL] on the model's device
    float         reward = 0.f;
    bool          expanded = false;

    // Statistics
    int   visit_count = 0;
    float value_sum   = 0.f;
    float prior       = 0.f;   // P(action | parent)

    // Children indexed by action
    std::array<std::unique_ptr<MCTSNode>, NUM_ACTIONS> children;

    explicit MCTSNode(torch::Tensor h, float prior = 0.f, float reward = 0.f);

    float value() const;   // mean Q = value_sum / visit_count (0 if unvisited)
};

// ── run_mcts ──────────────────────────────────────────────────────────────────
// Runs `cfg.num_simulations` PUCT simulations from a root hidden state.
//
// Returns:
//   first  – visit-count policy [NUM_ACTIONS] (sum=1) as a CPU float tensor
//   second – root value (mean Q of root node)
std::pair<torch::Tensor, float>
run_mcts(const torch::Tensor& root_hidden,   // [1, D_MODEL]
         MuZeroModel&          model,
         const MuZeroConfig&   cfg,
         torch::Device         device);

} // namespace pommerman::rl::muzero
