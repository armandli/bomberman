#pragma once
#include <torch/torch.h>
#include "pommerman/rl/obs_encoder.hpp"
#include <tuple>

namespace pommerman::rl {

struct PPOActorCriticImpl : torch::nn::Module {
    ObsEncoder      encoder{nullptr};
    torch::nn::Linear policy_head{nullptr};
    torch::nn::Linear value_head{nullptr};

    PPOActorCriticImpl();

    struct Output {
        torch::Tensor logits; // [B, NUM_ACTIONS]
        torch::Tensor value;  // [B]
    };

    Output forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat
    );

    // Stochastic action sampling. Returns (actions, log_probs, values, entropy).
    // All tensors are [B].
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    act(const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat);

    // Evaluate given actions (used in PPO update).
    // Returns (log_probs, values, entropy), all [B].
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluate_actions(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat,
        const torch::Tensor& actions // [B] int64
    );
};
TORCH_MODULE(PPOActorCritic);

} // namespace pommerman::rl
