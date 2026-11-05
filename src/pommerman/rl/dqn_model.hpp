#pragma once
#include <torch/torch.h>
#include "pommerman/rl/obs_encoder.hpp"

namespace pommerman::rl {

// Dueling Double DQN: Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))
struct DuelingDQNImpl : torch::nn::Module {
    ObsEncoder             encoder{nullptr};
    torch::nn::Sequential  advantage_stream{nullptr};
    torch::nn::Sequential  value_stream{nullptr};

    DuelingDQNImpl();

    // Returns Q-values [B, NUM_ACTIONS]
    torch::Tensor forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat
    );
};
TORCH_MODULE(DuelingDQN);

} // namespace pommerman::rl
