#include "pommerman/rl/dqn_model.hpp"

namespace pommerman::rl {

DuelingDQNImpl::DuelingDQNImpl() {
    encoder = register_module("encoder", ObsEncoder());

    advantage_stream = register_module("advantage_stream",
        torch::nn::Sequential(
            torch::nn::Linear(D_MODEL, D_MODEL),
            torch::nn::ReLU(),
            torch::nn::Linear(D_MODEL, NUM_ACTIONS)
        ));

    value_stream = register_module("value_stream",
        torch::nn::Sequential(
            torch::nn::Linear(D_MODEL, D_MODEL),
            torch::nn::ReLU(),
            torch::nn::Linear(D_MODEL, 1)
        ));
}

torch::Tensor DuelingDQNImpl::forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {
    auto features  = encoder->forward(item_idx, sp_cont, ag_feat);
    auto advantage = advantage_stream->forward(features);              // [B, NUM_ACTIONS]
    auto value     = value_stream->forward(features);                  // [B, 1]

    // Dueling aggregation
    return value + advantage - advantage.mean(/*dim=*/-1, /*keepdim=*/true);
}

} // namespace pommerman::rl
