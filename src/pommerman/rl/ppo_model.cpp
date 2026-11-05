#include "pommerman/rl/ppo_model.hpp"

namespace pommerman::rl {

PPOActorCriticImpl::PPOActorCriticImpl() {
    encoder     = register_module("encoder",      ObsEncoder());
    policy_head = register_module("policy_head",
                      torch::nn::Linear(D_MODEL, NUM_ACTIONS));
    value_head  = register_module("value_head",
                      torch::nn::Linear(D_MODEL, 1));

    // Orthogonal initialisation: small scale for policy, unit for value
    torch::nn::init::orthogonal_(policy_head->weight, 0.01);
    torch::nn::init::constant_(policy_head->bias,     0.0);
    torch::nn::init::orthogonal_(value_head->weight,  1.0);
    torch::nn::init::constant_(value_head->bias,      0.0);
}

PPOActorCriticImpl::Output PPOActorCriticImpl::forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {
    auto features = encoder->forward(item_idx, sp_cont, ag_feat);
    return {
        policy_head(features),
        value_head(features).squeeze(-1)
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PPOActorCriticImpl::act(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {
    auto out          = forward(item_idx, sp_cont, ag_feat);
    auto log_probs_all = torch::log_softmax(out.logits, -1);
    auto probs         = torch::softmax(out.logits, -1);

    auto actions   = torch::multinomial(probs, 1).squeeze(-1);              // [B]
    auto log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1); // [B]
    auto entropy   = -(probs * log_probs_all).sum(-1);                      // [B]

    return {actions, log_probs, out.value, entropy};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PPOActorCriticImpl::evaluate_actions(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat,
        const torch::Tensor& actions) {
    auto out          = forward(item_idx, sp_cont, ag_feat);
    auto log_probs_all = torch::log_softmax(out.logits, -1);
    auto probs         = torch::softmax(out.logits, -1);

    auto log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1);
    auto entropy   = -(probs * log_probs_all).sum(-1);

    return {log_probs, out.value, entropy};
}

} // namespace pommerman::rl
