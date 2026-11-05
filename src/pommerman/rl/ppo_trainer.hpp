#pragma once
#include <torch/torch.h>
#include "pommerman/rl/ppo_model.hpp"
#include "pommerman/rl/reward_shaper.hpp"
#include "pommerman/environment.hpp"
#include <memory>
#include <vector>

namespace pommerman::rl {

struct PPOConfig {
    int    rollout_steps  = 128;   // env steps per agent per update
    int    ppo_epochs     = 10;
    int    minibatch_size = 64;
    float  clip_eps       = 0.2f;
    float  value_coef     = 0.5f;
    float  entropy_coef   = 0.01f;
    float  gamma          = 0.99f;
    float  gae_lambda     = 0.95f;
    float  max_grad_norm  = 0.5f;
    double lr             = 3e-4;
};

class PPOTrainer {
public:
    PPOTrainer(PPOActorCritic model, PPOConfig cfg, torch::Device device);

    struct Stats {
        float policy_loss;
        float value_loss;
        float entropy;
        float mean_reward;
        float explained_variance;
    };

    // Collect one rollout and run PPO updates. Returns training statistics.
    Stats update(Environment& env);

    PPOActorCritic& model() { return model_; }

private:
    // One time-step of data for all NUM_AGENTS agents simultaneously
    struct Step {
        torch::Tensor item_idx;   // [4, 121] on device
        torch::Tensor sp_cont;    // [4, 121, SPATIAL_CONT_DIM] on device
        torch::Tensor ag_feat;    // [4, AGENT_FEAT_DIM] on device
        torch::Tensor actions;    // [4] int64
        torch::Tensor log_probs;  // [4]
        torch::Tensor values;     // [4]
        torch::Tensor rewards;    // [4]
        torch::Tensor dones;      // [4]  broadcast scalar
    };

    std::vector<Step> collect_rollout(Environment& env);

    std::pair<torch::Tensor, torch::Tensor> compute_gae(
        const std::vector<Step>& steps,
        const torch::Tensor& last_values,  // [4]
        bool last_done
    );

    PPOActorCritic                        model_;
    PPOConfig                             cfg_;
    torch::Device                         device_;
    std::unique_ptr<torch::optim::Adam>   optimizer_;

    std::vector<Observation> curr_obs_;
    bool                     curr_done_ = true;
};

} // namespace pommerman::rl
