#pragma once
#include <torch/torch.h>
#include "pommerman/rl/dqn_model.hpp"
#include "pommerman/rl/replay_buffer.hpp"
#include "pommerman/rl/reward_shaper.hpp"
#include "pommerman/environment.hpp"
#include <memory>
#include <vector>

namespace pommerman::rl {

struct DQNConfig {
    int    replay_capacity    = 100'000;
    int    batch_size         = 64;
    int    warmup_steps       = 10'000;
    int    target_update_freq = 1'000;
    float  eps_start          = 1.0f;
    float  eps_end            = 0.1f;
    int    eps_decay_steps    = 1'000'000;
    float  gamma              = 0.99f;
    float  max_grad_norm      = 10.0f;
    double lr                 = 1e-4;
};

class DQNTrainer {
public:
    DQNTrainer(DuelingDQN online_net, DuelingDQN target_net,
               DQNConfig cfg, torch::Device device);

    struct Stats {
        float loss;         // NaN while warming up
        float ep_reward;    // cumulative episode reward for agent 0
        float epsilon;
        int   buffer_size;
    };

    // Execute one environment step (and optionally one gradient update).
    Stats step(Environment& env);

    int global_step() const { return global_step_; }

    DuelingDQN& online_net() { return online_net_; }
    DuelingDQN& target_net() { return target_net_; }

private:
    float current_epsilon() const;
    void  sync_target();

    DuelingDQN                           online_net_;
    DuelingDQN                           target_net_;
    DQNConfig                            cfg_;
    torch::Device                        device_;
    std::unique_ptr<torch::optim::Adam>  optimizer_;
    ReplayBuffer                         buffer_;

    int   global_step_ = 0;
    float ep_reward_   = 0.0f;

    std::vector<Observation> curr_obs_;
    bool                     curr_done_ = true;
};

} // namespace pommerman::rl
