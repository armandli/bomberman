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
    int    train_freq         = 4;       // gradient step every N env steps
    float  eps_start          = 1.0f;    // epsilon is unused by a noisy model
    float  eps_end            = 0.1f;
    int    eps_decay_steps    = 1'000'000;
    float  gamma              = 0.99f;
    float  max_grad_norm      = 10.0f;
    double lr                 = 1e-4;

    int    n_step             = 3;
    bool   prioritized        = true;
    float  per_omega          = 0.5f;    // priority exponent
    float  per_beta_start     = 0.4f;    // importance-sampling exponent, annealed to 1
    int    total_steps        = 5'000'000;  // length of the beta schedule
};

class DQNTrainer {
public:
    // Noisy/distributional behaviour comes from the networks' DQNModelOptions.
    DQNTrainer(DuelingDQN online_net, DuelingDQN target_net,
               DQNConfig cfg, torch::Device device);

    struct Stats {
        float loss;         // NaN while warming up. KL for a distributional model
        float ep_reward;    // agent 0's total reward in the last completed episode
        float epsilon;
        int   buffer_size;
        int   episodes;     // games finished so far
        int   last_ep_len;  // length of the last finished game
        float q_mean;       // mean Q of the sampled actions (0 if no update ran)
        float edge_mass;    // target mass on the outermost atoms (0 if scalar or no update)
        float beta;         // current importance-sampling exponent
    };

    // Execute one environment step (and a gradient update every train_freq steps).
    // Stats::loss is NaN when no update ran this step.
    Stats step(Environment& env);

    int global_step() const { return global_step_; }
    // Resume support: continue the epsilon/beta schedules and target-sync cadence.
    void set_global_step(int s) { global_step_ = s; }
    torch::optim::Optimizer& optimizer() { return *optimizer_; }
    // Re-copy online → target (after loading a checkpoint into online_net).
    void sync_target_net() { sync_target(); }

    DuelingDQN& online_net() { return online_net_; }
    DuelingDQN& target_net() { return target_net_; }

private:
    struct UpdateResult {
        float loss;
        float q_mean;
        float edge_mass;
    };

    float current_epsilon() const;
    float current_beta() const;
    void  sync_target();
    UpdateResult learn();

    DuelingDQN                           online_net_;
    DuelingDQN                           target_net_;
    DQNConfig                            cfg_;
    torch::Device                        device_;
    std::unique_ptr<torch::optim::Adam>  optimizer_;
    ReplayBuffer                         buffer_;
    std::vector<NStepQueue>              nstep_;   // one per agent

    int   global_step_ = 0;
    float ep_reward_      = 0.0f;   // running total for the current episode
    float last_ep_reward_ = 0.0f;   // total of the last completed episode
    int   ep_len_         = 0;
    int   last_ep_len_    = 0;
    int   episodes_       = 0;

    std::vector<Observation> curr_obs_;
    bool                     curr_done_ = true;
};

} // namespace pommerman::rl
