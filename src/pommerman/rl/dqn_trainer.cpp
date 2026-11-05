#include "pommerman/rl/dqn_trainer.hpp"
#include "pommerman/rl/obs_encoder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace pommerman::rl {

DQNTrainer::DQNTrainer(DuelingDQN online_net, DuelingDQN target_net,
                       DQNConfig cfg, torch::Device device)
    : online_net_(online_net), target_net_(target_net),
      cfg_(cfg), device_(device), buffer_(cfg.replay_capacity) {
    optimizer_ = std::make_unique<torch::optim::Adam>(
        online_net_->parameters(),
        torch::optim::AdamOptions(cfg_.lr));
    sync_target();
    target_net_->eval();
}

float DQNTrainer::current_epsilon() const {
    float progress = std::min(1.0f,
        static_cast<float>(global_step_) / static_cast<float>(cfg_.eps_decay_steps));
    return cfg_.eps_start + (cfg_.eps_end - cfg_.eps_start) * progress;
}

void DQNTrainer::sync_target() {
    torch::NoGradGuard ng;
    auto op = online_net_->parameters();
    auto tp = target_net_->parameters();
    for (size_t i = 0; i < op.size(); ++i)
        tp[i].copy_(op[i]);

    auto ob = online_net_->buffers();
    auto tb = target_net_->buffers();
    for (size_t i = 0; i < ob.size(); ++i)
        tb[i].copy_(ob[i]);
}

DQNTrainer::Stats DQNTrainer::step(Environment& env) {
    if (curr_done_) {
        curr_obs_  = env.reset();
        curr_done_ = false;
        ep_reward_ = 0.0f;
    }

    float eps = current_epsilon();

    // Encode current observations (keep CPU copy for buffer)
    auto [item_cpu, sp_cpu, ag_cpu] = obs_to_tensors(curr_obs_);

    std::array<Action, NUM_AGENTS> acts{};

    if (torch::rand({1}).item<float>() < eps) {
        for (int i = 0; i < NUM_AGENTS; ++i)
            acts[i] = static_cast<Action>(
                torch::randint(0, NUM_ACTIONS, {1}).item<int64_t>());
    } else {
        torch::NoGradGuard ng;
        online_net_->eval();
        auto q = online_net_->forward(
            item_cpu.to(device_), sp_cpu.to(device_), ag_cpu.to(device_));
        auto chosen = q.argmax(-1); // [4]
        for (int i = 0; i < NUM_AGENTS; ++i)
            acts[i] = static_cast<Action>(chosen[i].item<int64_t>());
    }

    auto [new_obs, base_rew, done, info] = env.step(acts);
    auto shaped = RewardShaper::shape(curr_obs_, new_obs, base_rew, done);
    ep_reward_ += shaped[0];

    // Encode next observations (CPU)
    auto [next_item, next_sp, next_ag] = obs_to_tensors(new_obs);

    // Store one transition per alive agent
    for (int i = 0; i < NUM_AGENTS; ++i) {
        bool was_alive = std::find(curr_obs_[i].alive_agents.begin(),
                                   curr_obs_[i].alive_agents.end(), i)
                         != curr_obs_[i].alive_agents.end();
        if (!was_alive) continue;

        buffer_.add(
            item_cpu[i], sp_cpu[i], ag_cpu[i],
            static_cast<int>(acts[i]),
            shaped[i],
            next_item[i], next_sp[i], next_ag[i],
            done);
    }

    curr_obs_  = std::move(new_obs);
    curr_done_ = done;
    if (done) {
        curr_obs_  = env.reset();
        curr_done_ = false;
    }

    ++global_step_;

    // Not enough data yet
    if (buffer_.size() < cfg_.warmup_steps)
        return {std::numeric_limits<float>::quiet_NaN(),
                ep_reward_, eps, buffer_.size()};

    // --- Gradient update ---
    online_net_->train();

    auto [b_item, b_sp, b_ag, b_acts, b_rew,
          b_next_item, b_next_sp, b_next_ag, b_done] =
        buffer_.sample(cfg_.batch_size, device_);

    torch::Tensor targets;
    {
        torch::NoGradGuard ng;
        // Double DQN: online selects action, target evaluates value
        online_net_->eval();
        auto next_q_on = online_net_->forward(b_next_item, b_next_sp, b_next_ag);
        auto next_acts = next_q_on.argmax(-1);                                   // [B]
        auto next_q_tg = target_net_->forward(b_next_item, b_next_sp, b_next_ag);
        auto next_q    = next_q_tg
                           .gather(1, next_acts.unsqueeze(1)).squeeze(1);         // [B]
        targets = b_rew + cfg_.gamma * next_q * (1.0f - b_done);
        online_net_->train();
    }

    auto q_vals = online_net_
                    ->forward(b_item, b_sp, b_ag)
                    .gather(1, b_acts.unsqueeze(1)).squeeze(1);                   // [B]
    auto loss   = torch::smooth_l1_loss(q_vals, targets.detach());

    optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(online_net_->parameters(), cfg_.max_grad_norm);
    optimizer_->step();

    // Periodically hard-copy online → target
    if (global_step_ % cfg_.target_update_freq == 0)
        sync_target();

    return {loss.item<float>(), ep_reward_, eps, buffer_.size()};
}

} // namespace pommerman::rl
