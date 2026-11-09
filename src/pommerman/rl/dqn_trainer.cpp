#include "pommerman/rl/dqn_trainer.hpp"
#include "pommerman/rl/obs_encoder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace pommerman::rl {

DQNTrainer::DQNTrainer(DuelingDQN online_net, DuelingDQN target_net,
                       DQNConfig cfg, torch::Device device)
    : online_net_(online_net), target_net_(target_net),
      cfg_(cfg), device_(device),
      buffer_(cfg.replay_capacity, cfg.prioritized, cfg.per_omega),
      nstep_(NUM_AGENTS, NStepQueue(cfg.n_step, cfg.gamma)) {
    optimizer_ = std::make_unique<torch::optim::Adam>(
        online_net_->parameters(),
        torch::optim::AdamOptions(cfg_.lr));
    sync_target();
    // Training mode so noisy layers apply noise. It changes nothing else: the
    // encoder has no dropout or batch norm.
    online_net_->train();
    target_net_->train();
}

float DQNTrainer::current_epsilon() const {
    if (online_net_->opts.noisy) return 0.0f;  // exploration comes from the noise
    float progress = std::min(1.0f,
        static_cast<float>(global_step_) / static_cast<float>(cfg_.eps_decay_steps));
    return cfg_.eps_start + (cfg_.eps_end - cfg_.eps_start) * progress;
}

float DQNTrainer::current_beta() const {
    float progress = std::min(1.0f,
        static_cast<float>(global_step_) / static_cast<float>(std::max(1, cfg_.total_steps)));
    return cfg_.per_beta_start + (1.0f - cfg_.per_beta_start) * progress;
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

    // Evaluation puts the shared online net in eval mode between steps.
    online_net_->train();

    float eps = current_epsilon();

    // Encode current observations (keep CPU copy for buffer)
    auto [item_cpu, sp_cpu, ag_cpu] = obs_to_tensors(curr_obs_);

    std::array<Action, NUM_AGENTS> acts{};

    // Independent epsilon-greedy decision per agent
    {
        torch::NoGradGuard ng;
        online_net_->reset_noise();
        auto q = online_net_->forward(
            item_cpu.to(device_), sp_cpu.to(device_), ag_cpu.to(device_));
        auto chosen = q.argmax(-1).cpu(); // [4]
        for (int i = 0; i < NUM_AGENTS; ++i) {
            if (eps > 0.0f && torch::rand({1}).item<float>() < eps)
                acts[i] = static_cast<Action>(
                    torch::randint(0, NUM_ACTIONS, {1}).item<int64_t>());
            else
                acts[i] = static_cast<Action>(chosen[i].item<int64_t>());
        }
    }

    auto [new_obs, base_rew, done, info] = env.step(acts);
    auto shaped = RewardShaper::shape(curr_obs_, new_obs, base_rew, done);
    ep_reward_ += shaped[0];

    // Encode next observations (CPU)
    auto [next_item, next_sp, next_ag] = obs_to_tensors(new_obs);

    // Feed one step per agent that was alive; death is terminal for that agent,
    // so its target must not bootstrap from the post-death state. A game's end
    // is terminal for every living agent, so no queue carries over to the next.
    for (int i = 0; i < NUM_AGENTS; ++i) {
        if (!RewardShaper::agent_was_alive(curr_obs_, i)) continue;

        Transition t;
        t.item_idx      = item_cpu[i];
        t.sp_cont       = sp_cpu[i];
        t.ag_feat       = ag_cpu[i];
        t.action        = static_cast<int>(acts[i]);
        t.reward        = shaped[i];
        t.next_item_idx = next_item[i];
        t.next_sp_cont  = next_sp[i];
        t.next_ag_feat  = next_ag[i];

        const bool terminal = RewardShaper::agent_terminal(new_obs, i, done);
        for (const auto& ready : nstep_[i].push(std::move(t), terminal))
            buffer_.add(ready);
    }

    curr_obs_  = std::move(new_obs);
    curr_done_ = done;
    ++ep_len_;
    if (done) {
        ++episodes_;
        last_ep_len_    = ep_len_;
        ep_len_         = 0;
        curr_obs_       = env.reset();
        curr_done_      = false;
        last_ep_reward_ = ep_reward_;
        ep_reward_      = 0.0f;
    }

    ++global_step_;

    // Not enough data yet, or not an update step
    const int min_size = std::max(cfg_.warmup_steps, cfg_.batch_size);
    if (buffer_.size() < min_size ||
        global_step_ % std::max(1, cfg_.train_freq) != 0)
        return {std::numeric_limits<float>::quiet_NaN(),
                last_ep_reward_, eps, buffer_.size(), episodes_, last_ep_len_,
                0.0f, 0.0f, current_beta()};

    const float beta = current_beta();
    const UpdateResult result = learn();

    // Periodically hard-copy online → target (counted in env steps; this runs
    // only on update steps, so check the window rather than exact equality)
    if (global_step_ / cfg_.target_update_freq !=
        (global_step_ - std::max(1, cfg_.train_freq)) / cfg_.target_update_freq)
        sync_target();

    return {result.loss, last_ep_reward_, eps, buffer_.size(),
            episodes_, last_ep_len_, result.q_mean, result.edge_mass, beta};
}

DQNTrainer::UpdateResult DQNTrainer::learn() {
    auto b = buffer_.sample(cfg_.batch_size, current_beta(), device_);
    const bool distributional = online_net_->opts.distributional;

    torch::Tensor per_sample_loss;  // [B], differentiable
    torch::Tensor error;            // [B], the priority signal
    float q_mean    = 0.0f;
    float edge_mass = 0.0f;

    // Double DQN: online selects action, target evaluates value. Each network
    // pass below draws independent noise.
    torch::Tensor next_acts;
    {
        torch::NoGradGuard ng;
        online_net_->reset_noise();
        target_net_->reset_noise();
        next_acts = online_net_->forward(b.next_item_idx, b.next_sp_cont, b.next_ag_feat)
                        .argmax(-1);                                              // [B]
    }
    online_net_->reset_noise();

    if (distributional) {
        const auto& support = online_net_->support;
        const int64_t atoms = online_net_->num_atoms();
        auto pick_action = [atoms](const torch::Tensor& per_action, const torch::Tensor& acts) {
            auto idx = acts.view({-1, 1, 1}).expand({-1, 1, atoms});
            return per_action.gather(1, idx).squeeze(1);                          // [B, atoms]
        };

        torch::Tensor target;
        {
            torch::NoGradGuard ng;
            auto next_lp = target_net_->log_probs(b.next_item_idx, b.next_sp_cont, b.next_ag_feat);
            target = categorical_projection(
                pick_action(next_lp, next_acts).exp(), b.rewards, b.discounts, support);
        }

        auto log_p = pick_action(
            online_net_->log_probs(b.item_idx, b.sp_cont, b.ag_feat), b.actions);

        // Cross-entropy equals KL(target || p) up to the target's entropy, which
        // has no gradient. The priority uses true KL so that a fully learned
        // transition drops towards zero even when its target distribution is wide.
        per_sample_loss = -(target * log_p).sum(-1);
        auto entropy    = -(target * target.clamp_min(1e-8f).log()).sum(-1);
        error           = (per_sample_loss.detach() - entropy).clamp_min(0.0f);

        q_mean    = (log_p.detach().exp() * support).sum(-1).mean().item<float>();
        edge_mass = (target.select(1, 0) + target.select(1, atoms - 1)).mean().item<float>();
    } else {
        torch::Tensor targets;
        {
            torch::NoGradGuard ng;
            auto next_q = target_net_->forward(b.next_item_idx, b.next_sp_cont, b.next_ag_feat)
                              .gather(1, next_acts.unsqueeze(1)).squeeze(1);      // [B]
            targets = b.rewards + b.discounts * next_q;
        }
        auto q_vals = online_net_->forward(b.item_idx, b.sp_cont, b.ag_feat)
                          .gather(1, b.actions.unsqueeze(1)).squeeze(1);          // [B]
        per_sample_loss = torch::smooth_l1_loss(q_vals, targets, at::Reduction::None);
        error           = (q_vals.detach() - targets).abs();
        q_mean          = q_vals.detach().mean().item<float>();
    }

    auto loss = (b.weights * per_sample_loss).mean();

    optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(online_net_->parameters(), cfg_.max_grad_norm);
    optimizer_->step();

    auto error_cpu = error.cpu();
    auto error_acc = error_cpu.accessor<float, 1>();
    std::vector<float> errors(error_cpu.size(0));
    for (size_t i = 0; i < errors.size(); ++i) errors[i] = error_acc[i];
    buffer_.update_priorities(b.indices, errors);

    return {error_cpu.mean().item<float>(), q_mean, edge_mass};
}

} // namespace pommerman::rl
