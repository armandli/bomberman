#include "pommerman/rl/ppo_trainer.hpp"
#include "pommerman/rl/obs_encoder.hpp"

namespace pommerman::rl {

PPOTrainer::PPOTrainer(PPOActorCritic model, PPOConfig cfg, torch::Device device)
    : model_(model), cfg_(cfg), device_(device) {
    optimizer_ = std::make_unique<torch::optim::Adam>(
        model_->parameters(),
        torch::optim::AdamOptions(cfg_.lr));
}

std::vector<PPOTrainer::Step> PPOTrainer::collect_rollout(Environment& env) {
    std::vector<Step> steps;
    steps.reserve(cfg_.rollout_steps);

    if (curr_done_) {
        curr_obs_  = env.reset();
        curr_done_ = false;
    }

    model_->eval();

    for (int t = 0; t < cfg_.rollout_steps; ++t) {
        auto [item_cpu, sp_cpu, ag_cpu] = obs_to_tensors(curr_obs_);
        auto item_idx = item_cpu.to(device_);
        auto sp_cont  = sp_cpu.to(device_);
        auto ag_feat  = ag_cpu.to(device_);

        torch::Tensor actions, log_probs, values, entropy;
        {
            torch::NoGradGuard ng;
            std::tie(actions, log_probs, values, entropy) =
                model_->act(item_idx, sp_cont, ag_feat);
        }

        std::array<Action, NUM_AGENTS> acts{};
        for (int i = 0; i < NUM_AGENTS; ++i)
            acts[i] = static_cast<Action>(actions[i].item<int64_t>());

        auto [new_obs, base_rew, done, info] = env.step(acts);
        auto shaped = RewardShaper::shape(curr_obs_, new_obs, base_rew, done);

        auto rew_t  = torch::tensor(
            std::vector<float>(shaped.begin(), shaped.end()),
            torch::kFloat).to(device_);
        auto done_t = torch::full(
            {NUM_AGENTS}, done ? 1.0f : 0.0f, torch::kFloat).to(device_);

        steps.push_back({item_idx, sp_cont, ag_feat,
                         actions, log_probs, values, rew_t, done_t});

        curr_obs_  = std::move(new_obs);
        curr_done_ = done;
        if (done) {
            curr_obs_  = env.reset();
            curr_done_ = false;
        }
    }

    return steps;
}

std::pair<torch::Tensor, torch::Tensor> PPOTrainer::compute_gae(
        const std::vector<Step>& steps,
        const torch::Tensor& last_values,
        bool last_done) {
    int T = static_cast<int>(steps.size());
    auto advantages = torch::zeros({T, NUM_AGENTS}, torch::kFloat).to(device_);
    auto returns    = torch::zeros({T, NUM_AGENTS}, torch::kFloat).to(device_);
    auto gae        = torch::zeros({NUM_AGENTS},    torch::kFloat).to(device_);

    auto next_val  = last_values;
    auto next_done = torch::full({NUM_AGENTS}, last_done ? 1.0f : 0.0f,
                                 torch::kFloat).to(device_);

    for (int t = T - 1; t >= 0; --t) {
        const auto& s = steps[t];
        // TD delta
        auto delta = s.rewards + cfg_.gamma * next_val * (1.0f - next_done) - s.values;
        gae        = delta + cfg_.gamma * cfg_.gae_lambda * gae * (1.0f - next_done);
        advantages[t] = gae;
        returns[t]    = gae + s.values;

        next_val  = s.values;
        next_done = s.dones;
    }
    return {advantages, returns};
}

PPOTrainer::Stats PPOTrainer::update(Environment& env) {
    auto steps = collect_rollout(env);
    int  T     = static_cast<int>(steps.size());
    int  N     = T * NUM_AGENTS;

    // Was the last collected step a terminal?
    bool last_done = steps.back().dones[0].item<float>() > 0.5f;

    torch::Tensor last_values;
    if (last_done) {
        last_values = torch::zeros({NUM_AGENTS}, torch::kFloat).to(device_);
    } else {
        auto [li, ls, la] = obs_to_tensors(curr_obs_);
        torch::NoGradGuard ng;
        model_->eval();
        last_values = model_->forward(
            li.to(device_), ls.to(device_), la.to(device_)).value;
    }

    auto [advantages, returns] = compute_gae(steps, last_values, last_done);

    // Flatten rollout to [N] tensors for minibatch training
    std::vector<torch::Tensor> v_item, v_sp, v_ag, v_acts, v_lp, v_vals;
    v_item.reserve(T); v_sp.reserve(T); v_ag.reserve(T);
    v_acts.reserve(T); v_lp.reserve(T); v_vals.reserve(T);
    for (auto& s : steps) {
        v_item.push_back(s.item_idx);
        v_sp.push_back(s.sp_cont);
        v_ag.push_back(s.ag_feat);
        v_acts.push_back(s.actions);
        v_lp.push_back(s.log_probs);
        v_vals.push_back(s.values);
    }

    auto flat_item = torch::cat(v_item, 0);  // [N, 121]
    auto flat_sp   = torch::cat(v_sp,   0);  // [N, 121, SPATIAL_CONT_DIM]
    auto flat_ag   = torch::cat(v_ag,   0);  // [N, AGENT_FEAT_DIM]
    auto flat_acts = torch::cat(v_acts, 0);  // [N]
    auto flat_lp   = torch::cat(v_lp,   0);  // [N]
    auto flat_vals = torch::cat(v_vals, 0);  // [N]
    auto flat_adv  = advantages.reshape({N});// [N]
    auto flat_ret  = returns.reshape({N});   // [N]

    // Normalise advantages
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8f);

    model_->train();
    float total_pl = 0.0f, total_vl = 0.0f, total_ent = 0.0f;
    int   updates  = 0;

    for (int epoch = 0; epoch < cfg_.ppo_epochs; ++epoch) {
        auto perm = torch::randperm(N, torch::TensorOptions()
                                         .dtype(torch::kLong)
                                         .device(device_));

        for (int start = 0; start + cfg_.minibatch_size <= N;
             start += cfg_.minibatch_size) {
            auto idx = perm.slice(0, start, start + cfg_.minibatch_size);

            auto mb_item    = flat_item.index_select(0, idx);
            auto mb_sp      = flat_sp.index_select(0, idx);
            auto mb_ag      = flat_ag.index_select(0, idx);
            auto mb_actions = flat_acts.index_select(0, idx);
            auto mb_old_lp  = flat_lp.index_select(0, idx);
            auto mb_adv     = flat_adv.index_select(0, idx);
            auto mb_ret     = flat_ret.index_select(0, idx);
            auto mb_old_val = flat_vals.index_select(0, idx);

            auto [new_lp, new_val, entropy] = model_->evaluate_actions(
                mb_item, mb_sp, mb_ag, mb_actions);

            // PPO clipped policy loss
            auto ratio      = torch::exp(new_lp - mb_old_lp);
            auto clip_adv   = torch::clamp(ratio,
                                           1.0f - cfg_.clip_eps,
                                           1.0f + cfg_.clip_eps) * mb_adv;
            auto policy_loss = -torch::min(ratio * mb_adv, clip_adv).mean();

            // Clipped value loss (OpenAI implementation)
            auto val_clipped = mb_old_val + torch::clamp(new_val - mb_old_val,
                                                         -cfg_.clip_eps,
                                                          cfg_.clip_eps);
            auto value_loss = torch::max(
                torch::pow(new_val     - mb_ret, 2),
                torch::pow(val_clipped - mb_ret, 2)
            ).mean() * cfg_.value_coef;

            auto loss = policy_loss + value_loss - entropy.mean() * cfg_.entropy_coef;

            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                              cfg_.max_grad_norm);
            optimizer_->step();

            total_pl  += policy_loss.item<float>();
            total_vl  += value_loss.item<float>();
            total_ent += entropy.mean().item<float>();
            ++updates;
        }
    }

    float n       = static_cast<float>(updates);
    float mean_rew = 0.0f;
    for (auto& s : steps)
        mean_rew += s.rewards.mean().item<float>();
    mean_rew /= static_cast<float>(T);

    float var_ret  = torch::var(flat_ret).item<float>();
    float ev       = var_ret < 1e-8f ? 0.0f :
                     1.0f - torch::var(flat_ret - flat_vals).item<float>() / var_ret;

    return {total_pl / n, total_vl / n, total_ent / n, mean_rew, ev};
}

} // namespace pommerman::rl
