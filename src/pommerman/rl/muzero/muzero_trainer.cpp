#include "pommerman/rl/muzero/muzero_trainer.hpp"
#include "pommerman/rl/muzero/mcts.hpp"
#include "pommerman/rl/obs_encoder.hpp"
#include "pommerman/rl/reward_shaper.hpp"
#include "pommerman/constants.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace pommerman::rl::muzero {

static constexpr int NUM_AGENTS = 4;

// ── Constructor ───────────────────────────────────────────────────────────────

MuZeroTrainer::MuZeroTrainer(MuZeroModel model,
                              const MuZeroConfig& cfg,
                              torch::Device device)
    : cfg_(cfg)
    , device_(device)
    , model_(std::move(model))
    , buffer_(cfg)
    , optimizer_(model_->parameters(),
                 torch::optim::AdamOptions(cfg.lr).weight_decay(cfg.weight_decay))
{
    model_->to(device_);
}

// ── self_play ─────────────────────────────────────────────────────────────────

float MuZeroTrainer::self_play(pommerman::Environment& env) {
    env.reset();
    auto obs = env.get_observations();

    // Per-agent game segments
    std::array<GameSegment, NUM_AGENTS> segs;
    std::array<float, NUM_AGENTS> ep_rewards{};
    std::array<bool, NUM_AGENTS> alive{true, true, true, true};

    model_->eval();

    bool done = false;
    while (!done) {
        std::vector<pommerman::Action> actions(NUM_AGENTS, pommerman::Action::Stop);

        for (int a = 0; a < NUM_AGENTS; ++a) {
            if (!alive[a]) continue;

            // Build observation tensors for this agent (batch=1)
            std::vector<pommerman::Observation> single_obs = {obs[a]};
            auto [item_idx, sp_cont, ag_feat] =
                pommerman::rl::obs_to_tensors(single_obs);

            // Encode to hidden state
            torch::Tensor hidden;
            {
                torch::NoGradGuard ng;
                hidden = model_->represent(
                    item_idx.to(device_),
                    sp_cont.to(device_),
                    ag_feat.to(device_));  // [1, D_MODEL]
            }

            // MCTS
            auto [policy_t, root_value] = run_mcts(hidden, model_, cfg_, device_);

            // Store obs in segment
            segs[a].item_idxs.push_back(item_idx[0].cpu());
            segs[a].sp_conts.push_back(sp_cont[0].cpu());
            segs[a].ag_feats.push_back(ag_feat[0].cpu());
            segs[a].mcts_policies.push_back(policy_t.cpu());

            // Sample action
            int act;
            if (cfg_.temperature < 1e-6f) {
                act = static_cast<int>(policy_t.argmax().item<int64_t>());
            } else {
                // Temperature-adjusted sampling
                auto adj = policy_t.pow(1.f / cfg_.temperature);
                adj /= adj.sum();
                act = static_cast<int>(
                    torch::multinomial(adj, 1).item<int64_t>());
            }
            actions[a] = static_cast<pommerman::Action>(act);
            segs[a].actions.push_back(act);
            segs[a].root_values.push_back(root_value);
        }

        auto prev_obs = obs;
        auto [next_obs, base_rewards, is_done, _game_info] = env.step(actions);
        done = is_done;
        obs = next_obs;

        // Same shaping as PPO/DQN; includes -1 on the step an agent dies
        auto shaped = RewardShaper::shape(prev_obs, obs, base_rewards, done);
        for (int a = 0; a < NUM_AGENTS; ++a) {
            if (!alive[a]) continue;
            segs[a].rewards.push_back(shaped[a]);
            ep_rewards[a] += shaped[a];
            // An agent's segment ends on the step it dies (or at game end)
            if (RewardShaper::agent_terminal(obs, a, done)) alive[a] = false;
        }
    }

    // Compute value targets and push segments
    float total_reward = 0.f;
    int   pushed = 0;
    for (int a = 0; a < NUM_AGENTS; ++a) {
        if (segs[a].length() == 0) continue;
        segs[a].compute_value_targets(cfg_.discount, cfg_.td_steps);
        buffer_.add_game(std::move(segs[a]));
        total_reward += ep_rewards[a];
        ++pushed;
    }

    return pushed > 0 ? total_reward / static_cast<float>(pushed) : 0.f;
}

// ── train_step ────────────────────────────────────────────────────────────────

TrainStats MuZeroTrainer::train_step() {
    auto batch = buffer_.sample_batch(device_);
    int K = cfg_.unroll_steps;

    model_->train();
    optimizer_.zero_grad();

    // Initial representation
    torch::Tensor hidden = model_->represent(
        batch.item_idx, batch.sp_cont, batch.ag_feat);  // [B, D_MODEL]

    float total_v_loss = 0.f, total_r_loss = 0.f,
          total_p_loss = 0.f, total_c_loss = 0.f;

    torch::Tensor loss = torch::zeros({1}, torch::TensorOptions().device(device_));

    for (int k = 0; k < K; ++k) {
        auto action_k    = batch.actions.select(1, k);         // [B]
        auto reward_k    = batch.rewards.select(1, k);         // [B]
        auto policy_tgt  = batch.mcts_policies.select(1, k);  // [B, NUM_ACTIONS]
        auto value_tgt   = batch.value_targets.select(1, k);  // [B]

        // Dynamics step
        auto [next_hidden, pred_reward] = model_->step(hidden, action_k);

        // Prediction from current hidden
        auto [pred_logits, pred_value] = model_->predict(hidden);

        // Policy cross-entropy:  -sum(target * log_softmax(logits))
        auto log_probs   = torch::log_softmax(pred_logits, 1);
        auto p_loss      = -(policy_tgt * log_probs).sum(1).mean();

        // Value MSE
        auto v_loss = torch::mse_loss(pred_value, value_tgt);

        // Reward MSE
        auto r_loss = torch::mse_loss(pred_reward, reward_k);

        // EfficientZero consistency: dynamics' predicted hidden for t+k+1 vs the
        // representation of the real observation at t+k+1 (stop-gradient target),
        // masked where that observation does not exist (past episode end).
        torch::Tensor c_loss;
        {
            auto mask = batch.next_mask.select(1, k);                 // [B]
            torch::Tensor h_target;
            {
                torch::NoGradGuard ng;
                h_target = model_->represent(batch.next_item_idx.select(1, k),
                                             batch.next_sp_cont.select(1, k),
                                             batch.next_ag_feat.select(1, k));
            }
            auto z_dyn  = model_->simsiam->project(next_hidden);
            auto p_dyn  = model_->simsiam->predict_proj(z_dyn);
            auto z_repr = model_->simsiam->project(h_target).detach();
            auto cos    = torch::nn::functional::cosine_similarity(
                p_dyn, z_repr,
                torch::nn::functional::CosineSimilarityFuncOptions().dim(1)); // [B]
            c_loss = -(cos * mask).sum() / mask.sum().clamp_min(1.0f);
        }

        loss = loss
             + cfg_.policy_weight      * p_loss
             + cfg_.value_weight       * v_loss
             + cfg_.reward_weight      * r_loss
             + cfg_.consistency_weight * c_loss;

        total_p_loss += p_loss.item<float>();
        total_v_loss += v_loss.item<float>();
        total_r_loss += r_loss.item<float>();
        total_c_loss += c_loss.item<float>();

        // Carry hidden forward, scaling its gradient by 0.5 (MuZero) so later
        // unroll steps still train the dynamics network without exploding.
        hidden = 0.5f * next_hidden + 0.5f * next_hidden.detach();
    }

    loss = loss / static_cast<float>(K);
    loss.backward();

    torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.grad_clip);
    optimizer_.step();

    TrainStats stats;
    stats.value_loss       = total_v_loss / K;
    stats.reward_loss      = total_r_loss / K;
    stats.policy_loss      = total_p_loss / K;
    stats.consistency_loss = total_c_loss / K;
    stats.total_loss       = loss.item<float>();
    stats.games_in_buf     = buffer_.size();
    return stats;
}

} // namespace pommerman::rl::muzero
