#include <torch/torch.h>
#include <CLI/CLI.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "pommerman/constants.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/rl/checkpoint.hpp"
#include "pommerman/rl/evaluator.hpp"
#include "pommerman/rl/metrics_logger.hpp"
#include "pommerman/rl/obs_encoder.hpp"
#include "pommerman/rl/ppo_model.hpp"
#include "pommerman/rl/ppo_trainer.hpp"

namespace rl = pommerman::rl;

int main(int argc, char* argv[]) {
    CLI::App app{"Pommerman PPO Self-Play Trainer"};

    int    num_updates   = 10'000;
    int    rollout_steps = 128;
    int    ppo_epochs    = 10;
    int    minibatch     = 64;
    double lr            = 3e-4;
    float  gamma         = 0.99f;
    float  gae_lambda    = 0.95f;
    float  clip_eps      = 0.2f;
    float  value_coef    = 0.5f;
    float  entropy_coef  = 0.01f;
    float  grad_clip     = 0.5f;
    int    log_interval  = 10;
    int    save_interval = 500;
    int    eval_interval = 100;
    int    eval_games    = 20;
    int    start_iter    = 0;
    int    seed          = 42;
    bool   no_cuda       = false;
    std::string checkpoint = "ppo_checkpoint.pt";
    std::string metrics;

    app.add_option("--num-updates",   num_updates,   "Total PPO update iterations (absolute, incl. --start-iter)");
    app.add_option("--rollout-steps", rollout_steps, "Env steps per agent per rollout");
    app.add_option("--ppo-epochs",    ppo_epochs,    "Optimization epochs per update");
    app.add_option("--minibatch",     minibatch,     "Minibatch size");
    app.add_option("--lr",            lr,            "Adam learning rate");
    app.add_option("--gamma",         gamma,         "Discount factor");
    app.add_option("--gae-lambda",    gae_lambda,    "GAE lambda");
    app.add_option("--clip-eps",      clip_eps,      "PPO clip epsilon");
    app.add_option("--value-coef",    value_coef,    "Value loss coefficient");
    app.add_option("--entropy-coef",  entropy_coef,  "Entropy bonus coefficient");
    app.add_option("--grad-clip",     grad_clip,     "Gradient norm clip");
    app.add_option("--log-interval",  log_interval,  "Log every N updates");
    app.add_option("--save-interval", save_interval, "Checkpoint every N updates");
    app.add_option("--eval-interval", eval_interval, "Evaluate vs random agents every N updates (0 = off)");
    app.add_option("--eval-games",    eval_games,    "Games per evaluation");
    app.add_option("--start-iter",    start_iter,    "Iteration to resume counting from");
    app.add_option("--seed",          seed,          "Random seed");
    app.add_option("--checkpoint",    checkpoint,    "Checkpoint file path");
    app.add_option("--metrics",       metrics,       "Append JSONL metrics to this file");
    app.add_flag("--no-cuda",         no_cuda,       "Disable CUDA");

    CLI11_PARSE(app, argc, argv);
    rl::install_stop_handler();
    std::cout << std::unitbuf;  // keep train.log live when redirected to a file

    torch::manual_seed(seed + start_iter);

    torch::Device device{torch::kCPU};
    if (!no_cuda && torch::cuda::is_available()) {
        device = torch::Device{torch::kCUDA};
        std::cout << "Using CUDA\n";
    } else {
        std::cout << "Using CPU\n";
    }

    pommerman::Environment env{pommerman::GameType::FFA, false, pommerman::BOARD_SIZE,
                               static_cast<uint32_t>(seed + start_iter)};

    auto model = rl::PPOActorCritic{};
    model->to(device);

    rl::PPOConfig cfg{};
    cfg.rollout_steps  = rollout_steps;
    cfg.ppo_epochs     = ppo_epochs;
    cfg.minibatch_size = minibatch;
    cfg.clip_eps       = clip_eps;
    cfg.value_coef     = value_coef;
    cfg.entropy_coef   = entropy_coef;
    cfg.gamma          = gamma;
    cfg.gae_lambda     = gae_lambda;
    cfg.max_grad_norm  = grad_clip;
    cfg.lr             = lr;

    rl::PPOTrainer trainer{model, cfg, device};
    if (rl::load_checkpoint(model, trainer.optimizer(), checkpoint, lr, device))
        std::cout << "Loaded checkpoint: " << checkpoint << "\n";

    rl::MetricsLogger log{metrics};

    // Greedy policy for evaluation
    rl::PolicyFn greedy = [&](const pommerman::Observation& o) {
        torch::NoGradGuard ng;
        model->eval();
        auto [it, sp, ag] = rl::obs_to_tensors({o});
        auto out = model->forward(it.to(device), sp.to(device), ag.to(device));
        return static_cast<pommerman::Action>(out.logits.argmax(-1).item<int64_t>());
    };

    auto run_eval = [&](int update) {
        rl::run_and_log_eval(log, greedy, eval_games, static_cast<uint32_t>(seed), update);
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Starting PPO training: updates " << start_iter + 1 << ".." << num_updates << "\n";
    log.log("start", {{"iter", start_iter}, {"lr", lr}, {"entropy_coef", entropy_coef},
                      {"clip_eps", clip_eps}, {"rollout_steps", rollout_steps}});

    // Window accumulators between log lines
    double w_pl = 0, w_vl = 0, w_ent = 0, w_rew = 0, w_ev = 0, w_len = 0;
    int    w_n = 0, w_eps = 0, total_eps = 0;
    auto   w_start = std::chrono::steady_clock::now();
    int    update  = start_iter;

    // Baseline evaluation of the starting policy (fixed seed: every eval plays
    // the same boards and opponents, so scores are comparable across a run)
    if (eval_interval > 0 && start_iter == 0) run_eval(0);

    while (update < num_updates && !rl::stop_requested()) {
        ++update;
        auto s = trainer.update(env);
        w_pl += s.policy_loss; w_vl += s.value_loss; w_ent += s.entropy;
        w_rew += s.mean_reward; w_ev += s.explained_variance;
        w_len += s.mean_ep_len * s.episodes; w_eps += s.episodes; total_eps += s.episodes;
        ++w_n;

        if (update % log_interval == 0) {
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - w_start).count();
            double sps  = secs > 0 ? w_n * rollout_steps / secs : 0.0;
            double mlen = w_eps > 0 ? w_len / w_eps : 0.0;
            std::cout << "Update " << update << "/" << num_updates
                      << "  pl=" << w_pl / w_n << "  vl=" << w_vl / w_n
                      << "  ent=" << w_ent / w_n << "  rew=" << w_rew / w_n
                      << "  ev=" << w_ev / w_n << "  eplen=" << mlen
                      << "  sps=" << sps << "\n";
            log.log("train", {{"iter", update}, {"env_steps", double(update) * rollout_steps},
                              {"policy_loss", w_pl / w_n}, {"value_loss", w_vl / w_n},
                              {"entropy", w_ent / w_n}, {"mean_reward", w_rew / w_n},
                              {"explained_var", w_ev / w_n}, {"episodes", total_eps},
                              {"mean_ep_len", mlen}, {"steps_per_sec", sps}});
            w_pl = w_vl = w_ent = w_rew = w_ev = w_len = 0; w_n = w_eps = 0;
            w_start = std::chrono::steady_clock::now();
        }

        if (eval_interval > 0 && update % eval_interval == 0) run_eval(update);

        if (update % save_interval == 0) {
            rl::save_checkpoint(model, trainer.optimizer(), checkpoint);
            log.log("checkpoint", {{"iter", update}});
            std::cout << "Checkpoint saved: " << checkpoint << "\n";
        }
    }

    rl::save_checkpoint(model, trainer.optimizer(), checkpoint);
    bool stopped = rl::stop_requested();
    log.log(stopped ? "stopped" : "complete", {{"iter", update}});
    std::cout << (stopped ? "Training stopped" : "Training complete")
              << " at update " << update << ". Final checkpoint: " << checkpoint << "\n";
    return 0;
}
