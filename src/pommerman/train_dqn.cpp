#include <torch/torch.h>
#include <CLI/CLI.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "pommerman/constants.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/rl/checkpoint.hpp"
#include "pommerman/rl/dqn_model.hpp"
#include "pommerman/rl/dqn_trainer.hpp"
#include "pommerman/rl/evaluator.hpp"
#include "pommerman/rl/metrics_logger.hpp"
#include "pommerman/rl/obs_encoder.hpp"

namespace rl = pommerman::rl;

int main(int argc, char* argv[]) {
    CLI::App app{"Pommerman Rainbow DQN Self-Play Trainer"};

    int    total_steps     = 5'000'000;
    int    replay_capacity = 100'000;
    int    batch_size      = 64;
    int    warmup_steps    = 10'000;
    int    target_update   = 1'000;
    int    train_freq      = 4;
    float  eps_start       = 1.0f;
    float  eps_end         = 0.1f;
    int    eps_decay       = 1'000'000;
    float  gamma           = 0.99f;
    double lr              = 1e-4;
    float  grad_clip       = 10.0f;
    int    log_interval    = 1'000;
    int    save_interval   = 50'000;
    int    eval_interval   = 25'000;
    int    eval_games      = 20;
    int    start_iter      = 0;
    int    seed            = 42;
    bool   no_cuda         = false;
    int    n_step          = 3;
    float  per_omega       = 0.5f;
    float  per_beta_start  = 0.4f;
    int    atoms           = 51;
    float  v_min           = -2.0f;
    float  v_max           = 4.0f;
    double sigma0          = 0.5;
    bool   no_per          = false;
    bool   no_noisy        = false;
    bool   no_distributional = false;
    std::string checkpoint = "dqn_checkpoint.pt";
    std::string metrics;

    app.add_option("--total-steps",     total_steps,     "Total environment steps (absolute, incl. --start-iter)");
    app.add_option("--replay-capacity", replay_capacity, "Replay buffer capacity");
    app.add_option("--batch-size",      batch_size,      "Training batch size");
    app.add_option("--warmup-steps",    warmup_steps,    "Transitions in buffer before training starts");
    app.add_option("--target-update",   target_update,   "Target network sync frequency (env steps)");
    app.add_option("--train-freq",      train_freq,      "Gradient step every N env steps");
    app.add_option("--eps-start",       eps_start,       "Initial epsilon");
    app.add_option("--eps-end",         eps_end,         "Final epsilon");
    app.add_option("--eps-decay",       eps_decay,       "Steps to decay epsilon");
    app.add_option("--gamma",           gamma,           "Discount factor");
    app.add_option("--lr",              lr,              "Adam learning rate");
    app.add_option("--grad-clip",       grad_clip,       "Gradient norm clip");
    app.add_option("--log-interval",    log_interval,    "Log every N steps");
    app.add_option("--save-interval",   save_interval,   "Checkpoint every N steps");
    app.add_option("--eval-interval",   eval_interval,   "Evaluate vs random agents every N steps (0 = off)");
    app.add_option("--eval-games",      eval_games,      "Games per evaluation");
    app.add_option("--start-iter",      start_iter,      "Step to resume from (continues epsilon schedule)");
    app.add_option("--seed",            seed,            "Random seed");
    app.add_option("--checkpoint",      checkpoint,      "Checkpoint file path");
    app.add_option("--metrics",         metrics,         "Append JSONL metrics to this file");
    app.add_flag("--no-cuda",           no_cuda,         "Disable CUDA");
    app.add_option("--n-step",          n_step,          "Multi-step return length (1 = off)")
        ->check(CLI::PositiveNumber);
    app.add_option("--per-omega",       per_omega,       "Priority exponent")
        ->check(CLI::Range(0.0f, 1.0f));
    app.add_option("--per-beta-start",  per_beta_start,  "Initial importance-sampling exponent, annealed to 1 over --total-steps")
        ->check(CLI::Range(0.0f, 1.0f));
    app.add_option("--atoms",           atoms,           "Atoms of the return distribution")
        ->check(CLI::Range(2, 1001));
    app.add_option("--v-min",           v_min,           "Lowest atom of the return distribution");
    app.add_option("--v-max",           v_max,           "Highest atom of the return distribution");
    app.add_option("--sigma0",          sigma0,          "Initial noise scale of noisy layers")
        ->check(CLI::PositiveNumber);
    app.add_flag("--no-per",            no_per,            "Sample replay uniformly");
    app.add_flag("--no-noisy",          no_noisy,          "Plain layers and epsilon-greedy exploration");
    app.add_flag("--no-distributional", no_distributional, "Scalar Q-values and smooth L1 loss");

    CLI11_PARSE(app, argc, argv);
    if (!no_distributional && v_max <= v_min) {
        std::cerr << "--v-max must be greater than --v-min\n";
        return 1;
    }
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

    rl::DQNModelOptions model_opts{};
    model_opts.distributional = !no_distributional;
    model_opts.atoms          = atoms;
    model_opts.v_min          = v_min;
    model_opts.v_max          = v_max;
    model_opts.noisy          = !no_noisy;
    model_opts.sigma0         = sigma0;

    auto online_net = rl::DuelingDQN{model_opts};
    auto target_net = rl::DuelingDQN{model_opts};
    online_net->to(device);
    target_net->to(device);

    rl::DQNConfig cfg{};
    cfg.replay_capacity    = replay_capacity;
    cfg.batch_size         = batch_size;
    cfg.warmup_steps       = warmup_steps;
    cfg.target_update_freq = target_update;
    cfg.train_freq         = train_freq;
    cfg.eps_start          = eps_start;
    cfg.eps_end            = eps_end;
    cfg.eps_decay_steps    = eps_decay;
    cfg.gamma              = gamma;
    cfg.max_grad_norm      = grad_clip;
    cfg.lr                 = lr;
    cfg.n_step             = n_step;
    cfg.prioritized        = !no_per;
    cfg.per_omega          = per_omega;
    cfg.per_beta_start     = per_beta_start;
    cfg.total_steps        = total_steps;

    rl::DQNTrainer trainer{online_net, target_net, cfg, device};
    if (rl::load_checkpoint(online_net, trainer.optimizer(), checkpoint, lr, device)) {
        trainer.sync_target_net();
        std::cout << "Loaded checkpoint: " << checkpoint << "\n";
    }
    trainer.set_global_step(start_iter);

    rl::MetricsLogger log{metrics};

    rl::PolicyFn greedy = [&](const pommerman::Observation& o) {
        torch::NoGradGuard ng;
        online_net->eval();
        auto [it, sp, ag] = rl::obs_to_tensors({o});
        auto q = online_net->forward(it.to(device), sp.to(device), ag.to(device));
        return static_cast<pommerman::Action>(q.argmax(-1).item<int64_t>());
    };

    auto run_eval = [&](int step) {
        rl::run_and_log_eval(log, greedy, eval_games, static_cast<uint32_t>(seed), step);
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Starting DQN training: steps " << start_iter + 1 << ".." << total_steps << "\n";
    log.log("start", {{"iter", start_iter}, {"lr", lr}, {"eps_end", eps_end},
                      {"eps_decay", eps_decay}, {"batch_size", batch_size},
                      {"n_step", n_step}, {"prioritized", no_per ? 0 : 1},
                      {"noisy", no_noisy ? 0 : 1}, {"distributional", no_distributional ? 0 : 1},
                      {"atoms", atoms}, {"v_min", v_min}, {"v_max", v_max}});

    double loss_sum = 0.0, q_sum = 0.0, edge_sum = 0.0;
    int    loss_cnt = 0;
    auto   w_start  = std::chrono::steady_clock::now();
    int    w_steps  = 0;
    int    step     = start_iter;
    rl::DQNTrainer::Stats stats{};

    // Baseline evaluation of the starting policy (fixed seed: every eval plays
    // the same boards and opponents, so scores are comparable across a run)
    if (eval_interval > 0 && start_iter == 0) run_eval(0);

    while (step < total_steps && !rl::stop_requested()) {
        ++step;
        stats = trainer.step(env);
        ++w_steps;
        if (!std::isnan(stats.loss)) {
            loss_sum += stats.loss; q_sum += stats.q_mean; edge_sum += stats.edge_mass;
            ++loss_cnt;
        }

        if (step % log_interval == 0) {
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - w_start).count();
            double sps  = secs > 0 ? w_steps / secs : 0.0;
            std::cout << "Step " << step << "/" << total_steps
                      << "  loss=" << (loss_cnt ? loss_sum / loss_cnt : 0.0)
                      << "  q=" << (loss_cnt ? q_sum / loss_cnt : 0.0)
                      << "  edge=" << (loss_cnt ? edge_sum / loss_cnt : 0.0)
                      << "  eps=" << stats.epsilon << "  buf=" << stats.buffer_size
                      << "  ep_rew=" << stats.ep_reward << "  eplen=" << stats.last_ep_len
                      << "  sps=" << sps << "\n";
            if (loss_cnt > 0) {
                log.log("train", {{"iter", step}, {"env_steps", step}, {"loss", loss_sum / loss_cnt},
                                  {"q_mean", q_sum / loss_cnt}, {"edge_mass", edge_sum / loss_cnt},
                                  {"beta", stats.beta}, {"epsilon", stats.epsilon}, {"buffer_size", stats.buffer_size},
                                  {"mean_reward", stats.ep_reward}, {"episodes", stats.episodes},
                                  {"mean_ep_len", stats.last_ep_len}, {"steps_per_sec", sps}});
            } else {
                log.log("warmup", {{"iter", step}, {"buffer_size", stats.buffer_size},
                                   {"epsilon", stats.epsilon}, {"steps_per_sec", sps}});
            }
            loss_sum = q_sum = edge_sum = 0.0; loss_cnt = 0; w_steps = 0;
            w_start = std::chrono::steady_clock::now();
        }

        if (eval_interval > 0 && step % eval_interval == 0) run_eval(step);

        if (step % save_interval == 0) {
            rl::save_checkpoint(online_net, trainer.optimizer(), checkpoint);
            log.log("checkpoint", {{"iter", step}});
            std::cout << "Checkpoint saved: " << checkpoint << "\n";
        }
    }

    rl::save_checkpoint(online_net, trainer.optimizer(), checkpoint);
    bool stopped = rl::stop_requested();
    log.log(stopped ? "stopped" : "complete", {{"iter", step}});
    std::cout << (stopped ? "Training stopped" : "Training complete")
              << " at step " << step << ". Final checkpoint: " << checkpoint << "\n";
    return 0;
}
