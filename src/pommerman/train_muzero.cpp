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
#include "pommerman/rl/muzero/mcts.hpp"
#include "pommerman/rl/muzero/muzero_config.hpp"
#include "pommerman/rl/muzero/muzero_model.hpp"
#include "pommerman/rl/muzero/muzero_trainer.hpp"

namespace rl = pommerman::rl;
namespace mz = pommerman::rl::muzero;

int main(int argc, char* argv[]) {
    CLI::App app{"Pommerman MuZero/EfficientZero Self-Play Trainer"};

    int    total_games      = 10000;
    int    num_simulations  = 50;
    int    unroll_steps     = 5;
    int    td_steps         = 5;
    int    batch_size       = 256;
    int    warmup_games     = 50;
    int    buffer_capacity  = 1000;
    int    train_steps_per_game = 1;
    float  discount         = 0.997f;
    double lr               = 2e-4;
    double weight_decay     = 1e-4;
    float  grad_clip        = 10.0f;
    float  temperature      = 1.0f;
    float  dirichlet_alpha  = 0.3f;
    float  dirichlet_eps    = 0.25f;
    float  c_init           = 2.5f;
    float  c_base           = 19652.f;
    float  policy_weight    = 1.0f;
    float  value_weight     = 0.25f;
    float  reward_weight    = 1.0f;
    float  consist_weight   = 2.0f;
    int    log_interval     = 20;
    int    save_interval    = 500;
    int    eval_interval    = 100;
    int    eval_games       = 20;
    int    eval_sims        = 0;
    int    start_iter       = 0;
    int    seed             = 42;
    bool   no_cuda          = false;
    std::string checkpoint  = "muzero_checkpoint.pt";
    std::string metrics;

    app.add_option("--total-games",      total_games,      "Total self-play games (absolute, incl. --start-iter)");
    app.add_option("--num-simulations",  num_simulations,  "MCTS simulations per move");
    app.add_option("--unroll-steps",     unroll_steps,     "Dynamics unroll depth K");
    app.add_option("--td-steps",         td_steps,         "n-step TD bootstrap depth");
    app.add_option("--batch-size",       batch_size,       "Training batch size");
    app.add_option("--warmup-games",     warmup_games,     "Games before training starts");
    app.add_option("--buffer-capacity",  buffer_capacity,  "Max game segments in buffer (4 per game)");
    app.add_option("--train-steps-per-game", train_steps_per_game, "Gradient steps after each game");
    app.add_option("--discount",         discount,         "Discount factor");
    app.add_option("--lr",               lr,               "Adam learning rate");
    app.add_option("--weight-decay",     weight_decay,     "Adam weight decay");
    app.add_option("--grad-clip",        grad_clip,        "Gradient norm clip");
    app.add_option("--temperature",      temperature,      "Action sampling temperature");
    app.add_option("--dirichlet-alpha",  dirichlet_alpha,  "Dirichlet noise alpha");
    app.add_option("--dirichlet-eps",    dirichlet_eps,    "Dirichlet noise epsilon");
    app.add_option("--c-init",           c_init,           "PUCT c_init");
    app.add_option("--c-base",           c_base,           "PUCT c_base");
    app.add_option("--policy-weight",    policy_weight,    "Policy loss weight");
    app.add_option("--value-weight",     value_weight,     "Value loss weight");
    app.add_option("--reward-weight",    reward_weight,    "Reward loss weight");
    app.add_option("--consist-weight",   consist_weight,   "SimSiam consistency weight");
    app.add_option("--log-interval",     log_interval,     "Log every N games");
    app.add_option("--save-interval",    save_interval,    "Checkpoint every N games");
    app.add_option("--eval-interval",    eval_interval,    "Evaluate vs random agents every N games (0 = off)");
    app.add_option("--eval-games",       eval_games,       "Games per evaluation");
    app.add_option("--eval-sims",        eval_sims,        "MCTS sims for eval moves (0 = raw policy head)");
    app.add_option("--start-iter",       start_iter,       "Game index to resume counting from");
    app.add_option("--seed",             seed,             "Random seed");
    app.add_option("--checkpoint",       checkpoint,       "Checkpoint file path");
    app.add_option("--metrics",          metrics,          "Append JSONL metrics to this file");
    app.add_flag("--no-cuda",            no_cuda,          "Disable CUDA");

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

    mz::MuZeroConfig cfg;
    cfg.total_games        = total_games;
    cfg.num_simulations    = num_simulations;
    cfg.unroll_steps       = unroll_steps;
    cfg.td_steps           = td_steps;
    cfg.batch_size         = batch_size;
    cfg.warmup_games       = warmup_games;
    cfg.buffer_capacity    = buffer_capacity;
    cfg.discount           = discount;
    cfg.lr                 = lr;
    cfg.weight_decay       = weight_decay;
    cfg.grad_clip          = grad_clip;
    cfg.temperature        = temperature;
    cfg.dirichlet_alpha    = dirichlet_alpha;
    cfg.dirichlet_epsilon  = dirichlet_eps;
    cfg.c_init             = c_init;
    cfg.c_base             = c_base;
    cfg.policy_weight      = policy_weight;
    cfg.value_weight       = value_weight;
    cfg.reward_weight      = reward_weight;
    cfg.consistency_weight = consist_weight;
    cfg.log_interval       = log_interval;
    cfg.save_interval      = save_interval;

    auto model = mz::MuZeroModel{};
    model->to(device);

    pommerman::Environment env{pommerman::GameType::FFA, false, pommerman::BOARD_SIZE,
                               static_cast<uint32_t>(seed + start_iter)};

    mz::MuZeroTrainer trainer{model, cfg, device};
    if (rl::load_checkpoint(model, trainer.optimizer(), checkpoint, lr, device))
        std::cout << "Loaded checkpoint: " << checkpoint << "\n";

    rl::MetricsLogger log{metrics};

    // Evaluation policy: raw policy head (fast) or noise-free MCTS
    mz::MuZeroConfig eval_cfg = cfg;
    eval_cfg.num_simulations   = eval_sims;
    eval_cfg.dirichlet_epsilon = 0.0f;
    rl::PolicyFn eval_policy = [&](const pommerman::Observation& o) {
        torch::NoGradGuard ng;
        model->eval();
        auto [it, sp, ag] = rl::obs_to_tensors({o});
        auto h = model->represent(it.to(device), sp.to(device), ag.to(device));
        if (eval_sims > 0) {
            auto [pol, v] = mz::run_mcts(h, model, eval_cfg, device);
            return static_cast<pommerman::Action>(pol.argmax().item<int64_t>());
        }
        auto [logits, value] = model->predict(h);
        return static_cast<pommerman::Action>(logits.argmax(-1).item<int64_t>());
    };

    auto run_eval = [&](int game) {
        rl::run_and_log_eval(log, eval_policy, eval_games, static_cast<uint32_t>(seed), game);
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Starting MuZero training: games " << start_iter + 1 << ".." << total_games << "\n";
    log.log("start", {{"iter", start_iter}, {"lr", lr}, {"num_simulations", num_simulations},
                      {"unroll_steps", unroll_steps}, {"batch_size", batch_size}});

    double w_rew = 0, w_len = 0, w_tot = 0, w_pl = 0, w_vl = 0, w_rl = 0, w_cl = 0;
    int    w_games = 0, w_train = 0;
    long   env_steps = 0;
    auto   w_start = std::chrono::steady_clock::now();
    int    game = start_iter;
    // The buffer holds one segment per agent, i.e. NUM_AGENTS per game; after a
    // resume it is empty again and refills before training restarts.
    const int warmup_segments = warmup_games * pommerman::NUM_AGENTS;

    // Baseline evaluation of the starting policy (fixed seed: every eval plays
    // the same boards and opponents, so scores are comparable across a run)
    if (eval_interval > 0 && start_iter == 0) run_eval(0);

    while (game < total_games && !rl::stop_requested()) {
        ++game;
        w_rew += trainer.self_play(env);
        w_len += env.step_count();
        env_steps += env.step_count();
        ++w_games;

        if (trainer.buffer().ready(warmup_segments)) {
            for (int k = 0; k < train_steps_per_game; ++k) {
                auto s = trainer.train_step();
                w_tot += s.total_loss; w_pl += s.policy_loss; w_vl += s.value_loss;
                w_rl += s.reward_loss; w_cl += s.consistency_loss; ++w_train;
            }
        }

        if (game % log_interval == 0) {
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - w_start).count();
            double gps  = secs > 0 ? w_games / secs : 0.0;
            double sps  = secs > 0 ? w_len / secs : 0.0;
            std::cout << "Game " << game << "/" << total_games
                      << "  rew=" << w_rew / w_games << "  eplen=" << w_len / w_games
                      << "  buf=" << trainer.buffer().size() << "  games/s=" << gps;
            if (w_train > 0)
                std::cout << "  total=" << w_tot / w_train << "  pl=" << w_pl / w_train
                          << "  vl=" << w_vl / w_train << "  rl=" << w_rl / w_train
                          << "  cl=" << w_cl / w_train;
            std::cout << "\n";
            if (w_train > 0) {
                log.log("train", {{"iter", game}, {"env_steps", double(env_steps)},
                                  {"loss", w_tot / w_train}, {"policy_loss", w_pl / w_train},
                                  {"value_loss", w_vl / w_train}, {"reward_loss", w_rl / w_train},
                                  {"consistency_loss", w_cl / w_train},
                                  {"mean_reward", w_rew / w_games}, {"mean_ep_len", w_len / w_games},
                                  {"buffer_size", trainer.buffer().size()},
                                  {"games_per_sec", gps}, {"steps_per_sec", sps}});
            } else {
                log.log("warmup", {{"iter", game}, {"buffer_size", trainer.buffer().size()},
                                   {"mean_reward", w_rew / w_games}, {"mean_ep_len", w_len / w_games},
                                   {"games_per_sec", gps}});
            }
            w_rew = w_len = w_tot = w_pl = w_vl = w_rl = w_cl = 0; w_games = w_train = 0;
            w_start = std::chrono::steady_clock::now();
        }

        if (eval_interval > 0 && game % eval_interval == 0) run_eval(game);

        if (game % save_interval == 0) {
            rl::save_checkpoint(model, trainer.optimizer(), checkpoint);
            log.log("checkpoint", {{"iter", game}});
            std::cout << "Checkpoint saved: " << checkpoint << "\n";
        }
    }

    rl::save_checkpoint(model, trainer.optimizer(), checkpoint);
    bool stopped = rl::stop_requested();
    log.log(stopped ? "stopped" : "complete", {{"iter", game}});
    std::cout << (stopped ? "Training stopped" : "Training complete")
              << " at game " << game << ". Final checkpoint: " << checkpoint << "\n";
    return 0;
}
