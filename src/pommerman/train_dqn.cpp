#include <torch/torch.h>
#include <CLI/CLI.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include "pommerman/constants.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/rl/dqn_model.hpp"
#include "pommerman/rl/dqn_trainer.hpp"

int main(int argc, char* argv[]) {
    CLI::App app{"Pommerman Double Dueling DQN Self-Play Trainer"};

    int    total_steps     = 5'000'000;
    int    replay_capacity = 100'000;
    int    batch_size      = 64;
    int    warmup_steps    = 10'000;
    int    target_update   = 1'000;
    float  eps_start       = 1.0f;
    float  eps_end         = 0.1f;
    int    eps_decay       = 1'000'000;
    float  gamma           = 0.99f;
    double lr              = 1e-4;
    float  grad_clip       = 10.0f;
    int    log_interval    = 1'000;
    int    save_interval   = 50'000;
    int    seed            = 42;
    bool   no_cuda         = false;
    std::string checkpoint = "dqn_checkpoint.pt";

    app.add_option("--total-steps",     total_steps,     "Total environment steps");
    app.add_option("--replay-capacity", replay_capacity, "Replay buffer capacity");
    app.add_option("--batch-size",      batch_size,      "Training batch size");
    app.add_option("--warmup-steps",    warmup_steps,    "Steps before training starts");
    app.add_option("--target-update",   target_update,   "Target network sync frequency");
    app.add_option("--eps-start",       eps_start,       "Initial epsilon");
    app.add_option("--eps-end",         eps_end,         "Final epsilon");
    app.add_option("--eps-decay",       eps_decay,       "Steps to decay epsilon");
    app.add_option("--gamma",           gamma,           "Discount factor");
    app.add_option("--lr",              lr,              "Adam learning rate");
    app.add_option("--grad-clip",       grad_clip,       "Gradient norm clip");
    app.add_option("--log-interval",    log_interval,    "Log every N steps");
    app.add_option("--save-interval",   save_interval,   "Checkpoint every N steps");
    app.add_option("--seed",            seed,            "Random seed");
    app.add_option("--checkpoint",      checkpoint,      "Checkpoint file path");
    app.add_flag("--no-cuda",           no_cuda,         "Disable CUDA");

    CLI11_PARSE(app, argc, argv);

    torch::manual_seed(seed);

    torch::Device device{torch::kCPU};
    if (!no_cuda && torch::cuda::is_available()) {
        device = torch::Device{torch::kCUDA};
        std::cout << "Using CUDA (Flash Attention auto-selected by SDPA)\n";
    } else {
        std::cout << "Using CPU\n";
    }

    pommerman::Environment env{pommerman::GameType::FFA, false,
                                pommerman::BOARD_SIZE, static_cast<uint32_t>(seed)};

    auto online_net = pommerman::rl::DuelingDQN{};
    auto target_net = pommerman::rl::DuelingDQN{};
    online_net->to(device);
    target_net->to(device);

    if (std::filesystem::exists(checkpoint)) {
        torch::load(online_net, checkpoint);
        torch::load(target_net, checkpoint);
        std::cout << "Loaded checkpoint: " << checkpoint << "\n";
    }

    pommerman::rl::DQNConfig cfg{};
    cfg.replay_capacity    = replay_capacity;
    cfg.batch_size         = batch_size;
    cfg.warmup_steps       = warmup_steps;
    cfg.target_update_freq = target_update;
    cfg.eps_start          = eps_start;
    cfg.eps_end            = eps_end;
    cfg.eps_decay_steps    = eps_decay;
    cfg.gamma              = gamma;
    cfg.max_grad_norm      = grad_clip;
    cfg.lr                 = lr;

    pommerman::rl::DQNTrainer trainer{online_net, target_net, cfg, device};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Starting DQN training: " << total_steps << " steps\n";

    float loss_sum  = 0.0f;
    int   loss_cnt  = 0;

    for (int step = 1; step <= total_steps; ++step) {
        auto stats = trainer.step(env);

        if (!std::isnan(stats.loss)) {
            loss_sum += stats.loss;
            ++loss_cnt;
        }

        if (step % log_interval == 0) {
            float avg_loss = loss_cnt > 0 ? loss_sum / static_cast<float>(loss_cnt) : 0.0f;
            std::cout
                << "Step " << step << "/" << total_steps
                << "  loss=" << avg_loss
                << "  eps=" << stats.epsilon
                << "  buf=" << stats.buffer_size
                << "  ep_rew=" << stats.ep_reward
                << "\n";
            loss_sum = 0.0f;
            loss_cnt = 0;
        }

        if (step % save_interval == 0) {
            torch::save(online_net, checkpoint);
            std::cout << "Checkpoint saved: " << checkpoint << "\n";
        }
    }

    torch::save(online_net, checkpoint);
    std::cout << "Training complete. Final checkpoint: " << checkpoint << "\n";
    return 0;
}
