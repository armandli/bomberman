#include <torch/torch.h>
#include <CLI/CLI.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include "pommerman/constants.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/rl/muzero/muzero_config.hpp"
#include "pommerman/rl/muzero/muzero_model.hpp"
#include "pommerman/rl/muzero/muzero_trainer.hpp"

int main(int argc, char* argv[]) {
    CLI::App app{"Pommerman MuZero/EfficientZero Self-Play Trainer"};

    int    total_games      = 10000;
    int    num_simulations  = 50;
    int    unroll_steps     = 5;
    int    td_steps         = 5;
    int    batch_size       = 256;
    int    warmup_games     = 50;
    int    buffer_capacity  = 1000;
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
    int    seed             = 42;
    bool   no_cuda          = false;
    std::string checkpoint  = "muzero_checkpoint.pt";

    app.add_option("--total-games",      total_games,      "Total self-play games");
    app.add_option("--num-simulations",  num_simulations,  "MCTS simulations per move");
    app.add_option("--unroll-steps",     unroll_steps,     "Dynamics unroll depth K");
    app.add_option("--td-steps",         td_steps,         "n-step TD bootstrap depth");
    app.add_option("--batch-size",       batch_size,       "Training batch size");
    app.add_option("--warmup-games",     warmup_games,     "Games before training starts");
    app.add_option("--buffer-capacity",  buffer_capacity,  "Max game segments in buffer");
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
    app.add_option("--seed",             seed,             "Random seed");
    app.add_option("--checkpoint",       checkpoint,       "Checkpoint file path");
    app.add_flag("--no-cuda",            no_cuda,          "Disable CUDA");

    CLI11_PARSE(app, argc, argv);

    torch::manual_seed(seed);

    torch::Device device{torch::kCPU};
    if (!no_cuda && torch::cuda::is_available()) {
        device = torch::Device{torch::kCUDA};
        std::cout << "Using CUDA\n";
    } else {
        std::cout << "Using CPU\n";
    }

    pommerman::rl::muzero::MuZeroConfig cfg;
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

    auto model = pommerman::rl::muzero::MuZeroModel{};
    model->to(device);

    if (std::filesystem::exists(checkpoint)) {
        torch::load(model, checkpoint);
        std::cout << "Loaded checkpoint: " << checkpoint << "\n";
    }

    pommerman::Environment env{pommerman::GameType::FFA, false,
                                pommerman::BOARD_SIZE, static_cast<uint32_t>(seed)};

    pommerman::rl::muzero::MuZeroTrainer trainer{model, cfg, device};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Starting MuZero training: " << total_games << " games\n";

    float reward_sum = 0.f;
    int   reward_cnt = 0;

    for (int game = 1; game <= total_games; ++game) {
        float ep_reward = trainer.self_play(env);
        reward_sum += ep_reward;
        ++reward_cnt;

        if (trainer.buffer().ready(warmup_games)) {
            auto stats = trainer.train_step();

            if (game % log_interval == 0) {
                float avg_rew = reward_cnt > 0
                                    ? reward_sum / static_cast<float>(reward_cnt)
                                    : 0.f;
                std::cout
                    << "Game " << game << "/" << total_games
                    << "  total="   << stats.total_loss
                    << "  pl="      << stats.policy_loss
                    << "  vl="      << stats.value_loss
                    << "  rl="      << stats.reward_loss
                    << "  cl="      << stats.consistency_loss
                    << "  rew="     << avg_rew
                    << "  buf="     << stats.games_in_buf
                    << "\n";
                reward_sum = 0.f;
                reward_cnt = 0;
            }
        }

        if (game % save_interval == 0) {
            torch::save(model, checkpoint);
            std::cout << "Checkpoint saved: " << checkpoint << "\n";
        }
    }

    torch::save(model, checkpoint);
    std::cout << "Training complete. Final checkpoint: " << checkpoint << "\n";
    return 0;
}
