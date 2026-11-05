#include <torch/torch.h>
#include <CLI/CLI.hpp>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include "pommerman/constants.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/rl/ppo_model.hpp"
#include "pommerman/rl/ppo_trainer.hpp"

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
    int    seed          = 42;
    bool   no_cuda       = false;
    std::string checkpoint = "ppo_checkpoint.pt";

    app.add_option("--num-updates",   num_updates,   "Number of PPO update iterations");
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
    app.add_option("--seed",          seed,          "Random seed");
    app.add_option("--checkpoint",    checkpoint,    "Checkpoint file path");
    app.add_flag("--no-cuda",         no_cuda,       "Disable CUDA");

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

    auto model = pommerman::rl::PPOActorCritic{};
    model->to(device);

    if (std::filesystem::exists(checkpoint)) {
        torch::load(model, checkpoint);
        std::cout << "Loaded checkpoint: " << checkpoint << "\n";
    }

    pommerman::rl::PPOConfig cfg{};
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

    pommerman::rl::PPOTrainer trainer{model, cfg, device};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Starting PPO training: " << num_updates << " updates\n";

    for (int update = 1; update <= num_updates; ++update) {
        auto stats = trainer.update(env);

        if (update % log_interval == 0) {
            std::cout
                << "Update " << update << "/" << num_updates
                << "  pl=" << stats.policy_loss
                << "  vl=" << stats.value_loss
                << "  ent=" << stats.entropy
                << "  rew=" << stats.mean_reward
                << "  ev=" << stats.explained_variance
                << "\n";
        }

        if (update % save_interval == 0) {
            torch::save(model, checkpoint);
            std::cout << "Checkpoint saved: " << checkpoint << "\n";
        }
    }

    torch::save(model, checkpoint);
    std::cout << "Training complete. Final checkpoint: " << checkpoint << "\n";
    return 0;
}
