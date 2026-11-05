#pragma once
#include <torch/torch.h>
#include "pommerman/environment.hpp"
#include "pommerman/rl/muzero/muzero_config.hpp"
#include "pommerman/rl/muzero/muzero_model.hpp"
#include "pommerman/rl/muzero/game_buffer.hpp"

namespace pommerman::rl::muzero {

struct TrainStats {
    float value_loss      = 0.f;
    float reward_loss     = 0.f;
    float policy_loss     = 0.f;
    float consistency_loss= 0.f;
    float total_loss      = 0.f;
    float mean_ep_reward  = 0.f;
    int   games_in_buf    = 0;
};

class MuZeroTrainer {
public:
    MuZeroTrainer(MuZeroModel model,
                  const MuZeroConfig& cfg,
                  torch::Device device);

    // Run one full episode of self-play (4 agents share the same model).
    // Pushes up to 4 GameSegments into the buffer.
    // Returns mean shaped reward across surviving agents.
    float self_play(pommerman::Environment& env);

    // Sample a batch and perform one gradient update.
    // Returns loss stats. Requires buffer.ready(cfg.warmup_games).
    TrainStats train_step();

    GameBuffer& buffer() { return buffer_; }
    MuZeroModel& model() { return model_; }

private:
    MuZeroConfig        cfg_;
    torch::Device       device_;
    MuZeroModel         model_;
    GameBuffer          buffer_;
    torch::optim::Adam  optimizer_;

    // Shaped reward helpers (survival + wood + kill + powerup)
    float compute_shaped_reward(
        const pommerman::Observation& prev,
        const pommerman::Observation& curr,
        float base_reward,
        bool done) const;
};

} // namespace pommerman::rl::muzero
