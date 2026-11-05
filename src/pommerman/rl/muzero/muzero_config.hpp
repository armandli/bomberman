#pragma once

namespace pommerman::rl::muzero {

struct MuZeroConfig {
    // MCTS
    int   num_simulations      = 50;
    float c_init               = 2.5f;
    float c_base               = 19652.0f;
    float dirichlet_alpha      = 0.3f;
    float dirichlet_epsilon    = 0.25f;
    float temperature          = 1.0f;    // action sampling temperature (self-play)

    // Training
    int    unroll_steps        = 5;       // K: dynamics unroll depth
    int    td_steps            = 5;       // n-step bootstrap depth
    float  discount            = 0.997f;
    int    batch_size          = 256;
    double lr                  = 2e-4;
    double weight_decay        = 1e-4;
    float  grad_clip           = 10.0f;
    int    warmup_games        = 50;      // games before training starts
    int    games_per_update    = 1;       // games collected between gradient steps

    // Loss weights
    float  policy_weight       = 1.0f;
    float  value_weight        = 0.25f;
    float  reward_weight       = 1.0f;
    float  consistency_weight  = 2.0f;   // EfficientZero SimSiam loss

    // Game buffer
    int    buffer_capacity     = 1000;    // max number of game segments

    // Network
    int    d_model             = 256;
    int    d_action            = 32;

    // Training schedule
    int    total_games         = 10000;
    int    log_interval        = 20;      // games
    int    save_interval       = 500;
};

} // namespace pommerman::rl::muzero
