#pragma once
#include <torch/torch.h>
#include "pommerman/rl/obs_encoder.hpp"

namespace pommerman::rl {

// Linear layer with factorised Gaussian noise (Fortunato et al.):
//   y = (b + W x) + (b_sigma * e_b + (W_sigma * e_w) x)
// Noise is applied in training mode only and stays fixed until reset_noise();
// eval mode uses the mean weights, so evaluation is deterministic.
struct NoisyLinearImpl : torch::nn::Module {
    torch::Tensor weight, bias;
    torch::Tensor weight_sigma, bias_sigma;
    torch::Tensor eps_in, eps_out;

    NoisyLinearImpl(int64_t n_in, int64_t n_out, double sigma0);

    void reset_noise();
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(NoisyLinear);

struct DQNModelOptions {
    bool   distributional = true;   // false: one scalar Q per action
    int    atoms          = 51;
    float  v_min          = -2.0f;
    float  v_max          = 4.0f;
    bool   noisy          = true;   // false: plain Linear layers
    double sigma0         = 0.5;
};

// Dueling DQN, optionally categorical (C51) and with noisy stream layers.
// The dueling aggregation runs per atom:
//   logit_i(s,a) = V_i(s) + A_i(s,a) − mean_a(A_i(s,a))
//   Q(s,a)       = sum_i z_i * softmax(logit)_i
struct DuelingDQNImpl : torch::nn::Module {
    DQNModelOptions        opts;
    ObsEncoder             encoder{nullptr};
    torch::nn::Sequential  advantage_stream{nullptr};
    torch::nn::Sequential  value_stream{nullptr};
    torch::Tensor          support;  // [atoms], undefined in scalar mode

    explicit DuelingDQNImpl(DQNModelOptions options = {});

    // Atoms per action: opts.atoms, or 1 in scalar mode.
    int num_atoms() const { return opts.distributional ? opts.atoms : 1; }

    // Returns Q-values [B, NUM_ACTIONS]
    torch::Tensor forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat
    );

    // Returns log-probabilities over atoms [B, NUM_ACTIONS, atoms].
    // Only meaningful for a distributional model.
    torch::Tensor log_probs(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat
    );

    // Resamples the noise of every noisy layer; no-op for a non-noisy model.
    void reset_noise();

private:
    torch::Tensor logits(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat
    );
};
TORCH_MODULE(DuelingDQN);

// Projects the distribution (returns + discounts * support, next_probs) back
// onto the fixed support: each atom's mass is split linearly between its two
// neighbouring grid points, after clipping to [v_min, v_max].
//   next_probs: [B, atoms], returns: [B], discounts: [B], support: [atoms]
torch::Tensor categorical_projection(
    const torch::Tensor& next_probs,
    const torch::Tensor& returns,
    const torch::Tensor& discounts,
    const torch::Tensor& support
);

} // namespace pommerman::rl
