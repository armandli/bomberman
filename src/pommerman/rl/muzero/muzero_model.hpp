#pragma once
#include <torch/torch.h>
#include "pommerman/rl/obs_encoder.hpp"
#include <utility>

namespace pommerman::rl::muzero {

constexpr int D_ACTION = 32;

// ── Dynamics: (hidden, action) → (next_hidden, reward) ───────────────────────
struct DynamicsImpl : torch::nn::Module {
    torch::nn::Embedding action_embed{nullptr};
    torch::nn::Linear    input_proj{nullptr};
    torch::nn::Linear    res1a{nullptr}, res1b{nullptr};
    torch::nn::Linear    res2a{nullptr}, res2b{nullptr};
    torch::nn::LayerNorm ln_in{nullptr};
    torch::nn::LayerNorm ln_1a{nullptr}, ln_1b{nullptr};
    torch::nn::LayerNorm ln_2a{nullptr}, ln_2b{nullptr};
    torch::nn::LayerNorm out_norm{nullptr};
    torch::nn::Linear    reward_head{nullptr};

    DynamicsImpl();

    // Returns (next_hidden [B, D_MODEL], reward [B])
    std::pair<torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& hidden,       // [B, D_MODEL]
            const torch::Tensor& action_idx);  // [B] int64
};
TORCH_MODULE(Dynamics);

// ── Prediction: hidden → (policy_logits, value) ───────────────────────────────
struct PredictionImpl : torch::nn::Module {
    torch::nn::Linear policy_head{nullptr};
    torch::nn::Linear value_head{nullptr};

    PredictionImpl();

    // Returns (policy_logits [B, NUM_ACTIONS], value [B])
    std::pair<torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& hidden);  // [B, D_MODEL]
};
TORCH_MODULE(Prediction);

// ── SimSiam projector/predictor (EfficientZero consistency loss) ──────────────
struct SimSiamImpl : torch::nn::Module {
    torch::nn::Sequential projector{nullptr};
    torch::nn::Sequential predictor{nullptr};

    SimSiamImpl();

    // Returns projection (for target branch with sg) [B, D_MODEL]
    torch::Tensor project(const torch::Tensor& hidden);

    // Returns prediction on top of projection [B, D_MODEL]
    torch::Tensor predict_proj(const torch::Tensor& proj);

    // Full consistency loss: -cosine_sim(pred(proj(h_dyn)), sg(proj(h_repr)))
    torch::Tensor consistency_loss(const torch::Tensor& h_dynamics,
                                   const torch::Tensor& h_repr);
};
TORCH_MODULE(SimSiam);

// ── Full MuZero model ─────────────────────────────────────────────────────────
struct MuZeroModelImpl : torch::nn::Module {
    ObsEncoder representation{nullptr};
    Dynamics   dynamics{nullptr};
    Prediction prediction{nullptr};
    SimSiam    simsiam{nullptr};

    MuZeroModelImpl();

    // Encode observation to hidden state
    torch::Tensor represent(const torch::Tensor& item_idx,
                            const torch::Tensor& sp_cont,
                            const torch::Tensor& ag_feat);

    // Dynamics step: (hidden, action) → (next_hidden, reward)
    std::pair<torch::Tensor, torch::Tensor>
    step(const torch::Tensor& hidden,
         const torch::Tensor& action_idx);

    // Policy + value from hidden
    std::pair<torch::Tensor, torch::Tensor>
    predict(const torch::Tensor& hidden);
};
TORCH_MODULE(MuZeroModel);

} // namespace pommerman::rl::muzero
