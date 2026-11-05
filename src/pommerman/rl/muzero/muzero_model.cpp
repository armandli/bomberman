#include "pommerman/rl/muzero/muzero_model.hpp"
#include "pommerman/rl/obs_encoder.hpp"

namespace pommerman::rl::muzero {

// ── Dynamics ──────────────────────────────────────────────────────────────────

DynamicsImpl::DynamicsImpl() {
    action_embed = register_module("action_embed",
                       torch::nn::Embedding(NUM_ACTIONS, D_ACTION));
    input_proj   = register_module("input_proj",
                       torch::nn::Linear(D_MODEL + D_ACTION, D_MODEL));
    res1a = register_module("res1a", torch::nn::Linear(D_MODEL, D_MODEL));
    res1b = register_module("res1b", torch::nn::Linear(D_MODEL, D_MODEL));
    res2a = register_module("res2a", torch::nn::Linear(D_MODEL, D_MODEL));
    res2b = register_module("res2b", torch::nn::Linear(D_MODEL, D_MODEL));

    ln_in  = register_module("ln_in",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
    ln_1a  = register_module("ln_1a",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
    ln_1b  = register_module("ln_1b",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
    ln_2a  = register_module("ln_2a",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
    ln_2b  = register_module("ln_2b",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
    out_norm = register_module("out_norm",
                   torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
    reward_head = register_module("reward_head",
                      torch::nn::Linear(D_MODEL, 1));
}

std::pair<torch::Tensor, torch::Tensor>
DynamicsImpl::forward(const torch::Tensor& hidden, const torch::Tensor& action_idx) {
    auto a_emb = action_embed(action_idx);             // [B, D_ACTION]
    auto x     = torch::cat({hidden, a_emb}, /*dim=*/1); // [B, D_MODEL+D_ACTION]
    x = torch::relu(ln_in(input_proj(x)));             // [B, D_MODEL]

    // Residual block 1
    auto res = x;
    x = torch::relu(ln_1a(res1a(x)));
    x = ln_1b(res1b(x)) + res;
    x = torch::relu(x);

    // Residual block 2
    res = x;
    x = torch::relu(ln_2a(res2a(x)));
    x = ln_2b(res2b(x)) + res;
    x = torch::relu(x);

    auto h_next = out_norm(x);
    auto reward  = reward_head(h_next).squeeze(-1);  // [B]
    return {h_next, reward};
}

// ── Prediction ────────────────────────────────────────────────────────────────

PredictionImpl::PredictionImpl() {
    policy_head = register_module("policy_head",
                      torch::nn::Linear(D_MODEL, NUM_ACTIONS));
    value_head  = register_module("value_head",
                      torch::nn::Linear(D_MODEL, 1));

    torch::nn::init::orthogonal_(policy_head->weight, 0.01);
    torch::nn::init::constant_(policy_head->bias,     0.0);
    torch::nn::init::orthogonal_(value_head->weight,  1.0);
    torch::nn::init::constant_(value_head->bias,      0.0);
}

std::pair<torch::Tensor, torch::Tensor>
PredictionImpl::forward(const torch::Tensor& hidden) {
    return {policy_head(hidden), value_head(hidden).squeeze(-1)};
}

// ── SimSiam ───────────────────────────────────────────────────────────────────

SimSiamImpl::SimSiamImpl() {
    projector = register_module("projector",
        torch::nn::Sequential(
            torch::nn::Linear(D_MODEL, D_MODEL),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})),
            torch::nn::ReLU(),
            torch::nn::Linear(D_MODEL, D_MODEL),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})),
            torch::nn::ReLU(),
            torch::nn::Linear(D_MODEL, D_MODEL)
        ));
    predictor = register_module("predictor",
        torch::nn::Sequential(
            torch::nn::Linear(D_MODEL, D_MODEL),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})),
            torch::nn::ReLU(),
            torch::nn::Linear(D_MODEL, D_MODEL)
        ));
}

torch::Tensor SimSiamImpl::project(const torch::Tensor& hidden) {
    return projector->forward(hidden);
}

torch::Tensor SimSiamImpl::predict_proj(const torch::Tensor& proj) {
    return predictor->forward(proj);
}

torch::Tensor SimSiamImpl::consistency_loss(const torch::Tensor& h_dynamics,
                                             const torch::Tensor& h_repr) {
    // Dynamics branch: project + predict
    auto z_dyn = project(h_dynamics);
    auto p_dyn = predict_proj(z_dyn);

    // Representation branch: project only, stop gradient
    auto z_repr = project(h_repr).detach();

    // Negative cosine similarity (mean over batch)
    return -torch::nn::functional::cosine_similarity(
        p_dyn, z_repr,
        torch::nn::functional::CosineSimilarityFuncOptions().dim(1)
    ).mean();
}

// ── MuZeroModel ───────────────────────────────────────────────────────────────

MuZeroModelImpl::MuZeroModelImpl() {
    representation = register_module("representation", ObsEncoder());
    dynamics       = register_module("dynamics",       Dynamics());
    prediction     = register_module("prediction",     Prediction());
    simsiam        = register_module("simsiam",        SimSiam());
}

torch::Tensor MuZeroModelImpl::represent(const torch::Tensor& item_idx,
                                          const torch::Tensor& sp_cont,
                                          const torch::Tensor& ag_feat) {
    return representation->forward(item_idx, sp_cont, ag_feat);
}

std::pair<torch::Tensor, torch::Tensor>
MuZeroModelImpl::step(const torch::Tensor& hidden, const torch::Tensor& action_idx) {
    return dynamics->forward(hidden, action_idx);
}

std::pair<torch::Tensor, torch::Tensor>
MuZeroModelImpl::predict(const torch::Tensor& hidden) {
    return prediction->forward(hidden);
}

} // namespace pommerman::rl::muzero
