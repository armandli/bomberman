#include "pommerman/rl/dqn_model.hpp"
#include <cmath>
#include <stdexcept>

namespace pommerman::rl {

// ── NoisyLinear ──────────────────────────────────────────────────────────────

NoisyLinearImpl::NoisyLinearImpl(int64_t n_in, int64_t n_out, double sigma0) {
    const double bound = 1.0 / std::sqrt(static_cast<double>(n_in));
    const double sigma = sigma0 * bound;

    weight       = register_parameter("weight", torch::empty({n_out, n_in}));
    bias         = register_parameter("bias",   torch::empty({n_out}));
    weight_sigma = register_parameter("weight_sigma", torch::full({n_out, n_in}, sigma));
    bias_sigma   = register_parameter("bias_sigma",   torch::full({n_out}, sigma));
    eps_in       = register_buffer("eps_in",  torch::zeros({n_in}));
    eps_out      = register_buffer("eps_out", torch::zeros({n_out}));

    torch::nn::init::uniform_(weight, -bound, bound);
    torch::nn::init::uniform_(bias,   -bound, bound);
    reset_noise();
}

void NoisyLinearImpl::reset_noise() {
    torch::NoGradGuard ng;
    auto scale = [](const torch::Tensor& x) { return x.sign() * x.abs().sqrt(); };
    eps_in.copy_(scale(torch::randn_like(eps_in)));
    eps_out.copy_(scale(torch::randn_like(eps_out)));
}

torch::Tensor NoisyLinearImpl::forward(const torch::Tensor& x) {
    if (!is_training())
        return torch::nn::functional::linear(x, weight, bias);
    auto eps_w = torch::outer(eps_out, eps_in);
    return torch::nn::functional::linear(
        x, weight + weight_sigma * eps_w, bias + bias_sigma * eps_out);
}

// ── DuelingDQN ───────────────────────────────────────────────────────────────

namespace {

torch::nn::Sequential make_stream(const DQNModelOptions& opts, int64_t n_out) {
    torch::nn::Sequential stream;
    if (opts.noisy) {
        stream->push_back(NoisyLinear(D_MODEL, D_MODEL, opts.sigma0));
        stream->push_back(torch::nn::ReLU());
        stream->push_back(NoisyLinear(D_MODEL, n_out, opts.sigma0));
    } else {
        stream->push_back(torch::nn::Linear(D_MODEL, D_MODEL));
        stream->push_back(torch::nn::ReLU());
        stream->push_back(torch::nn::Linear(D_MODEL, n_out));
    }
    return stream;
}

} // namespace

DuelingDQNImpl::DuelingDQNImpl(DQNModelOptions options) : opts(options) {
    if (opts.distributional && (opts.atoms < 2 || opts.v_max <= opts.v_min))
        throw std::invalid_argument("DuelingDQN: need atoms >= 2 and v_max > v_min");

    encoder = register_module("encoder", ObsEncoder());

    const int64_t atoms = num_atoms();
    advantage_stream = register_module("advantage_stream",
        make_stream(opts, NUM_ACTIONS * atoms));
    value_stream = register_module("value_stream",
        make_stream(opts, atoms));

    if (opts.distributional)
        support = register_buffer("support",
            torch::linspace(opts.v_min, opts.v_max, opts.atoms));
}

torch::Tensor DuelingDQNImpl::logits(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {
    const int64_t atoms = num_atoms();
    auto features  = encoder->forward(item_idx, sp_cont, ag_feat);
    auto advantage = advantage_stream->forward(features).view({-1, NUM_ACTIONS, atoms});
    auto value     = value_stream->forward(features).view({-1, 1, atoms});

    // Dueling aggregation, per atom: [B, NUM_ACTIONS, atoms]
    return value + advantage - advantage.mean(/*dim=*/1, /*keepdim=*/true);
}

torch::Tensor DuelingDQNImpl::forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {
    auto out = logits(item_idx, sp_cont, ag_feat);
    if (!opts.distributional) return out.squeeze(-1);
    return (torch::softmax(out, -1) * support).sum(-1);
}

torch::Tensor DuelingDQNImpl::log_probs(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {
    return torch::log_softmax(logits(item_idx, sp_cont, ag_feat), -1);
}

void DuelingDQNImpl::reset_noise() {
    if (!opts.noisy) return;
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* noisy = module->as<NoisyLinearImpl>())
            noisy->reset_noise();
    }
}

// ── Categorical projection ───────────────────────────────────────────────────

torch::Tensor categorical_projection(
        const torch::Tensor& next_probs,
        const torch::Tensor& returns,
        const torch::Tensor& discounts,
        const torch::Tensor& support) {
    const int64_t atoms = support.numel();
    const double v_min = support[0].item<double>();
    const double v_max = support[atoms - 1].item<double>();
    const double delta = (v_max - v_min) / static_cast<double>(atoms - 1);

    auto tz = (returns.unsqueeze(1) + discounts.unsqueeze(1) * support.unsqueeze(0))
                  .clamp(v_min, v_max);
    auto b  = ((tz - v_min) / delta).clamp(0.0, static_cast<double>(atoms - 1));

    // A value exactly on a grid point gets upper_w = 0, so all of its mass stays
    // on `lower`; clamping `upper` keeps the top atom in range without losing mass.
    auto lower   = b.floor().to(torch::kLong);
    auto upper   = (lower + 1).clamp_max(atoms - 1);
    auto upper_w = b - lower.to(b.dtype());

    auto target = torch::zeros_like(next_probs);
    target.scatter_add_(1, lower, next_probs * (1.0f - upper_w));
    target.scatter_add_(1, upper, next_probs * upper_w);
    return target;
}

} // namespace pommerman::rl
