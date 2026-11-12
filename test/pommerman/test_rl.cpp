#include <torch/torch.h>
// libtorch's logging header defines its own aborting CHECK; use Catch2's here.
#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include "pommerman/constants.hpp"
#include "pommerman/environment.hpp"
#include "pommerman/rl/dqn_model.hpp"
#include "pommerman/rl/dqn_trainer.hpp"
#include "pommerman/rl/obs_encoder.hpp"
#include "pommerman/rl/replay_buffer.hpp"

using namespace pommerman::rl;
using Catch::Matchers::WithinAbs;

namespace {

Transition make_step(float reward, int action = 0) {
    Transition t;
    t.item_idx      = torch::full({1}, static_cast<int64_t>(reward * 100.0f));
    t.sp_cont       = torch::zeros({1});
    t.ag_feat       = torch::zeros({1});
    t.action        = action;
    t.reward        = reward;
    t.next_item_idx = torch::full({1}, static_cast<int64_t>(reward * 100.0f) + 1);
    t.next_sp_cont  = torch::zeros({1});
    t.next_ag_feat  = torch::zeros({1});
    return t;
}

torch::Tensor uniform_probs(int64_t atoms) {
    return torch::full({int64_t{1}, atoms}, 1.0 / static_cast<double>(atoms), torch::kFloat);
}

float mean_of(const torch::Tensor& probs, const torch::Tensor& support) {
    return (probs * support).sum().item<float>();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_obs(int batch) {
    pommerman::Environment env{pommerman::GameType::FFA, false, pommerman::BOARD_SIZE, 7};
    auto obs = env.reset();
    std::vector<pommerman::Observation> many;
    for (int i = 0; i < batch; ++i) many.push_back(obs[i % pommerman::NUM_AGENTS]);
    return obs_to_tensors(many);
}

} // namespace

// ── SumTree ──────────────────────────────────────────────────────────────────

TEST_CASE("SumTree maps mass to leaves in proportion to priority", "[rl][per]") {
    SumTree tree{4};
    for (int i = 0; i < 4; ++i) tree.update(i, i + 1.0);
    CHECK(tree.total() == 10.0);
    CHECK(tree.find(0.5) == 0);
    CHECK(tree.find(2.5) == 1);
    CHECK(tree.find(5.5) == 2);
    CHECK(tree.find(9.5) == 3);
}

TEST_CASE("SumTree stays correct after a priority changes", "[rl][per]") {
    SumTree tree{4};
    for (int i = 0; i < 4; ++i) tree.update(i, 1.0);
    tree.update(1, 5.0);
    CHECK(tree.total() == 8.0);
    CHECK(tree.find(3.0) == 1);
    CHECK(tree.find(7.5) == 3);
}

TEST_CASE("SumTree works for a capacity that is not a power of two", "[rl][per]") {
    SumTree tree{5};
    std::vector<double> priorities{1.0, 2.0, 3.0, 4.0, 5.0};
    for (int i = 0; i < 5; ++i) tree.update(i, priorities[i]);
    CHECK(tree.total() == 15.0);

    // Every leaf must own a share of the mass range equal to its priority.
    std::vector<int> hits(5, 0);
    const int samples = 15'000;
    for (int s = 0; s < samples; ++s)
        ++hits[tree.find((s + 0.5) * 15.0 / samples)];
    for (int i = 0; i < 5; ++i)
        CHECK(hits[i] == static_cast<int>(priorities[i]) * 1000);
}

TEST_CASE("SumTree never selects an empty leaf", "[rl][per]") {
    SumTree tree{8};
    tree.update(0, 1.0);
    tree.update(1, 1.0);
    CHECK(tree.find(0.0) == 0);
    CHECK(tree.find(1.5) == 1);
    CHECK(tree.find(2.0) == 1);   // top of the range
    CHECK(tree.find(100.0) == 1); // out of range, as rounding could produce
}

// ── NStepQueue ───────────────────────────────────────────────────────────────

TEST_CASE("NStepQueue emits discounted returns once n steps are pending", "[rl][nstep]") {
    NStepQueue q{2, 0.5f};
    CHECK(q.push(make_step(1.0f, 3), false).empty());

    auto ready = q.push(make_step(2.0f, 4), false);
    REQUIRE(ready.size() == 1);
    CHECK(ready[0].action == 3);                  // the action of the first step
    CHECK(ready[0].reward == 1.0f + 0.5f * 2.0f);
    CHECK(ready[0].discount == 0.25f);            // gamma^2
    CHECK(ready[0].item_idx.item<int64_t>() == 100);       // s_t
    CHECK(ready[0].next_item_idx.item<int64_t>() == 201);  // s_{t+2}
    CHECK(q.pending() == 1);
}

TEST_CASE("NStepQueue flushes partial returns on a terminal step", "[rl][nstep]") {
    NStepQueue q{3, 0.5f};
    q.push(make_step(1.0f), false);
    q.push(make_step(2.0f), false);

    auto ready = q.push(make_step(4.0f), true);
    REQUIRE(ready.size() == 3);
    CHECK(ready[0].reward == 1.0f + 0.5f * 2.0f + 0.25f * 4.0f);
    CHECK(ready[1].reward == 2.0f + 0.5f * 4.0f);
    CHECK(ready[2].reward == 4.0f);
    for (const auto& t : ready) {
        CHECK(t.discount == 0.0f);  // terminal: nothing to bootstrap from
        CHECK(t.next_item_idx.item<int64_t>() == 401);
    }
    CHECK(q.pending() == 0);
}

TEST_CASE("NStepQueue with n = 1 is a plain one-step transition", "[rl][nstep]") {
    NStepQueue q{1, 0.9f};
    auto ready = q.push(make_step(2.0f), false);
    REQUIRE(ready.size() == 1);
    CHECK(ready[0].reward == 2.0f);
    CHECK_THAT(ready[0].discount, WithinAbs(0.9, 1e-6));
}

// ── ReplayBuffer ─────────────────────────────────────────────────────────────

TEST_CASE("ReplayBuffer samples in proportion to priority", "[rl][per]") {
    torch::manual_seed(1);
    ReplayBuffer buf{4, /*prioritized=*/true, /*omega=*/1.0f};
    for (int i = 0; i < 4; ++i) buf.add(make_step(0.0f, i));
    buf.update_priorities({0, 1, 2, 3}, {1.0f, 1.0f, 1.0f, 7.0f});

    int top = 0, total = 0;
    for (int s = 0; s < 500; ++s) {
        auto b = buf.sample(4, 1.0, torch::kCPU);
        for (int idx : b.indices) { top += idx == 3; ++total; }
    }
    CHECK_THAT(static_cast<double>(top) / total, WithinAbs(0.7, 0.05));
}

TEST_CASE("ReplayBuffer importance weights offset the sampling bias", "[rl][per]") {
    torch::manual_seed(2);
    ReplayBuffer buf{2, /*prioritized=*/true, /*omega=*/1.0f};
    buf.add(make_step(0.0f, 0));
    buf.add(make_step(0.0f, 1));
    buf.update_priorities({0, 1}, {1.0f, 3.0f});

    // Both draws may land on the likelier slot, so sample until each slot appears.
    auto b = buf.sample(2, 1.0, torch::kCPU);
    for (int tries = 0; tries < 100 && b.indices[0] == b.indices[1]; ++tries)
        b = buf.sample(2, 1.0, torch::kCPU);
    REQUIRE(b.indices[0] != b.indices[1]);
    auto w = b.weights.accessor<float, 1>();
    const float w_low  = b.indices[0] == 0 ? w[0] : w[1];
    const float w_high = b.indices[0] == 0 ? w[1] : w[0];
    CHECK_THAT(w_low, WithinAbs(1.0, 1e-5));          // rarest sample: full weight
    CHECK_THAT(w_high, WithinAbs(1.0 / 3.0, 1e-5));   // three times likelier: a third
}

TEST_CASE("ReplayBuffer gives new transitions the highest priority so far", "[rl][per]") {
    ReplayBuffer buf{3, /*prioritized=*/true, /*omega=*/1.0f};
    buf.add(make_step(0.0f));
    buf.update_priorities({0}, {9.0f});
    buf.add(make_step(0.0f));
    CHECK_THAT(buf.priority(1), WithinAbs(9.0, 1e-3));
}

TEST_CASE("ReplayBuffer without prioritization stays uniform", "[rl][per]") {
    torch::manual_seed(3);
    ReplayBuffer buf{8, /*prioritized=*/false};
    for (int i = 0; i < 8; ++i) buf.add(make_step(0.0f, i));
    buf.update_priorities({0}, {100.0f});
    CHECK(buf.priority(0) == 1.0);

    auto b = buf.sample(8, 0.4, torch::kCPU);
    auto w = b.weights.accessor<float, 1>();
    for (int i = 0; i < 8; ++i) CHECK(w[i] == 1.0f);
    // One draw per equal-mass segment: each slot exactly once.
    auto sorted = b.indices;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 8; ++i) CHECK(sorted[i] == i);
}

TEST_CASE("ReplayBuffer overwrites the oldest slot at capacity", "[rl][per]") {
    ReplayBuffer buf{2, /*prioritized=*/false};
    buf.add(make_step(0.0f, 1));
    buf.add(make_step(0.0f, 2));
    buf.add(make_step(0.0f, 3));
    CHECK(buf.size() == 2);
    auto b = buf.sample(2, 1.0, torch::kCPU);
    auto a = b.actions.accessor<int64_t, 1>();
    CHECK(a[0] + a[1] == 5);  // actions 2 and 3 remain
}

// ── Categorical projection ───────────────────────────────────────────────────

TEST_CASE("Projection of a terminal return lands on the nearest atoms", "[rl][c51]") {
    auto support = torch::linspace(-10.0, 10.0, 51);
    auto p = categorical_projection(uniform_probs(51), torch::tensor({1.3f}),
                                    torch::tensor({0.0f}), support);
    CHECK_THAT(p.sum().item<float>(), WithinAbs(1.0, 1e-5));
    CHECK_THAT(mean_of(p, support), WithinAbs(1.3, 1e-4));
    CHECK((p > 1e-6f).sum().item<int64_t>() == 2);
}

TEST_CASE("Projection of a return exactly on an atom keeps all its mass", "[rl][c51]") {
    auto support = torch::linspace(-2.0, 2.0, 5);  // spacing 1, so grid values are exact
    auto p = categorical_projection(uniform_probs(5), torch::tensor({1.0f}),
                                    torch::tensor({0.0f}), support);
    CHECK_THAT(p[0][3].item<float>(), WithinAbs(1.0, 1e-6));
}

TEST_CASE("Projection preserves the mean inside the support", "[rl][c51]") {
    auto support = torch::linspace(-10.0, 10.0, 51);
    auto probs = torch::softmax(torch::randn({1, 51}), -1);
    auto p = categorical_projection(probs, torch::tensor({0.5f}),
                                    torch::tensor({0.5f}), support);
    const float expected = 0.5f + 0.5f * mean_of(probs, support);
    CHECK_THAT(p.sum().item<float>(), WithinAbs(1.0, 1e-5));
    CHECK_THAT(mean_of(p, support), WithinAbs(expected, 1e-3));
}

TEST_CASE("Projection clips to the support without losing mass", "[rl][c51]") {
    auto support = torch::linspace(-2.0, 2.0, 5);
    auto high = categorical_projection(uniform_probs(5), torch::tensor({25.0f}),
                                       torch::tensor({0.0f}), support);
    CHECK_THAT(high[0][4].item<float>(), WithinAbs(1.0, 1e-6));
    auto low = categorical_projection(uniform_probs(5), torch::tensor({-25.0f}),
                                      torch::tensor({0.0f}), support);
    CHECK_THAT(low[0][0].item<float>(), WithinAbs(1.0, 1e-6));
}

TEST_CASE("Projection matches a worked example", "[rl][c51]") {
    // Five atoms on [-2, 2], next distribution with mean 1, reward 0.1, gamma 0.9.
    auto support = torch::linspace(-2.0, 2.0, 5);
    auto probs = torch::tensor({0.0f, 0.1f, 0.2f, 0.3f, 0.4f}).unsqueeze(0);
    auto p = categorical_projection(probs, torch::tensor({0.1f}),
                                    torch::tensor({0.9f}), support);
    const std::vector<float> expected{0.0f, 0.08f, 0.2f, 0.36f, 0.36f};
    for (int i = 0; i < 5; ++i)
        CHECK_THAT(p[0][i].item<float>(), WithinAbs(expected[i], 1e-5));
}

// ── NoisyLinear ──────────────────────────────────────────────────────────────

TEST_CASE("NoisyLinear output changes only when noise is resampled", "[rl][noisy]") {
    NoisyLinear layer{8, 4, 0.5};
    layer->train();
    auto x = torch::ones({2, 8});
    auto a = layer->forward(x).detach();
    auto b = layer->forward(x).detach();
    CHECK(torch::equal(a, b));
    layer->reset_noise();
    CHECK_FALSE(torch::equal(a, layer->forward(x).detach()));
}

TEST_CASE("NoisyLinear uses the mean weights in eval mode", "[rl][noisy]") {
    NoisyLinear layer{8, 4, 0.5};
    layer->eval();
    auto x = torch::ones({2, 8});
    auto a = layer->forward(x).detach();
    layer->reset_noise();
    CHECK(torch::equal(a, layer->forward(x).detach()));
    auto mean = torch::nn::functional::linear(x, layer->weight, layer->bias).detach();
    CHECK(torch::equal(a, mean));
}

TEST_CASE("NoisyLinear trains both the mean and the noise scale", "[rl][noisy]") {
    NoisyLinear layer{8, 4, 0.5};
    layer->train();
    CHECK(layer->parameters().size() == 4);
    layer->forward(torch::ones({2, 8})).sum().backward();
    CHECK(layer->weight.grad().abs().sum().item<float>() > 0.0f);
    CHECK(layer->weight_sigma.grad().abs().sum().item<float>() > 0.0f);
}

// ── DuelingDQN ───────────────────────────────────────────────────────────────

TEST_CASE("Distributional DuelingDQN outputs valid distributions and their means", "[rl][model]") {
    DuelingDQN net{DQNModelOptions{}};
    net->eval();
    torch::NoGradGuard ng;
    auto [it, sp, ag] = sample_obs(3);

    auto lp = net->log_probs(it, sp, ag);
    CHECK(lp.sizes() == std::vector<int64_t>{3, pommerman::rl::NUM_ACTIONS, 51});
    auto sums = lp.exp().sum(-1);
    CHECK((sums - 1.0f).abs().max().item<float>() < 1e-4f);

    auto q = net->forward(it, sp, ag);
    CHECK(q.sizes() == std::vector<int64_t>{3, pommerman::rl::NUM_ACTIONS});
    auto expected = (lp.exp() * net->support).sum(-1);
    CHECK((q - expected).abs().max().item<float>() < 1e-4f);
}

TEST_CASE("Scalar plain DuelingDQN outputs one Q per action", "[rl][model]") {
    DQNModelOptions opts;
    opts.distributional = false;
    opts.noisy = false;
    DuelingDQN net{opts};
    net->train();
    torch::NoGradGuard ng;
    auto [it, sp, ag] = sample_obs(3);
    auto q = net->forward(it, sp, ag);
    CHECK(q.sizes() == std::vector<int64_t>{3, pommerman::rl::NUM_ACTIONS});
    net->reset_noise();  // no-op
    CHECK(torch::equal(q, net->forward(it, sp, ag)));
}

TEST_CASE("Noisy DuelingDQN reaches every noisy layer on reset", "[rl][model]") {
    DuelingDQN net{DQNModelOptions{}};
    net->train();
    torch::NoGradGuard ng;
    auto [it, sp, ag] = sample_obs(1);
    auto a = net->forward(it, sp, ag);
    CHECK(torch::equal(a, net->forward(it, sp, ag)));

    int noisy_layers = 0;
    std::vector<torch::Tensor> before;
    for (auto& m : net->modules(false)) {
        if (auto* n = m->as<NoisyLinearImpl>()) {
            ++noisy_layers;
            before.push_back(n->eps_in.clone());
        }
    }
    CHECK(noisy_layers == 4);

    net->reset_noise();
    CHECK_FALSE(torch::equal(a, net->forward(it, sp, ag)));
    int i = 0;
    for (auto& m : net->modules(false)) {
        if (auto* n = m->as<NoisyLinearImpl>())
            CHECK_FALSE(torch::equal(before[i++], n->eps_in));
    }
}

// ── DQNTrainer ───────────────────────────────────────────────────────────────

namespace {

void run_trainer_smoke(bool distributional, bool noisy, bool prioritized, int n_step) {
    torch::manual_seed(11);
    DQNModelOptions opts;
    opts.distributional = distributional;
    opts.noisy = noisy;

    DuelingDQN online{opts}, target{opts};
    DQNConfig cfg;
    cfg.replay_capacity = 64;
    cfg.batch_size = 4;
    cfg.warmup_steps = 8;
    cfg.target_update_freq = 8;
    cfg.train_freq = 2;
    cfg.n_step = n_step;
    cfg.prioritized = prioritized;
    cfg.total_steps = 30;
    DQNTrainer trainer{online, target, cfg, torch::kCPU};

    auto before = online->named_parameters().find("encoder.spatial_proj.weight")->clone();

    pommerman::Environment env{pommerman::GameType::FFA, false, pommerman::BOARD_SIZE, 5};
    int updates = 0;
    DQNTrainer::Stats s{};
    for (int i = 0; i < 30; ++i) {
        s = trainer.step(env);
        if (std::isnan(s.loss)) continue;
        ++updates;
        CHECK(std::isfinite(s.q_mean));
        CHECK(s.loss >= 0.0f);
        CHECK(s.edge_mass >= 0.0f);
    }
    CHECK(updates > 0);
    CHECK(s.epsilon == (noisy ? 0.0f : s.epsilon));
    CHECK_FALSE(torch::equal(
        before, *online->named_parameters().find("encoder.spatial_proj.weight")));

    // Evaluation-style use must not break the next training step.
    online->eval();
    trainer.step(env);
    CHECK(online->is_training());
}

} // namespace

TEST_CASE("DQNTrainer updates under each component combination", "[rl][trainer]") {
    SECTION("full Rainbow")     { run_trainer_smoke(true,  true,  true,  3); }
    SECTION("no noisy nets")    { run_trainer_smoke(true,  false, true,  3); }
    SECTION("no prioritization"){ run_trainer_smoke(true,  true,  false, 3); }
    SECTION("no multi-step")    { run_trainer_smoke(true,  true,  true,  1); }
    SECTION("no distributional"){ run_trainer_smoke(false, true,  true,  3); }
    SECTION("double dueling only") { run_trainer_smoke(false, false, false, 1); }
}

TEST_CASE("DQNTrainer anneals beta to 1 over the configured steps", "[rl][trainer]") {
    DQNModelOptions opts;
    opts.noisy = false;
    DuelingDQN online{opts}, target{opts};
    DQNConfig cfg;
    cfg.replay_capacity = 64;
    cfg.warmup_steps = 1'000;
    cfg.total_steps = 100;
    cfg.per_beta_start = 0.4f;
    DQNTrainer trainer{online, target, cfg, torch::kCPU};
    pommerman::Environment env{pommerman::GameType::FFA, false, pommerman::BOARD_SIZE, 5};

    trainer.set_global_step(49);
    CHECK_THAT(trainer.step(env).beta, WithinAbs(0.7, 1e-5));
    trainer.set_global_step(500);
    CHECK_THAT(trainer.step(env).beta, WithinAbs(1.0, 1e-5));
}

TEST_CASE("DQN model checkpoints round-trip the distributional noisy network", "[rl][trainer]") {
    DuelingDQN a{DQNModelOptions{}};
    DuelingDQN b{DQNModelOptions{}};
    const std::string path = "test_rl_checkpoint.pt";
    torch::save(a, path);
    torch::load(b, path);
    std::remove(path.c_str());

    a->eval();
    b->eval();
    torch::NoGradGuard ng;
    auto [it, sp, ag] = sample_obs(2);
    CHECK(torch::equal(a->forward(it, sp, ag), b->forward(it, sp, ag)));
}
