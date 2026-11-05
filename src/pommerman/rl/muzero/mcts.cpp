#include "pommerman/rl/muzero/mcts.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace pommerman::rl::muzero {

// ── MinMaxStats ───────────────────────────────────────────────────────────────

void MinMaxStats::update(float val) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
}

float MinMaxStats::normalize(float val) const {
    if (max_val > min_val)
        return (val - min_val) / (max_val - min_val);
    return val;
}

// ── MCTSNode ──────────────────────────────────────────────────────────────────

MCTSNode::MCTSNode(torch::Tensor h, float p, float r)
    : hidden(std::move(h)), reward(r), prior(p) {}

float MCTSNode::value() const {
    if (visit_count == 0) return 0.f;
    return value_sum / static_cast<float>(visit_count);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static void expand_node(MCTSNode& node, MuZeroModel& model) {
    // prediction: policy_logits [1,6], value [1]
    auto [logits, value_t] = model->predict(node.hidden);
    auto priors = torch::softmax(logits, /*dim=*/1); // [1,6]
    float v = value_t.item<float>();

    // Store value_sum so it contributes on first backup
    node.value_sum = v;
    node.visit_count = 1;
    node.expanded = true;

    for (int a = 0; a < NUM_ACTIONS; ++a) {
        float p = priors[0][a].item<float>();
        // next_hidden, reward from dynamics
        auto action_t = torch::tensor(a, torch::TensorOptions().dtype(torch::kLong))
                            .unsqueeze(0)
                            .to(node.hidden.device());  // [1]
        auto [next_h, reward_t] = model->step(node.hidden, action_t);
        float r = reward_t.item<float>();
        node.children[a] = std::make_unique<MCTSNode>(next_h, p, r);
    }
}

// PUCT score for child at action `a`
static float puct_score(const MCTSNode& parent,
                         const MCTSNode& child,
                         const MinMaxStats& stats,
                         const MuZeroConfig& cfg) {
    float n_parent = static_cast<float>(parent.visit_count);
    float n_child  = static_cast<float>(child.visit_count);
    float prior    = child.prior;

    // Exploration term
    float c = std::log((n_parent + cfg.c_base + 1.f) / cfg.c_base) + cfg.c_init;
    float u = c * prior * std::sqrt(n_parent) / (1.f + n_child);

    // Q normalised to [0,1]
    float q = (child.visit_count > 0)
                  ? stats.normalize(child.reward + cfg.discount * child.value())
                  : 0.f;

    return q + u;
}

// Select from root to a leaf, return path of raw pointers
static std::vector<MCTSNode*>
select_path(MCTSNode& root, const MinMaxStats& stats, const MuZeroConfig& cfg) {
    std::vector<MCTSNode*> path;
    MCTSNode* node = &root;
    path.push_back(node);

    while (node->expanded) {
        int best_a = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            if (!node->children[a]) continue;
            float s = puct_score(*node, *node->children[a], stats, cfg);
            if (s > best_score) {
                best_score = s;
                best_a = a;
            }
        }
        node = node->children[best_a].get();
        path.push_back(node);
    }
    return path;
}

// Back-propagate value up the path
static void backup(const std::vector<MCTSNode*>& path,
                   float leaf_value,
                   const MuZeroConfig& cfg,
                   MinMaxStats& stats) {
    float value = leaf_value;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        MCTSNode* node = path[i];
        node->value_sum  += value;
        node->visit_count += 1;
        stats.update(node->reward + cfg.discount * node->value());
        value = node->reward + cfg.discount * value;
    }
}

// Add Dirichlet noise to root priors
static void add_dirichlet(MCTSNode& root,
                           float alpha,
                           float epsilon,
                           std::mt19937& rng) {
    // Sample symmetric Dirichlet via Gamma
    std::gamma_distribution<float> gamma_dist(alpha, 1.0f);
    std::array<float, NUM_ACTIONS> dir{};
    float sum = 0.f;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        dir[a] = gamma_dist(rng);
        sum += dir[a];
    }
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        if (root.children[a]) {
            root.children[a]->prior =
                (1.f - epsilon) * root.children[a]->prior + epsilon * dir[a] / sum;
        }
    }
}

// ── run_mcts ──────────────────────────────────────────────────────────────────

std::pair<torch::Tensor, float>
run_mcts(const torch::Tensor& root_hidden,
         MuZeroModel&          model,
         const MuZeroConfig&   cfg,
         torch::Device         device) {
    torch::NoGradGuard no_grad;
    model->eval();

    // Create and expand root
    auto root = std::make_unique<MCTSNode>(root_hidden.to(device));
    expand_node(*root, model);

    // Add Dirichlet noise at root for exploration
    static thread_local std::mt19937 rng{std::random_device{}()};
    add_dirichlet(*root, cfg.dirichlet_alpha, cfg.dirichlet_epsilon, rng);

    MinMaxStats stats;
    // Initialise stats from root's children
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        if (root->children[a]) {
            stats.update(root->children[a]->reward);
        }
    }

    for (int sim = 0; sim < cfg.num_simulations; ++sim) {
        auto path = select_path(*root, stats, cfg);
        MCTSNode* leaf = path.back();

        // Expand leaf via dynamics + prediction
        expand_node(*leaf, model);

        float leaf_value = leaf->value();
        backup(path, leaf_value, cfg, stats);
    }

    // Build visit-count policy
    auto policy_t = torch::zeros({NUM_ACTIONS}, torch::kFloat);
    int total_visits = 0;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        if (root->children[a]) {
            int n = root->children[a]->visit_count;
            policy_t[a] = static_cast<float>(n);
            total_visits += n;
        }
    }
    if (total_visits > 0)
        policy_t = policy_t / static_cast<float>(total_visits);

    return {policy_t, root->value()};
}

} // namespace pommerman::rl::muzero
