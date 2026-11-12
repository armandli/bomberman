#include "pommerman/rl/replay_buffer.hpp"
#include <cmath>
#include <stdexcept>

namespace pommerman::rl {

namespace {
constexpr double PRIORITY_EPS = 1e-6;  // keeps zero-error transitions sampleable
}

// ── SumTree ──────────────────────────────────────────────────────────────────

SumTree::SumTree(int capacity) : capacity_(capacity), tree_(2 * capacity, 0.0) {}

void SumTree::update(int idx, double priority) {
    int node = idx + capacity_;
    tree_[node] = priority;
    for (node /= 2; node >= 1; node /= 2) {
        // Recomputed from the children, so rounding error cannot accumulate.
        tree_[node] = tree_[2 * node] + tree_[2 * node + 1];
    }
}

int SumTree::find(double mass) const {
    int node = 1;
    while (node < capacity_) {
        int left = 2 * node;
        if (mass < tree_[left] || tree_[left + 1] <= 0.0) {
            node = left;
        } else {
            mass -= tree_[left];
            node = left + 1;
        }
    }
    return node - capacity_;
}

// ── NStepQueue ───────────────────────────────────────────────────────────────

NStepQueue::NStepQueue(int n, float gamma) : n_(n), gamma_(gamma) {
    if (n < 1) throw std::invalid_argument("NStepQueue: n must be >= 1");
}

Transition NStepQueue::make_front(const Transition& last, bool terminal) const {
    Transition out = pending_.front();
    float ret  = 0.0f;
    float disc = 1.0f;
    for (const auto& step : pending_) {
        ret  += disc * step.reward;
        disc *= gamma_;
    }
    out.reward        = ret;
    out.next_item_idx = last.next_item_idx;
    out.next_sp_cont  = last.next_sp_cont;
    out.next_ag_feat  = last.next_ag_feat;
    out.discount      = terminal ? 0.0f : disc;
    return out;
}

std::vector<Transition> NStepQueue::push(Transition step, bool terminal) {
    pending_.push_back(std::move(step));
    const Transition last = pending_.back();

    std::vector<Transition> ready;
    if (terminal) {
        while (!pending_.empty()) {
            ready.push_back(make_front(last, true));
            pending_.pop_front();
        }
    } else if (static_cast<int>(pending_.size()) == n_) {
        ready.push_back(make_front(last, false));
        pending_.pop_front();
    }
    return ready;
}

// ── ReplayBuffer ─────────────────────────────────────────────────────────────

ReplayBuffer::ReplayBuffer(int capacity, bool prioritized, float omega)
    : capacity_(capacity), prioritized_(prioritized), omega_(omega), tree_(capacity) {
    item_idxs_.resize(capacity);
    sp_conts_.resize(capacity);
    ag_feats_.resize(capacity);
    actions_.resize(capacity, 0);
    rewards_.resize(capacity, 0.0f);
    next_item_idxs_.resize(capacity);
    next_sp_conts_.resize(capacity);
    next_ag_feats_.resize(capacity);
    discounts_.resize(capacity, 0.0f);
}

void ReplayBuffer::add(const Transition& t) {
    // Clone to own the storage (inputs may be views/slices)
    item_idxs_[pos_]      = t.item_idx.clone();
    sp_conts_[pos_]       = t.sp_cont.clone();
    ag_feats_[pos_]       = t.ag_feat.clone();
    actions_[pos_]        = t.action;
    rewards_[pos_]        = t.reward;
    next_item_idxs_[pos_] = t.next_item_idx.clone();
    next_sp_conts_[pos_]  = t.next_sp_cont.clone();
    next_ag_feats_[pos_]  = t.next_ag_feat.clone();
    discounts_[pos_]      = t.discount;

    tree_.update(pos_, max_priority_);

    pos_ = (pos_ + 1) % capacity_;
    ++count_;
}

ReplayBatch ReplayBuffer::sample(int batch_size, double beta, torch::Device device) {
    const int n = size();
    if (n < batch_size)
        throw std::runtime_error("ReplayBuffer: not enough samples");

    const double total   = tree_.total();
    const double segment = total / batch_size;
    auto draws     = torch::rand({batch_size}, torch::kDouble);
    auto draws_acc = draws.accessor<double, 1>();

    ReplayBatch batch;
    batch.indices.reserve(batch_size);
    std::vector<float> weights;
    weights.reserve(batch_size);
    float max_weight = 0.0f;

    std::vector<torch::Tensor> b_item, b_sp, b_ag;
    std::vector<torch::Tensor> b_ni,   b_ns, b_na;
    std::vector<int64_t>       b_acts;
    std::vector<float>         b_rew, b_disc;
    b_item.reserve(batch_size); b_sp.reserve(batch_size); b_ag.reserve(batch_size);
    b_ni.reserve(batch_size);   b_ns.reserve(batch_size); b_na.reserve(batch_size);
    b_acts.reserve(batch_size);
    b_rew.reserve(batch_size);  b_disc.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const int idx = tree_.find((i + draws_acc[i]) * segment);
        batch.indices.push_back(idx);

        const double prob = tree_.priority(idx) / total;
        const float weight = static_cast<float>(std::pow(n * prob, -beta));
        weights.push_back(weight);
        max_weight = std::max(max_weight, weight);

        b_item.push_back(item_idxs_[idx]);
        b_sp.push_back(sp_conts_[idx]);
        b_ag.push_back(ag_feats_[idx]);
        b_ni.push_back(next_item_idxs_[idx]);
        b_ns.push_back(next_sp_conts_[idx]);
        b_na.push_back(next_ag_feats_[idx]);
        b_acts.push_back(static_cast<int64_t>(actions_[idx]));
        b_rew.push_back(rewards_[idx]);
        b_disc.push_back(discounts_[idx]);
    }

    // Scale so that the correction only ever shrinks updates.
    for (auto& w : weights) w /= max_weight;

    batch.item_idx      = torch::stack(b_item, 0).to(device);
    batch.sp_cont       = torch::stack(b_sp,   0).to(device);
    batch.ag_feat       = torch::stack(b_ag,   0).to(device);
    batch.actions       = torch::tensor(b_acts, torch::TensorOptions().dtype(torch::kLong)).to(device);
    batch.rewards       = torch::tensor(b_rew,  torch::kFloat).to(device);
    batch.next_item_idx = torch::stack(b_ni,   0).to(device);
    batch.next_sp_cont  = torch::stack(b_ns,   0).to(device);
    batch.next_ag_feat  = torch::stack(b_na,   0).to(device);
    batch.discounts     = torch::tensor(b_disc, torch::kFloat).to(device);
    batch.weights       = torch::tensor(weights, torch::kFloat).to(device);
    return batch;
}

void ReplayBuffer::update_priorities(const std::vector<int>& indices,
                                     const std::vector<float>& errors) {
    if (!prioritized_) return;
    for (size_t i = 0; i < indices.size(); ++i) {
        const double p = std::pow(static_cast<double>(errors[i]) + PRIORITY_EPS,
                                  static_cast<double>(omega_));
        max_priority_ = std::max(max_priority_, p);
        tree_.update(indices[i], p);
    }
}

} // namespace pommerman::rl
