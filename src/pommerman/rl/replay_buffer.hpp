#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <deque>
#include <vector>

namespace pommerman::rl {

// One stored transition. `reward` is the n-step return R^(n) and `discount` is
// the factor applied to the bootstrap value: gamma^k for a return of k steps,
// or 0 when the transition reached a terminal state.
struct Transition {
    torch::Tensor item_idx;       // [121] int64
    torch::Tensor sp_cont;        // [121, SPATIAL_CONT_DIM] float
    torch::Tensor ag_feat;        // [AGENT_FEAT_DIM] float
    int           action = 0;
    float         reward = 0.0f;
    torch::Tensor next_item_idx;
    torch::Tensor next_sp_cont;
    torch::Tensor next_ag_feat;
    float         discount = 0.0f;
};

// Binary tree of priority sums: O(log N) update and proportional lookup.
// Works for any capacity >= 1; leaves live at [capacity, 2 * capacity).
class SumTree {
public:
    explicit SumTree(int capacity);

    void update(int idx, double priority);

    // Index of the leaf whose cumulative priority range contains `mass`.
    // Never returns a zero-priority leaf while total() > 0, even if `mass` is
    // pushed out of range by rounding.
    int find(double mass) const;

    double total() const { return tree_[1]; }
    double priority(int idx) const { return tree_[idx + capacity_]; }

private:
    int                 capacity_;
    std::vector<double> tree_;
};

// Builds n-step transitions for one agent. Push single steps in order; each
// call returns the transitions that became complete. A terminal step flushes
// every pending step as a shorter return with discount 0.
class NStepQueue {
public:
    NStepQueue(int n, float gamma);

    std::vector<Transition> push(Transition step, bool terminal);

    int pending() const { return static_cast<int>(pending_.size()); }

private:
    Transition make_front(const Transition& last, bool terminal) const;

    int                    n_;
    float                  gamma_;
    std::deque<Transition> pending_;
};

struct ReplayBatch {
    std::vector<int> indices;    // buffer slots, for update_priorities
    torch::Tensor    item_idx;
    torch::Tensor    sp_cont;
    torch::Tensor    ag_feat;
    torch::Tensor    actions;
    torch::Tensor    rewards;    // n-step returns
    torch::Tensor    next_item_idx;
    torch::Tensor    next_sp_cont;
    torch::Tensor    next_ag_feat;
    torch::Tensor    discounts;  // bootstrap factor per sample
    torch::Tensor    weights;    // importance-sampling weights, max 1
};

// Circular replay buffer with proportional prioritization (CPU storage, moved
// to the device on sample). With prioritized = false every slot keeps priority
// 1, so sampling is uniform and all importance weights are 1.
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity, bool prioritized = true, float omega = 0.5f);

    // New transitions enter at the maximum priority seen so far.
    void add(const Transition& t);

    // Stratified proportional sample: one draw per equal-mass segment.
    ReplayBatch sample(int batch_size, double beta, torch::Device device);

    // Sets each slot's priority to (error + small constant)^omega.
    void update_priorities(const std::vector<int>& indices,
                           const std::vector<float>& errors);

    int size()     const { return std::min(count_, capacity_); }
    int capacity() const { return capacity_; }

    double priority(int idx) const { return tree_.priority(idx); }

private:
    int     capacity_;
    bool    prioritized_;
    float   omega_;
    int     pos_   = 0;
    int     count_ = 0;
    double  max_priority_ = 1.0;
    SumTree tree_;

    std::vector<torch::Tensor> item_idxs_;
    std::vector<torch::Tensor> sp_conts_;
    std::vector<torch::Tensor> ag_feats_;
    std::vector<int>           actions_;
    std::vector<float>         rewards_;
    std::vector<torch::Tensor> next_item_idxs_;
    std::vector<torch::Tensor> next_sp_conts_;
    std::vector<torch::Tensor> next_ag_feats_;
    std::vector<float>         discounts_;
};

} // namespace pommerman::rl
