#include "pommerman/rl/replay_buffer.hpp"
#include <stdexcept>

namespace pommerman::rl {

ReplayBuffer::ReplayBuffer(int capacity) : capacity_(capacity) {
    item_idxs_.resize(capacity);
    sp_conts_.resize(capacity);
    ag_feats_.resize(capacity);
    actions_.resize(capacity, 0);
    rewards_.resize(capacity, 0.0f);
    next_item_idxs_.resize(capacity);
    next_sp_conts_.resize(capacity);
    next_ag_feats_.resize(capacity);
    dones_.resize(capacity, 0.0f);
}

void ReplayBuffer::add(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat,
        int   action,
        float reward,
        const torch::Tensor& next_item_idx,
        const torch::Tensor& next_sp_cont,
        const torch::Tensor& next_ag_feat,
        bool  done) {
    // Clone to own the storage (inputs may be views/slices)
    item_idxs_[pos_]      = item_idx.clone();
    sp_conts_[pos_]       = sp_cont.clone();
    ag_feats_[pos_]       = ag_feat.clone();
    actions_[pos_]        = action;
    rewards_[pos_]        = reward;
    next_item_idxs_[pos_] = next_item_idx.clone();
    next_sp_conts_[pos_]  = next_sp_cont.clone();
    next_ag_feats_[pos_]  = next_ag_feat.clone();
    dones_[pos_]          = done ? 1.0f : 0.0f;

    pos_ = (pos_ + 1) % capacity_;
    ++count_;
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor>
ReplayBuffer::sample(int batch_size, torch::Device device) {
    int n = size();
    if (n < batch_size)
        throw std::runtime_error("ReplayBuffer: not enough samples");

    auto indices  = torch::randperm(n, torch::kLong).slice(0, 0, batch_size);
    auto idx_acc  = indices.accessor<int64_t, 1>();

    std::vector<torch::Tensor> b_item, b_sp, b_ag;
    std::vector<torch::Tensor> b_ni,   b_ns, b_na;
    std::vector<int64_t>       b_acts;
    std::vector<float>         b_rew, b_done;
    b_item.reserve(batch_size); b_sp.reserve(batch_size); b_ag.reserve(batch_size);
    b_ni.reserve(batch_size);   b_ns.reserve(batch_size); b_na.reserve(batch_size);
    b_acts.reserve(batch_size);
    b_rew.reserve(batch_size);  b_done.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        int idx = static_cast<int>(idx_acc[i]);
        b_item.push_back(item_idxs_[idx]);
        b_sp.push_back(sp_conts_[idx]);
        b_ag.push_back(ag_feats_[idx]);
        b_ni.push_back(next_item_idxs_[idx]);
        b_ns.push_back(next_sp_conts_[idx]);
        b_na.push_back(next_ag_feats_[idx]);
        b_acts.push_back(static_cast<int64_t>(actions_[idx]));
        b_rew.push_back(rewards_[idx]);
        b_done.push_back(dones_[idx]);
    }

    return {
        torch::stack(b_item, 0).to(device),
        torch::stack(b_sp,   0).to(device),
        torch::stack(b_ag,   0).to(device),
        torch::tensor(b_acts, torch::TensorOptions().dtype(torch::kLong)).to(device),
        torch::tensor(b_rew,  torch::kFloat).to(device),
        torch::stack(b_ni,   0).to(device),
        torch::stack(b_ns,   0).to(device),
        torch::stack(b_na,   0).to(device),
        torch::tensor(b_done, torch::kFloat).to(device)
    };
}

} // namespace pommerman::rl
