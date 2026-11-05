#pragma once
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace pommerman::rl {

// Uniform-random circular replay buffer (CPU storage, moves to device on sample).
class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacity);

    void add(
        const torch::Tensor& item_idx,       // [121] int64
        const torch::Tensor& sp_cont,        // [121, SPATIAL_CONT_DIM] float
        const torch::Tensor& ag_feat,        // [AGENT_FEAT_DIM] float
        int   action,
        float reward,
        const torch::Tensor& next_item_idx,
        const torch::Tensor& next_sp_cont,
        const torch::Tensor& next_ag_feat,
        bool  done
    );

    // Sample a minibatch; moves batched tensors to `device`.
    // Returns: (item_idx, sp_cont, ag_feat, actions, rewards,
    //           next_item_idx, next_sp_cont, next_ag_feat, dones)
    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor,  // obs
        torch::Tensor, torch::Tensor,                  // actions, rewards
        torch::Tensor, torch::Tensor, torch::Tensor,  // next_obs
        torch::Tensor                                  // dones
    > sample(int batch_size, torch::Device device);

    int size()     const { return std::min(count_, capacity_); }
    int capacity() const { return capacity_; }

private:
    int capacity_;
    int pos_   = 0;
    int count_ = 0;

    std::vector<torch::Tensor> item_idxs_;
    std::vector<torch::Tensor> sp_conts_;
    std::vector<torch::Tensor> ag_feats_;
    std::vector<int>           actions_;
    std::vector<float>         rewards_;
    std::vector<torch::Tensor> next_item_idxs_;
    std::vector<torch::Tensor> next_sp_conts_;
    std::vector<torch::Tensor> next_ag_feats_;
    std::vector<float>         dones_;
};

} // namespace pommerman::rl
