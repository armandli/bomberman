#pragma once
#include <torch/torch.h>
#include "pommerman/types.hpp"
#include "pommerman/constants.hpp"
#include <tuple>
#include <vector>

namespace pommerman::rl {

// Feature dimensions
constexpr int ITEM_TYPES       = 14;   // Item::Passage(0)..Item::Agent3(13)
constexpr int ITEM_EMBED_DIM   = 16;
constexpr int SPATIAL_CONT_DIM = 8;    // blast_norm, bomb_life_norm, dir[5], flame_norm
constexpr int SPATIAL_FEAT_DIM = ITEM_EMBED_DIM + SPATIAL_CONT_DIM; // 24
constexpr int AGENT_FEAT_DIM   = 13;   // ammo, blast, can_kick, pos_r, pos_c, id[4], alive[4]
constexpr int D_MODEL          = 256;
constexpr int N_HEADS          = 8;
constexpr int N_LAYERS         = 6;
constexpr int DIM_FF           = 1024;
constexpr int NUM_SPATIAL      = BOARD_SIZE * BOARD_SIZE; // 121
constexpr int NUM_TOKENS       = NUM_SPATIAL + 1;          // +1 agent CLS token
constexpr int NUM_ACTIONS      = 6;    // Stop, Up, Down, Left, Right, Bomb

struct ObsEncoderImpl : torch::nn::Module {
    torch::nn::Embedding       item_embed{nullptr};
    torch::nn::Linear          spatial_proj{nullptr};
    torch::nn::Linear          agent_proj{nullptr};
    torch::nn::Embedding       pos_embed{nullptr};
    torch::nn::TransformerEncoder transformer{nullptr};
    torch::nn::LayerNorm       post_norm{nullptr};

    ObsEncoderImpl();

    // item_idx : [B, 121]           int64
    // sp_cont  : [B, 121, 8]        float32
    // ag_feat  : [B, AGENT_FEAT_DIM] float32
    // returns  : [B, D_MODEL]        float32
    torch::Tensor forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat
    );
};
TORCH_MODULE(ObsEncoder);

// Convert a batch of Observations to raw tensors (CPU).
// Returns (item_idx [B,121] int64, sp_cont [B,121,8] float, ag_feat [B,13] float).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
obs_to_tensors(const std::vector<Observation>& obs);

} // namespace pommerman::rl
