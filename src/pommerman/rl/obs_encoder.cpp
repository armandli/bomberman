#include "pommerman/rl/obs_encoder.hpp"

namespace pommerman::rl {

ObsEncoderImpl::ObsEncoderImpl() {
    item_embed   = register_module("item_embed",
                       torch::nn::Embedding(ITEM_TYPES, ITEM_EMBED_DIM));
    spatial_proj = register_module("spatial_proj",
                       torch::nn::Linear(SPATIAL_FEAT_DIM, D_MODEL));
    agent_proj   = register_module("agent_proj",
                       torch::nn::Linear(AGENT_FEAT_DIM, D_MODEL));
    pos_embed    = register_module("pos_embed",
                       torch::nn::Embedding(NUM_TOKENS, D_MODEL));

    auto layer_opts = torch::nn::TransformerEncoderLayerOptions(D_MODEL, N_HEADS)
        .dim_feedforward(DIM_FF)
        .dropout(0.0)
        .activation(torch::kGELU);
    auto enc_opts = torch::nn::TransformerEncoderOptions(
        torch::nn::TransformerEncoderLayer(layer_opts), N_LAYERS);
    transformer = register_module("transformer",
                      torch::nn::TransformerEncoder(enc_opts));

    post_norm = register_module("post_norm",
                    torch::nn::LayerNorm(torch::nn::LayerNormOptions({D_MODEL})));
}

torch::Tensor ObsEncoderImpl::forward(
        const torch::Tensor& item_idx,
        const torch::Tensor& sp_cont,
        const torch::Tensor& ag_feat) {

    // Item embeddings: [B, 121, ITEM_EMBED_DIM]
    auto ie = item_embed(item_idx);

    // Spatial features: [B, 121, SPATIAL_FEAT_DIM]
    auto spatial = torch::cat({ie, sp_cont}, /*dim=*/2);
    // Project to model dim: [B, 121, D_MODEL]
    spatial = torch::gelu(spatial_proj(spatial));

    // Agent CLS token: [B, 1, D_MODEL]
    auto ag_token = torch::gelu(agent_proj(ag_feat)).unsqueeze(1);

    // Full token sequence: [B, NUM_TOKENS, D_MODEL]  (CLS first)
    auto tokens = torch::cat({ag_token, spatial}, /*dim=*/1);

    // Add learned positional embeddings
    auto pos_ids = torch::arange(
        NUM_TOKENS,
        torch::TensorOptions().dtype(torch::kLong).device(tokens.device()));
    tokens = tokens + pos_embed(pos_ids).unsqueeze(0);

    // Transformer expects [seq, batch, d_model]
    tokens = tokens.permute({1, 0, 2}).contiguous();
    tokens = transformer(tokens);

    // CLS token output: [B, D_MODEL]
    return post_norm(tokens[0]);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
obs_to_tensors(const std::vector<Observation>& obs) {
    int B = static_cast<int>(obs.size());

    auto item_idx = torch::zeros({B, NUM_SPATIAL}, torch::kLong);
    auto sp_cont  = torch::zeros({B, NUM_SPATIAL, SPATIAL_CONT_DIM});
    auto ag_feat  = torch::zeros({B, AGENT_FEAT_DIM});

    auto item_acc = item_idx.accessor<int64_t, 2>();
    auto sp_acc   = sp_cont.accessor<float, 3>();
    auto ag_acc   = ag_feat.accessor<float, 2>();

    constexpr float MAX_BLAST = static_cast<float>(DEFAULT_BLAST_STRENGTH + MAX_AMMO);
    constexpr float MAX_LIFE  = static_cast<float>(DEFAULT_BOMB_LIFE);

    for (int b = 0; b < B; ++b) {
        const auto& o = obs[b];

        int cell = 0;
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c, ++cell) {
                item_acc[b][cell] = static_cast<int64_t>(o.board[r][c]);

                float blast_n = o.bomb_blast_strength[r][c] / MAX_BLAST;
                float life_n  = o.bomb_life[r][c]           / MAX_LIFE;
                int   dir     = o.bomb_moving_direction[r][c]; // 0=stop, 1-4=dir
                float flame_n = o.flame_life[r][c]           / static_cast<float>(FLAME_LIFE);

                sp_acc[b][cell][0] = blast_n;
                sp_acc[b][cell][1] = life_n;
                if (dir >= 0 && dir <= 4)
                    sp_acc[b][cell][2 + dir] = 1.0f; // one-hot direction
                sp_acc[b][cell][7] = flame_n;
            }
        }

        ag_acc[b][0] = static_cast<float>(o.ammo)          / static_cast<float>(MAX_AMMO);
        ag_acc[b][1] = static_cast<float>(o.blast_strength) / MAX_BLAST;
        ag_acc[b][2] = o.can_kick ? 1.0f : 0.0f;
        ag_acc[b][3] = static_cast<float>(o.position.row)   / static_cast<float>(BOARD_SIZE - 1);
        ag_acc[b][4] = static_cast<float>(o.position.col)   / static_cast<float>(BOARD_SIZE - 1);
        // agent_id one-hot [5..8]
        if (o.agent_id >= 0 && o.agent_id < NUM_AGENTS)
            ag_acc[b][5 + o.agent_id] = 1.0f;
        // alive agents one-hot [9..12]
        for (int id : o.alive_agents)
            if (id >= 0 && id < NUM_AGENTS)
                ag_acc[b][9 + id] = 1.0f;
    }

    return {item_idx, sp_cont, ag_feat};
}

} // namespace pommerman::rl
