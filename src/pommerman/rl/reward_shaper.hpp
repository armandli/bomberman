#pragma once
#include "pommerman/types.hpp"
#include "pommerman/constants.hpp"
#include <algorithm>
#include <array>
#include <vector>

namespace pommerman::rl {

// Augments sparse base rewards with dense shaping signals.
// All rewards are from an individual agent's self-play perspective.
struct RewardShaper {
    static std::array<float, NUM_AGENTS> shape(
            const std::vector<Observation>& prev_obs,
            const std::vector<Observation>& curr_obs,
            const std::vector<double>&      base_rewards,
            bool                            done) {

        // Count Wood tiles on the shared board (all agents see the same board)
        int prev_woods = 0, curr_woods = 0;
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                if (prev_obs[0].board[r][c] == Item::Wood) ++prev_woods;
                if (curr_obs[0].board[r][c] == Item::Wood) ++curr_woods;
            }
        }
        float wood_bonus = static_cast<float>(prev_woods - curr_woods) * 0.1f;

        std::array<float, NUM_AGENTS> rewards{};

        for (int i = 0; i < NUM_AGENTS; ++i) {
            rewards[i] = static_cast<float>(base_rewards[i]);

            auto alive_before = [&](int id) {
                return std::find(prev_obs[i].alive_agents.begin(),
                                 prev_obs[i].alive_agents.end(), id)
                       != prev_obs[i].alive_agents.end();
            };
            auto alive_after = [&](int id) {
                return std::find(curr_obs[i].alive_agents.begin(),
                                 curr_obs[i].alive_agents.end(), id)
                       != curr_obs[i].alive_agents.end();
            };

            if (!alive_before(i)) continue; // skip already-dead agents

            // Survival bonus per step
            if (alive_after(i) && !done)
                rewards[i] += 0.003f;

            // Wood destruction (attributed to all alive agents equally)
            if (wood_bonus > 0.0f)
                rewards[i] += wood_bonus;

            // Enemy elimination bonus
            for (int j = 0; j < NUM_AGENTS; ++j) {
                if (j == i) continue;
                if (alive_before(j) && !alive_after(j))
                    rewards[i] += 0.3f;
            }

            // Powerup collection
            if (curr_obs[i].blast_strength > prev_obs[i].blast_strength)
                rewards[i] += 0.03f;
            if (curr_obs[i].can_kick && !prev_obs[i].can_kick)
                rewards[i] += 0.03f;
        }

        return rewards;
    }
};

} // namespace pommerman::rl
