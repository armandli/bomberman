#pragma once
#include "pommerman/types.hpp"
#include "pommerman/constants.hpp"
#include <algorithm>
#include <array>
#include <vector>

namespace pommerman::rl {

inline bool is_alive_in(const Observation& obs, int id) {
    return std::find(obs.alive_agents.begin(), obs.alive_agents.end(), id)
           != obs.alive_agents.end();
}

// Augments sparse base rewards with dense shaping signals.
// All rewards are from an individual agent's self-play perspective.
//
// Per-agent episode semantics used by all trainers: an agent's episode ends
// on the step it dies (it receives -1 there) or when the game ends. Steps
// taken while already dead are not experience and must be discarded
// (see agent_was_alive / agent_terminal).
struct RewardShaper {
    static constexpr float DEATH_PENALTY = -1.0f;

    // Did agent i act in this transition (alive before the step)?
    static bool agent_was_alive(const std::vector<Observation>& prev_obs, int i) {
        return is_alive_in(prev_obs[i], i);
    }

    // Is this transition terminal for agent i (died this step, or game over)?
    static bool agent_terminal(const std::vector<Observation>& curr_obs, int i, bool done) {
        return done || !is_alive_in(curr_obs[i], i);
    }

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
            // Already-dead agents get nothing (their episode ended earlier)
            if (!is_alive_in(prev_obs[i], i)) { rewards[i] = 0.0f; continue; }

            const bool alive_now = is_alive_in(curr_obs[i], i);
            rewards[i] = static_cast<float>(base_rewards[i]);

            // Died mid-game: the environment only scores at game end, so apply
            // the loss now, on the step it happened.
            if (!alive_now && !done) rewards[i] += DEATH_PENALTY;

            // Survival bonus per step
            if (alive_now && !done)
                rewards[i] += 0.003f;

            // Wood destruction (attributed to all alive agents equally)
            if (wood_bonus > 0.0f)
                rewards[i] += wood_bonus;

            // Enemy elimination bonus (teammates excluded)
            for (int j : prev_obs[i].enemies) {
                if (j < 0 || j >= NUM_AGENTS) continue;
                if (is_alive_in(prev_obs[i], j) && !is_alive_in(curr_obs[i], j))
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
