#include "environment.hpp"
#include "utility.hpp"

namespace pommerman {

Environment::Environment(GameType game_type, bool partially_observable,
                         int board_size, uint32_t seed)
    : _game_type(game_type)
    , _partially_observable(partially_observable)
    , _board_size(board_size)
    , _rng(seed == 0 ? std::random_device{}() : seed) {}

std::vector<Observation> Environment::reset() {
    _step_count = 0;
    _done       = false;
    _last_info  = {};
    _bombs.clear();
    _flames.clear();
    _hidden_items.clear();

    _static_board = make_board(_rng, _board_size);
    _hidden_items = make_items(_static_board, _rng);

    auto starts = agent_start_positions(_board_size);
    _agents.resize(NUM_AGENTS);
    for (int i = 0; i < NUM_AGENTS; ++i) {
        _agents[i].reset();
        _agents[i].set_agent_id(i, _game_type);
        _agents[i].position = starts[i];
    }

    rebuild_board(_board, _static_board, _agents, _bombs, _flames);
    return get_observations();
}

std::tuple<std::vector<Observation>, std::vector<double>, bool, GameInfo>
Environment::step(std::span<const Action> actions) {
    if (_done) return {get_observations(), std::vector<double>(NUM_AGENTS, 0.0), true, _last_info};

    _fm.step(actions, _board, _agents, _bombs, _flames, _static_board, _hidden_items);
    ++_step_count;

    _done      = _fm.is_done(_agents, _step_count, _game_type);
    _last_info = _fm.get_info(_agents, _game_type, _step_count);
    auto rewards = _fm.get_rewards(_agents, _game_type, _done, _step_count);

    return {get_observations(), rewards, _done, _last_info};
}

std::vector<Observation> Environment::get_observations() const {
    std::vector<Observation> obs;
    obs.reserve(_agents.size());
    for (auto& agent : _agents) obs.push_back(_build_observation(agent));
    return obs;
}

Observation Environment::_build_observation(const Agent& agent) const {
    Observation obs;
    obs.agent_id       = agent.agent_id;
    obs.position       = agent.position;
    obs.ammo           = agent.ammo;
    obs.blast_strength = agent.blast_strength;
    obs.can_kick       = agent.can_kick;
    obs.teammate       = (agent.teammate_id >= 0)
                             ? std::optional<int>(agent.teammate_id)
                             : std::nullopt;
    obs.enemies        = agent.enemy_ids;

    for (auto& a : _agents) {
        if (a.is_alive) obs.alive_agents.push_back(a.agent_id);
    }

    // Board (with optional fog-of-war)
    if (!_partially_observable) {
        obs.board = _board;
    } else {
        for (auto& row : obs.board) row.fill(Item::Fog);
        for (int r = -AGENT_VIEW_SIZE; r <= AGENT_VIEW_SIZE; ++r) {
            for (int c = -AGENT_VIEW_SIZE; c <= AGENT_VIEW_SIZE; ++c) {
                int br = agent.position.row + r;
                int bc = agent.position.col + c;
                if (br >= 0 && br < _board_size && bc >= 0 && bc < _board_size)
                    obs.board[br][bc] = _board[br][bc];
            }
        }
    }

    // Bomb overlays
    for (auto& bomb : _bombs) {
        int r = bomb.position.row, c = bomb.position.col;
        obs.bomb_blast_strength[r][c] = bomb.blast_strength;
        obs.bomb_life[r][c]           = bomb.life;
        obs.bomb_moving_direction[r][c] =
            bomb.is_moving() ? static_cast<int>(*bomb.moving_direction) : 0;
    }

    // Flame life overlay
    for (auto& flame : _flames) {
        int r = flame.position.row, c = flame.position.col;
        obs.flame_life[r][c] = flame.life;
    }

    return obs;
}

} // namespace pommerman
