#pragma once
#include <ftxui/dom/elements.hpp>
#include "../environment.hpp"

namespace pommerman::ui {

// Returns an FTXUI Element rendering the full game state:
// board grid on the left, agent status panel on the right,
// step counter / game result at the bottom.
ftxui::Element render_game(const Environment& env);

} // namespace pommerman::ui
