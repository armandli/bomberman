#include "renderer.hpp"
#include <format>
#include <ftxui/dom/elements.hpp>

namespace pommerman::ui {

using namespace ftxui;

// ── Cell rendering ────────────────────────────────────────────────────────────

static Color agent_bg_color(int id) noexcept {
    switch (id) {
        case 0: return Color::Blue;
        case 1: return Color::Green;
        case 2: return Color::Cyan;
        case 3: return Color::Magenta;
        default: return Color::White;
    }
}

static Element render_cell(Item item, int r, int c, const Environment& env) {
    switch (item) {
        case Item::Passage:
            return text("  ") | bgcolor(Color::RGB(40, 40, 40));

        case Item::Rigid:
            return text("██") | color(Color::RGB(140, 140, 140));

        case Item::Wood:
            return text("▪▪") | color(Color::RGB(205, 133, 63))
                              | bgcolor(Color::RGB(60, 30, 5));

        case Item::Bomb: {
            // Show countdown from bombs list
            for (auto& b : env.bombs()) {
                if (b.position.row == r && b.position.col == c) {
                    auto s = std::format("{:>2}", b.life);
                    return text(s) | color(Color::Black) | bgcolor(Color::Yellow);
                }
            }
            return text(" B") | bgcolor(Color::Yellow) | color(Color::Black);
        }

        case Item::Flames:
            return text("✦✦") | color(Color::RGB(255, 80, 0))
                              | bgcolor(Color::RGB(80, 10, 0));

        case Item::ExtraBomb:
            return text("+B") | color(Color::Yellow) | bgcolor(Color::RGB(80, 0, 80));

        case Item::IncrRange:
            return text("+R") | color(Color::Yellow) | bgcolor(Color::RGB(0, 60, 80));

        case Item::Kick:
            return text(" K") | color(Color::Black) | bgcolor(Color::White);

        case Item::Agent0:
        case Item::Agent1:
        case Item::Agent2:
        case Item::Agent3: {
            int id  = static_cast<int>(item) - static_cast<int>(Item::Agent0);
            auto lbl = std::format(" {}", id + 1);
            return text(lbl) | bold | color(Color::White) | bgcolor(agent_bg_color(id));
        }

        case Item::Fog:
            return text("  ") | bgcolor(Color::RGB(20, 20, 20));

        default:
            return text("  ");
    }
}

static Element render_board(const Environment& env) {
    std::vector<Element> rows;
    rows.reserve(BOARD_SIZE);
    for (int r = 0; r < BOARD_SIZE; ++r) {
        std::vector<Element> cells;
        cells.reserve(BOARD_SIZE);
        for (int c = 0; c < BOARD_SIZE; ++c) {
            cells.push_back(render_cell(env.board()[r][c], r, c, env));
        }
        rows.push_back(hbox(std::move(cells)));
    }
    return vbox(std::move(rows)) | border;
}

// ── Agent status panel ────────────────────────────────────────────────────────

static Element render_agents(const Environment& env) {
    std::vector<Element> entries;
    entries.push_back(text(" Agents ") | bold | center);
    entries.push_back(separator());

    for (auto& agent : env.agents()) {
        Color bg = agent_bg_color(agent.agent_id);
        auto header = hbox({
            text(std::format(" {} ", agent.agent_id + 1)) | bold | bgcolor(bg) | color(Color::White),
            text(agent.is_alive ? " ALIVE" : " DEAD ") |
                color(agent.is_alive ? Color::Green : Color::Red),
        });
        entries.push_back(header);
        if (agent.is_alive) {
            entries.push_back(text(std::format("  ammo:  {}", agent.ammo)));
            entries.push_back(text(std::format("  range: {}", agent.blast_strength)));
            entries.push_back(text(std::format("  kick:  {}", agent.can_kick ? "yes" : "no")));
        }
        entries.push_back(separator());
    }

    return vbox(std::move(entries)) | border;
}

// ── Status bar ────────────────────────────────────────────────────────────────

static Element render_status(const Environment& env) {
    std::string msg;
    if (!env.is_done()) {
        msg = std::format(" Step {:>4}/{} ", env.step_count(), MAX_STEPS);
    } else {
        auto& info = env.last_info();
        if (info.result == Result::Tie) {
            msg = " Game over: TIE ";
        } else if (!info.winners.empty()) {
            msg = std::format(" Game over: Agent {} wins! ", info.winners[0] + 1);
        } else {
            msg = " Game over ";
        }
    }
    return text(msg) | bold | center | border;
}

// ── Public entry point ────────────────────────────────────────────────────────

ftxui::Element render_game(const Environment& env) {
    return vbox({
        hbox({
            render_board(env),
            render_agents(env),
        }),
        render_status(env),
    });
}

} // namespace pommerman::ui
