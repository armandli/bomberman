#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include "environment.hpp"
#include "agents/random_agent.hpp"
#include "ui/renderer.hpp"

using namespace pommerman;
using namespace ftxui;

int main() {
    Environment env(GameType::FFA);

    std::vector<std::unique_ptr<BaseAgent>> agents;
    for (int i = 0; i < NUM_AGENTS; ++i)
        agents.push_back(std::make_unique<RandomAgent>());

    auto obs = env.reset();
    for (int i = 0; i < NUM_AGENTS; ++i) agents[i]->episode_start(i);

    auto screen = ScreenInteractive::TerminalOutput();
    std::atomic<bool> quit{false};
    // env is stepped on the game thread and rendered on the UI thread.
    std::mutex env_mutex;

    auto component = Renderer([&] {
        std::lock_guard lock(env_mutex);
        return ui::render_game(env);
    });
    component |= CatchEvent([&](Event event) {
        if (event == Event::Character('q') || event == Event::Escape) {
            quit = true;
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    // Game loop runs in a background thread at ~4 steps/second.
    std::thread game_thread([&] {
        bool done = false;
        while (!quit && !done) {
            std::vector<Action> actions;
            actions.reserve(NUM_AGENTS);
            for (int i = 0; i < NUM_AGENTS; ++i)
                actions.push_back(agents[i]->act(obs[i]));

            std::vector<double> rewards;
            {
                std::lock_guard lock(env_mutex);
                auto [new_obs, step_rewards, step_done, info] = env.step(actions);
                obs     = std::move(new_obs);
                rewards = std::move(step_rewards);
                done    = step_done;
            }

            if (done) {
                for (int i = 0; i < NUM_AGENTS; ++i)
                    agents[i]->episode_end(rewards[i]);
            }

            screen.PostEvent(Event::Custom);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        // Final render of end-state
        screen.PostEvent(Event::Custom);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        if (!quit) screen.ExitLoopClosure()();
    });

    screen.Loop(component);
    quit = true;
    game_thread.join();
    return 0;
}
