add_subdirectory(rl)

# ── Training apps ─────────────────────────────────────────────────────────────
add_executable(train_ppo train_ppo.cpp)
target_include_directories(train_ppo SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(train_ppo PRIVATE pommerman_rl_lib ${TORCH_LIBRARIES} CLI11::CLI11)
target_compile_options(train_ppo PRIVATE ${TORCH_CXX_FLAGS} -Wall -Wextra)

add_executable(train_dqn train_dqn.cpp)
target_include_directories(train_dqn SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(train_dqn PRIVATE pommerman_rl_lib ${TORCH_LIBRARIES} CLI11::CLI11)
target_compile_options(train_dqn PRIVATE ${TORCH_CXX_FLAGS} -Wall -Wextra)

# ── Core library (no UI dependencies) ────────────────────────────────────────
add_library(pommerman_lib
    characters.cpp
    utility.cpp
    forward_model.cpp
    environment.cpp
    agents/random_agent.cpp
)
target_include_directories(pommerman_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(pommerman_lib PRIVATE -Wall -Wextra -Wpedantic)

# ── UI library (depends on core + FTXUI) ─────────────────────────────────────
add_library(pommerman_ui
    ui/renderer.cpp
)
target_include_directories(pommerman_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(pommerman_ui
    PUBLIC  pommerman_lib
    PRIVATE ftxui::dom
            ftxui::screen
)
target_compile_options(pommerman_ui PRIVATE -Wall -Wextra -Wpedantic)

# ── Demo executable ───────────────────────────────────────────────────────────
add_executable(pommerman_ffa main.cpp)
target_link_libraries(pommerman_ffa
    PRIVATE pommerman_ui
            ftxui::component
            ftxui::screen
            ftxui::dom
)
target_compile_options(pommerman_ffa PRIVATE -Wall -Wextra -Wpedantic)
