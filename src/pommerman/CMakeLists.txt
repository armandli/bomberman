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
