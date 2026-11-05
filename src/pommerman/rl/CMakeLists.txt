# ── RL library (pommerman_lib + libtorch) ─────────────────────────────────────
add_library(pommerman_rl_lib
    obs_encoder.cpp
    ppo_model.cpp
    dqn_model.cpp
    replay_buffer.cpp
    ppo_trainer.cpp
    dqn_trainer.cpp
)
# Public include: src/ so consumers use #include "pommerman/rl/..."
target_include_directories(pommerman_rl_lib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(pommerman_rl_lib
    PUBLIC  pommerman_lib
    PRIVATE ${TORCH_LIBRARIES}
)
target_compile_options(pommerman_rl_lib PRIVATE
    ${TORCH_CXX_FLAGS}
    -Wall -Wextra
)
