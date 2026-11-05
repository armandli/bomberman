add_library(pommerman_muzero_lib
    muzero_model.cpp
    mcts.cpp
    game_buffer.cpp
    muzero_trainer.cpp
)

target_include_directories(pommerman_muzero_lib
    PUBLIC  ${CMAKE_SOURCE_DIR}/src
    SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(pommerman_muzero_lib
    PUBLIC  pommerman_rl_lib
            ${TORCH_LIBRARIES}
)

target_compile_options(pommerman_muzero_lib PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET pommerman_muzero_lib PROPERTY CXX_STANDARD 20)
