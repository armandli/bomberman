add_executable(test_pommerman
    test_characters.cpp
    test_utility.cpp
    test_forward_model.cpp
    test_environment.cpp
    test_simple_agent.cpp
)
target_link_libraries(test_pommerman
    PRIVATE pommerman_lib
            Catch2::Catch2WithMain
)
target_compile_options(test_pommerman PRIVATE -Wall -Wextra)

catch_discover_tests(test_pommerman)

# RL components; separate because they need libtorch and test_pommerman does not.
add_executable(test_rl test_rl.cpp)
target_include_directories(test_rl SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(test_rl
    PRIVATE pommerman_rl_lib
            ${TORCH_LIBRARIES}
            Catch2::Catch2WithMain
)
target_compile_options(test_rl PRIVATE ${TORCH_CXX_FLAGS} -Wall -Wextra)

catch_discover_tests(test_rl)
