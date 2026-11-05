add_executable(test_pommerman
    test_characters.cpp
    test_utility.cpp
    test_forward_model.cpp
    test_environment.cpp
)
target_link_libraries(test_pommerman
    PRIVATE pommerman_lib
            Catch2::Catch2WithMain
)
target_compile_options(test_pommerman PRIVATE -Wall -Wextra)

catch_discover_tests(test_pommerman)
