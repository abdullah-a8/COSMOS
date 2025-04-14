pybind11_add_module(text_chunker text_chunker.cpp)

# Make pybind11 headers available
target_include_directories(text_chunker PRIVATE 
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/lib/pybind11/include
)

# Install the module
install(TARGETS text_chunker
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../core/cpp_modules) 