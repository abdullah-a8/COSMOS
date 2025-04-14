find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)

# Create the pybind11 module
pybind11_add_module(pdf_extractor pdf_extractor.cpp)

# Link against libraries
target_link_libraries(pdf_extractor PRIVATE 
    OpenSSL::Crypto
    ${POPPLER_LIBRARIES}
)

# Include headers
target_include_directories(pdf_extractor PRIVATE 
    ${OPENSSL_INCLUDE_DIR}
    ${POPPLER_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/lib/pybind11/include
)

# Add compile flags from pkg-config
target_compile_options(pdf_extractor PRIVATE ${POPPLER_CFLAGS_OTHER})

# Install the module
install(TARGETS pdf_extractor
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../core/cpp_modules) 