cmake_minimum_required(VERSION 3.16)

project(DetectIt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

find_package(OpenCV 4 REQUIRED)

set(ONNXRUNTIME_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.21.0)
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_BASE_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_BASE_DIR}/lib")

if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIB_DIR}.")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}.")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(SOURCES
    main.cpp
    MainWindow.cpp
    DetectionPipeline.cpp
)

set(HEADERS
    MainWindow.h
    DetectionPipeline.h
    DetectionConfig.h
)

add_executable(DetectIt ${SOURCES} ${HEADERS})

target_link_libraries(DetectIt
    PRIVATE
    Qt6::Widgets
    Qt6::Concurrent
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
)

target_include_directories(DetectIt
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

# Copy data files to the 'data' subdirectory in the build directory
file(COPY ../data/ DESTINATION ${CMAKE_BINARY_DIR}/data)