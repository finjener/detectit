#ifndef DETECTIONPIPELINE_H
#define DETECTIONPIPELINE_H

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <chrono>
#include <mutex>
#include "DetectionConfig.h" 

struct Detection {
    cv::Rect box;
    float score;
    int classId;
    std::string className;
};

struct DetectionPipeline {
public:
    DetectionPipeline(); 
    ~DetectionPipeline(); 

    
    bool initialize(const std::string& modelPath = DetectionConfig::DEFAULT_MODEL_PATH, 
                    const std::string& classNamesPath = DetectionConfig::DEFAULT_CLASS_NAMES_PATH);

    cv::Mat processFrame(const cv::Mat& inputFrame, double& outFps);

    void setConfidenceThreshold(float value);
    void setScoreThreshold(float value);
    void setNmsThreshold(float value);

private:

    std::mutex settingsMutex;


    std::unique_ptr<Ort::Env> onnxEnv;
    std::unique_ptr<Ort::Session> onnxSession;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo{nullptr};

    std::vector<std::string> inputNodeNames;
    std::vector<std::vector<int64_t>> inputNodeShapes;
    std::vector<std::string> outputNodeNames;
    std::vector<std::vector<int64_t>> outputNodeShapes;

    std::vector<std::string> classNames;


    int INPUT_WIDTH = 640;
    int INPUT_HEIGHT = 640;
    float scoreThreshold;    
    float nmsThreshold;       
    float confidenceThreshold; 


    void loadClassNames(const std::string& filename);
    std::vector<float> preprocessFrame(const cv::Mat& frame);
    std::vector<Ort::Value> runInference(const std::vector<float>& inputTensorValues);
    std::vector<Detection> postprocessOutput(const std::vector<Ort::Value>& outputTensors, const cv::Size& originalFrameSize);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections, double fps);


    std::chrono::high_resolution_clock::time_point fpsStartTime;
    double currentFps = 0.0;
};

#endif // DETECTIONPIPELINE_H 