#include "DetectionPipeline.h"
#include <QDebug> 
#include <stdexcept>
#include <numeric>
#include <algorithm> 
#include <mutex> 
#include "DetectionConfig.h" 

DetectionPipeline::DetectionPipeline()
    : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      scoreThreshold(DetectionConfig::DEFAULT_SCORE_THRESHOLD),    
      nmsThreshold(DetectionConfig::DEFAULT_NMS_THRESHOLD),
      confidenceThreshold(DetectionConfig::DEFAULT_CONFIDENCE_THRESHOLD)
{
    fpsStartTime = std::chrono::high_resolution_clock::now(); 
}

DetectionPipeline::~DetectionPipeline()
{
}


void DetectionPipeline::setConfidenceThreshold(float value) {
    std::lock_guard<std::mutex> lock(settingsMutex); 
    confidenceThreshold = std::clamp(value, 0.0f, 1.0f); 
    qDebug() << "Confidence threshold set to" << confidenceThreshold;
}

void DetectionPipeline::setScoreThreshold(float value) {
    std::lock_guard<std::mutex> lock(settingsMutex);
    scoreThreshold = std::clamp(value, 0.0f, 1.0f); 
    qDebug() << "Score threshold set to" << scoreThreshold;
}

void DetectionPipeline::setNmsThreshold(float value) {
    std::lock_guard<std::mutex> lock(settingsMutex);
    nmsThreshold = std::clamp(value, 0.0f, 1.0f); 
    qDebug() << "NMS threshold set to" << nmsThreshold;
}

bool DetectionPipeline::initialize(const std::string& modelPath, const std::string& classNamesPath)
{
    try {
        static const std::string env_name = "DetectIt-ONNXRuntime-Env";
        onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, env_name.c_str());
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        onnxSession = std::make_unique<Ort::Session>(*onnxEnv, modelPath.c_str(), sessionOptions);

        
        qDebug() << "ONNX Runtime session created successfully from" << QString::fromStdString(modelPath);
        
        size_t numInputNodes = onnxSession->GetInputCount();
        inputNodeNames.resize(numInputNodes);
        inputNodeShapes.resize(numInputNodes);
        for (size_t i = 0; i < numInputNodes; ++i) {
            Ort::AllocatedStringPtr name = onnxSession->GetInputNameAllocated(i, allocator);
            inputNodeNames[i] = name.get();
            Ort::TypeInfo typeInfo = onnxSession->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputNodeShapes[i] = tensorInfo.GetShape();
        }
        size_t numOutputNodes = onnxSession->GetOutputCount();
        outputNodeNames.resize(numOutputNodes);
        outputNodeShapes.resize(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; ++i) {
            Ort::AllocatedStringPtr name = onnxSession->GetOutputNameAllocated(i, allocator);
            outputNodeNames[i] = name.get();
            Ort::TypeInfo typeInfo = onnxSession->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            outputNodeShapes[i] = tensorInfo.GetShape();
        }
        if (inputNodeShapes.empty() || outputNodeShapes.empty()) {
             throw std::runtime_error("Model has no input or output nodes.");
        }
        loadClassNames(classNamesPath);
        return true;
    } catch (const Ort::Exception& e) {
        qCritical() << "ONNX Runtime Error during initialize:" << e.what();
        return false;
    } catch (const std::exception& e) {
        qCritical() << "Initialization Error:" << e.what();
        return false;
    }
}


cv::Mat DetectionPipeline::processFrame(const cv::Mat& inputFrame, double& outFps)
{
    if (!onnxSession) {
        qWarning() << "ProcessFrame called, ONNX session is not initialized.";
        return cv::Mat(); 
    }
    if (inputFrame.empty()) {
        qWarning() << "ProcessFrame called with empty input frame.";
        return cv::Mat();
    }

    
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - fpsStartTime);
    fpsStartTime = currentTime;
    if (duration.count() > 0) {
        currentFps = 1000.0 / duration.count();
    }
    outFps = currentFps;

    cv::Mat annotatedFrame = inputFrame.clone(); 
    cv::Size originalSize = annotatedFrame.size();

    try {
        std::vector<float> inputTensorValues = preprocessFrame(annotatedFrame);
        std::vector<Ort::Value> outputTensors = runInference(inputTensorValues);
        std::vector<Detection> detections = postprocessOutput(outputTensors, originalSize);
        drawDetections(annotatedFrame, detections, currentFps);

    } catch (const Ort::Exception& e) {
         qCritical() << "ONNX Inference Error:" << e.what();
         return cv::Mat(); 
    } catch (const std::exception& e) {
         qCritical() << "Processing Error:" << e.what();
         return cv::Mat(); 
    }

    return annotatedFrame;
}


void DetectionPipeline::loadClassNames(const std::string& filename)
{
    classNames.clear();
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Class file name couldn't open: " + filename);
    }
    std::string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    qDebug() << "Loaded" << classNames.size() << "class names.";
}


std::vector<float> DetectionPipeline::preprocessFrame(const cv::Mat& frame)
{
    cv::Mat blob;
    cv::resize(frame, blob, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, cv::INTER_LINEAR);
    blob.convertTo(blob, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(blob, channels);
    std::vector<float> inputTensorValues(1 * 3 * INPUT_HEIGHT * INPUT_WIDTH);
    size_t offset = 0;
    for (int i = 0; i < 3; ++i) {
        memcpy(inputTensorValues.data() + offset, channels[i].data, INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
        offset += INPUT_HEIGHT * INPUT_WIDTH;
    }
    return inputTensorValues;
}

std::vector<Ort::Value> DetectionPipeline::runInference(const std::vector<float>& inputTensorValues)
{
    if (!onnxSession) {
        throw std::runtime_error("ONNX session not initialized.");
    }
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                              const_cast<float*>(inputTensorValues.data()),
                                                              inputTensorValues.size(),
                                                              inputNodeShapes[0].data(),
                                                              inputNodeShapes[0].size());
    std::vector<const char*> inputNamesCStr;
    inputNamesCStr.reserve(inputNodeNames.size());
    for (const auto& name : inputNodeNames) { inputNamesCStr.push_back(name.c_str()); }
    std::vector<const char*> outputNamesCStr;
    outputNamesCStr.reserve(outputNodeNames.size());
    for (const auto& name : outputNodeNames) { outputNamesCStr.push_back(name.c_str()); }
    return onnxSession->Run(Ort::RunOptions{nullptr},
                            inputNamesCStr.data(), &inputTensor, 1,
                            outputNamesCStr.data(), outputNamesCStr.size());
}

std::vector<Detection> DetectionPipeline::postprocessOutput(const std::vector<Ort::Value>& outputTensors, const cv::Size& originalFrameSize)
{
    if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
        throw std::runtime_error("Invalid output tensor from model.");
    }
    const float* outputData = outputTensors[0].GetTensorData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.size() != 3 || outputShape[0] != 1) {
        throw std::runtime_error("Unexpected output tensor shape.");
    }
    int numClasses = classNames.size();
    int elementsPerBox = outputShape[1];
    int numBoxes = outputShape[2];
    if (elementsPerBox != numClasses + 4) {
         numClasses = elementsPerBox - 4;
         if (numClasses <= 0) {
             throw std::runtime_error("Model output doesn't contain class scores.");
         }
    }
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    float scaleX = (float)originalFrameSize.width / INPUT_WIDTH;
    float scaleY = (float)originalFrameSize.height / INPUT_HEIGHT;

    float currentConfidenceThreshold;
    float currentScoreThreshold;
    float currentNmsThreshold;
    {
        std::lock_guard<std::mutex> lock(settingsMutex);
        currentConfidenceThreshold = confidenceThreshold;
        currentScoreThreshold = scoreThreshold;
        currentNmsThreshold = nmsThreshold;
    }

    for (int i = 0; i < numBoxes; ++i) {
        float cx = outputData[i];
        float cy = outputData[i + numBoxes];
        float w = outputData[i + 2 * numBoxes];
        float h = outputData[i + 3 * numBoxes];
        float maxClassScore = 0.0f;
        int classId = -1;
        for (int j = 0; j < numClasses; ++j) {
            float currentScore = outputData[(4 + j) * numBoxes + i];
            if (currentScore > maxClassScore) {
                maxClassScore = currentScore;
                classId = j;
            }
        }
        if (maxClassScore > currentConfidenceThreshold) {
            int left = static_cast<int>((cx - w / 2.0f) * scaleX);
            int top = static_cast<int>((cy - h / 2.0f) * scaleY);
            int width = static_cast<int>(w * scaleX);
            int height = static_cast<int>(h * scaleY);
            left = std::max(0, std::min(left, originalFrameSize.width - 1));
            top = std::max(0, std::min(top, originalFrameSize.height - 1));
            width = std::max(1, std::min(width, originalFrameSize.width - left));
            height = std::max(1, std::min(height, originalFrameSize.height - top));
            boxes.emplace_back(left, top, width, height);
            scores.push_back(maxClassScore);
            classIds.push_back(classId);
        }
    }
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, scores, currentScoreThreshold, currentNmsThreshold, nmsIndices);
    std::vector<Detection> detections;
    for (int index : nmsIndices) {
        Detection det;
        det.box = boxes[index];
        det.score = scores[index];
        det.classId = classIds[index];
        if (det.classId >= 0 && det.classId < classNames.size()) {
            det.className = classNames[det.classId];
        } else {
             det.className = "Unknown";
        }
        detections.push_back(det);
    }
    return detections;
}

void DetectionPipeline::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections, double fps)
{
    std::string fpsText = cv::format("FPS: %.1f", fps);
    cv::Point fpsPos(10, 25);
    cv::Scalar fpsColor(0, 0, 255);
    cv::putText(frame, fpsText, fpsPos, cv::FONT_HERSHEY_SIMPLEX, 0.7, fpsColor, 2);
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("%s: %.2f", det.className.c_str(), det.score);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int top = std::max(det.box.y, labelSize.height);
        if (top - labelSize.height < 0) { top = labelSize.height; }
        cv::rectangle(frame,
                      cv::Point(det.box.x, top - labelSize.height),
                      cv::Point(det.box.x + labelSize.width, top + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(det.box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
} 