#ifndef DETECTIONCONFIG_H
#define DETECTIONCONFIG_H

struct DetectionConfig {
    
    static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.40f;
    static constexpr float DEFAULT_SCORE_THRESHOLD = 0.50f;
    static constexpr float DEFAULT_NMS_THRESHOLD = 0.45f;

    static constexpr const char* DEFAULT_MODEL_PATH = "./data/yolov8n.onnx";
    static constexpr const char* DEFAULT_CLASS_NAMES_PATH = "./data/coco.names";
};

#endif // DETECTIONCONFIG_H 