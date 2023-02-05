#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QImage> 
#include <QFuture> 
#include <QtConcurrent> 
#include <atomic> 
#include "DetectionPipeline.h" 
#include <opencv2/videoio.hpp> 
#include <QSlider>  
#include <QHBoxLayout> 
#include <QStatusBar> 


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

signals:
    
    void frameReady(const QImage &frame, double fps);
    void processingError(const QString &errorMessage);

private slots:
    void startWebcam();
    void stopWebcam();
    void handleFrameProcessed(const QImage &frame, double fps);
    void handleProcessingError(const QString &errorMessage);
    void onConfidenceSliderChanged(int value);
    void onScoreSliderChanged(int value);
    void onNmsSliderChanged(int value);
    void togglePauseResume(); 

private:
    QWidget *centralWidget;
    QVBoxLayout *mainLayout;

    QLabel *videoDisplayLabel;
    QPushButton *startButton;
    QPushButton *stopButton;
    QPushButton* pauseResumeButton;
    QSlider* confidenceSlider;
    QLabel* confidenceLabel;
    QSlider* scoreSlider;
    QLabel* scoreLabel;
    QSlider* nmsSlider;
    QLabel* nmsLabel;

    std::atomic<bool> isProcessing;
    std::atomic<bool> isPaused; 
    cv::VideoCapture videoCapture; 
    QFuture<void> processingFuture; 

    DetectionPipeline pipeline;


    void setupUi();
    void runDetectionLoop(); 
    void setupSliderWidget(QHBoxLayout* layout, QLabel*& label, QSlider*& slider, const QString& labelPrefix, int initialValue, const char* slotName);


};

#endif // MAINWINDOW_H 