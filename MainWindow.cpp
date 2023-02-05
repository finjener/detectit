#include "MainWindow.h"
#include "DetectionPipeline.h"
#include "DetectionConfig.h"

#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QtConcurrent>
#include <QThread>
#include <QSlider>
#include <QHBoxLayout>
#include <QStatusBar>

#include <opencv2/videoio.hpp> 
#include <opencv2/imgproc.hpp> 


void MainWindow::handleFrameProcessed(const QImage &frame, double fps) {
    
    if (!isProcessing) return; 
    if (frame.isNull()) {
        qWarning() << "GUI received null frame.";
        return;
    }
    videoDisplayLabel->setPixmap(QPixmap::fromImage(frame).scaled(videoDisplayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    
    statusBar()->showMessage(QString("Processing... FPS: %1").arg(QString::number(fps, 'f', 1)));
}

void MainWindow::handleProcessingError(const QString &errorMessage) {
    
    if (!isProcessing) return; 
    QMessageBox::critical(this, "Processing Error", errorMessage);
    isProcessing = false; 
    isPaused = false; 
    startButton->setEnabled(true);
    stopButton->setEnabled(false);
    pauseResumeButton->setEnabled(false);
    pauseResumeButton->setText("Pause");
    videoDisplayLabel->setText("Error occurred. Webcam stopped.");
    videoDisplayLabel->setPixmap(QPixmap());
    videoDisplayLabel->setStyleSheet("QLabel { background-color : black; color : gray; }");
    statusBar()->showMessage(QString("Error: %1").arg(errorMessage), 5000); // Show error for 5s
}

void MainWindow::onConfidenceSliderChanged(int value) {
    float floatValue = static_cast<float>(value) / 100.0f;
    confidenceLabel->setText(QString("Confidence: %1").arg(QString::number(floatValue, 'f', 2)));
    pipeline.setConfidenceThreshold(floatValue);
}

void MainWindow::onScoreSliderChanged(int value) {
    float floatValue = static_cast<float>(value) / 100.0f;
    scoreLabel->setText(QString("Score Thresh: %1").arg(QString::number(floatValue, 'f', 2)));
    pipeline.setScoreThreshold(floatValue);
}

void MainWindow::onNmsSliderChanged(int value) {
    float floatValue = static_cast<float>(value) / 100.0f;
    nmsLabel->setText(QString("NMS Thresh: %1").arg(QString::number(floatValue, 'f', 2)));
    pipeline.setNmsThreshold(floatValue);
}

void MainWindow::togglePauseResume() {
    isPaused = !isPaused;
    if (isPaused) {
        pauseResumeButton->setText("Resume");
        statusBar()->showMessage("Paused.");
        qDebug() << "Process paused.";
    } else {
        pauseResumeButton->setText("Pause");
        qDebug() << "Process resumed.";
    }
}


void MainWindow::startWebcam()
{
    if (isProcessing) {
        qWarning() << "Process already in progress.";
        return;
    }
    isProcessing = true;
    isPaused = false;
    startButton->setEnabled(false);
    stopButton->setEnabled(true);
    pauseResumeButton->setEnabled(true);
    pauseResumeButton->setText("Pause");
    videoDisplayLabel->setText("Starting...");
    videoDisplayLabel->setStyleSheet("QLabel { background-color : black; color : white; }");
    statusBar()->showMessage("Starting process and camera...");

    qDebug() << "Launching detection...";

    processingFuture = QtConcurrent::run([this]() { 
        this->runDetectionLoop(); 
    });
}

void MainWindow::stopWebcam()
{
    if (!isProcessing) {
        return;
    }
    qDebug() << "Requesting detection loop to stop...";
    isProcessing = false; 
    isPaused = false; 
    startButton->setEnabled(true);
    stopButton->setEnabled(false);
    pauseResumeButton->setEnabled(false);
    pauseResumeButton->setText("Pause");
    videoDisplayLabel->setText("Stopping..."); 
    statusBar()->showMessage("Stopped.", 2000); // Show stopped for 2s
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      isProcessing(false),
      isPaused(false)
{
    setupUi(); 

    
    connect(this, &MainWindow::frameReady, this, &MainWindow::handleFrameProcessed, Qt::QueuedConnection);
    connect(this, &MainWindow::processingError, this, &MainWindow::handleProcessingError, Qt::QueuedConnection);


    onConfidenceSliderChanged(confidenceSlider->value());
    onScoreSliderChanged(scoreSlider->value());
    onNmsSliderChanged(nmsSlider->value());

    stopButton->setEnabled(false);
    pauseResumeButton->setEnabled(false);
}


MainWindow::~MainWindow()
{
    qDebug() << "Main window destructor called.";
    isProcessing = false; 
    
    qDebug() << "Waiting for background task to finish...";
    
    if (processingFuture.isValid()) { 
         processingFuture.waitForFinished();
         qDebug() << "Background task finished.";
    } else {
         qDebug() << "Background task was not running.";
    }

    if (videoCapture.isOpened()) { 
        videoCapture.release();
    }
}


void MainWindow::setupUi()
{
    setWindowTitle("DetectIt");
    resize(800, 780); 
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    mainLayout = new QVBoxLayout(centralWidget);
    videoDisplayLabel = new QLabel("No video feed", this);
    videoDisplayLabel->setAlignment(Qt::AlignCenter);
    videoDisplayLabel->setMinimumSize(640, 480);
    videoDisplayLabel->setFrameShape(QFrame::Box);
    videoDisplayLabel->setStyleSheet("QLabel { background-color : black; color : gray; }");
    mainLayout->addWidget(videoDisplayLabel);

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    startButton = new QPushButton("Start Webcam", this);
    stopButton = new QPushButton("Stop Webcam", this);
    pauseResumeButton = new QPushButton("Pause", this);
    buttonLayout->addWidget(startButton);
    buttonLayout->addWidget(stopButton);
    buttonLayout->addWidget(pauseResumeButton);
    buttonLayout->addStretch();
    mainLayout->addLayout(buttonLayout);

    QHBoxLayout *confSliderLayout = new QHBoxLayout();
    setupSliderWidget(confSliderLayout, confidenceLabel, confidenceSlider, "Confidence", 
                      static_cast<int>(DetectionConfig::DEFAULT_CONFIDENCE_THRESHOLD * 100), SLOT(onConfidenceSliderChanged(int)));
    mainLayout->addLayout(confSliderLayout);

    QHBoxLayout *scoreSliderLayout = new QHBoxLayout();
    setupSliderWidget(scoreSliderLayout, scoreLabel, scoreSlider, "Score Thresh", 
                      static_cast<int>(DetectionConfig::DEFAULT_SCORE_THRESHOLD * 100), SLOT(onScoreSliderChanged(int)));
    mainLayout->addLayout(scoreSliderLayout);

    QHBoxLayout *nmsSliderLayout = new QHBoxLayout();
    setupSliderWidget(nmsSliderLayout, nmsLabel, nmsSlider, "NMS Thresh", 
                      static_cast<int>(DetectionConfig::DEFAULT_NMS_THRESHOLD * 100), SLOT(onNmsSliderChanged(int)));
    mainLayout->addLayout(nmsSliderLayout);


    connect(startButton, &QPushButton::clicked, this, &MainWindow::startWebcam);
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopWebcam);
    connect(pauseResumeButton, &QPushButton::clicked, this, &MainWindow::togglePauseResume);


    statusBar();
}


void MainWindow::setupSliderWidget(QHBoxLayout* layout, QLabel*& label, QSlider*& slider, const QString& labelPrefix, int initialValue, const char* slotName) {

    label = new QLabel(QString("%1: %2").arg(labelPrefix).arg(QString::number(static_cast<float>(initialValue)/100.0f, 'f', 2)), this);
    

    slider = new QSlider(Qt::Horizontal, this);
    slider->setRange(0, 100); 
    slider->setValue(initialValue);
    slider->setTickInterval(10);
    slider->setTickPosition(QSlider::TicksBelow);


    layout->addWidget(label);
    layout->addWidget(slider);

    connect(slider, SIGNAL(valueChanged(int)), this, slotName);
}

void MainWindow::runDetectionLoop()
{
    qDebug() << "Detection starting:" << QThread::currentThreadId();
    cv::Mat currentFrame; 
    double currentFps = 0.0;

    if (!pipeline.initialize()) { 
        emit processingError("Failed to initialize detection.");
        isProcessing = false; 
        return; 
    }
    if (!videoCapture.open(0)) { 
        emit processingError("Could not open webcam.");
        isProcessing = false;
        return;
    }


    while (isProcessing) {
        
        while(isPaused && isProcessing) { 
            QThread::msleep(100); 
        }
        if (!isProcessing) break;

        if (!videoCapture.isOpened()) {
            if(isProcessing) emit processingError("Webcam became unavailable.");
            break; 
        }

        if (!videoCapture.read(currentFrame)) { 
             if(isProcessing) qWarning() << "Failed to capture frame.";
             QThread::msleep(20); 
             continue; 
        }
        if (currentFrame.empty()) { 
            continue;
        }


        cv::Mat annotatedFrame = pipeline.processFrame(currentFrame, currentFps);

        if (!isProcessing || annotatedFrame.empty()) {
             if(annotatedFrame.empty() && isProcessing) {
                 qWarning() << "Processing failed for a frame."; 
             }
             break; 
        }

        cv::Mat rgbFrame;
        cv::cvtColor(annotatedFrame, rgbFrame, cv::COLOR_BGR2RGB);
        QImage qimg(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, static_cast<int>(rgbFrame.step), QImage::Format_RGB888);

        if (!qimg.isNull()) {
            emit frameReady(qimg.copy(), currentFps); 
        }
    }

    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
    qDebug() << "Detection finished:" << QThread::currentThreadId();
}

