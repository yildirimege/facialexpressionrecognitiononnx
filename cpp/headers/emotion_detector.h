#pragma once
#ifndef EMOTION_DETECTOR

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/videoio.hpp"

#include <thread>
#include <iostream>

class emotionRecognizer
{
private:
    std::string result = "";
    cv::Rect faceRect = cv::Rect();

public:
    double minVal = 0;
    double maxVal = INFINITY;
    cv::Point minLoc;
    cv::Point maxLoc;

    /**
     * \brief C-Tor
     * 
     */
    inline emotionRecognizer(){}

    /**
     * \brief D-Tor
     * 
     */
    inline ~emotionRecognizer(){}
    
    /**
     * .
     * \brief Setter for Emotion Result
     * \param result
     */
    void setResult(std::string& result)
    {
        this->result = result;
    }

    /**
     * .
     * \brief Getter for Emotion Result
     * \return 
     */
    std::string getResult()
    {
        return this->result;
    }

    /**
     * .
     * \brief Setter for FaceRect
     * \param faceRect
     */
    void setFaceRect(cv::Rect& faceRect)
    {
        this->faceRect = faceRect;
    }

    /**
     * .
     * \brief Getter for FaceRect
     * \return Face Rectangle of object
     */
    cv::Rect getFaceRect()
    {
        return this->faceRect;
    }

    /**
     * .
     * \brief
     * \param displayImage: Image that will be displayed to user.
     * \param frame: Processed frame for emotion recognition
     * \param faceDetector: OpenCV CascadeClassifier object 
     * \param faceRects: Vector of Rectangles that store information of ROI (Face)
     */
     void detectFacesAndCrop(cv::Mat& displayImage, cv::Mat& frame, cv::CascadeClassifier& faceDetector, std::vector<cv::Rect>& faceRects) // Thread 1
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        faceDetector.detectMultiScale(frame, faceRects, 1.1, 12, 0, cv::Size(150, 150));
        if (!faceRects.empty())
        {
            frame = frame(faceRects[0]);
            cv::rectangle(displayImage, faceRects[0], cv::Scalar(0, 255, 0));
        }
    }

    /**
     * .
     * \brief
     * \param emotionsVector: Constant Vector that stores emotions
     * \param resultMatrix: Matrix of Probabilities of each label
     * \param minVal: Minimum value in matrix
     * \param maxVal: Maximum value in matrix
     * \param minLoc: Location of minimum value
     * \param maxLoc: Location of maximum value
     */
    static void returnEmotionFromArray(const std::vector<std::string>& emotionsVector, cv::Mat& resultMatrix, 
        double& minVal, double& maxVal, cv::Point& minLoc, cv::Point& maxLoc, std::string& emotionResult)
    {
        cv::minMaxLoc(resultMatrix, &minVal, &maxVal, &minLoc, &maxLoc);
        emotionResult = emotionsVector.at(maxLoc.x);
    }

    /**
     * .
     * \brief Resizes the 
     * \param frame: Processed frame for emotion recognition
     */
    void preprocessFrame(cv::Mat& frame)
    {
        cv::resize(frame, frame, cv::Size(48, 48));
    }

    /**
     * .
     * \brief Recognizes the facial expression in given 48x48 Image/Matrix by cross-platform ONNX Model.
     * 
     * \param frame: Processed frame for emotion recognition
     * \param model: OpenCV Deep Learning Neural Network Object ( with Loaded ONNX model)
     * \param resultMatrix: Matrix of Probabilities of each label
     * \param emotionResult: String of the resulting emotion.
     */
    void predictEmotion(cv::Mat& frame, cv::dnn::Net model, cv::Mat& resultMatrix)
    {
        model.setInput(frame);
        resultMatrix = model.forward();
    }
};
#endif // !EMOTION_DETECTOR