// source file with main function.
// 
// Author: Ege Yýldýrým
// 
// !Note: This project is not built by CMake, Make etc.
// if you're trying to run his on Windows OS, please add
// %(OpenCV_DIR) to your environment path and specify
// %(OpenCV_DIR)/lib , %(OpenCV_DIR)/include paths too.


/* Standart C/C++ Headers */
#include <future>
#include <fstream>
#include <ctime>
#include <chrono>
#include <map>

/* Custom Header (Includes necessary OpenCV headers.)*/
#include "emotion_detector.h" 


const static int IMAGE_WIDTH = 48;
const static int IMAGE_HEIGHT = 48;

const static int VIDEO_WIDTH = 1280;
const static int VIDEO_HEIGHT = 720;


const std::string ONNX_MODEL_PATH = "model.onnx";
const std::string HAARCASCADE_PATH = "haarcascade_frontalface_default.xml";


const std::vector<std::string> emotionVector = { "Angry", "Disgust",
                                "Fear", "Happy",
                                "Sad", "Surprise",
                                "Neutral" };


/**
 * \brief Loads ONNX file into cv::dnn::Net Object.
 * 
 * \param onnxModelPath
 * \return ONNX Loaded cv::dnn::Net Object
 */
cv::dnn::Net loadModel(std::string onnxModelPath)
{
    return cv::dnn::readNetFromONNX(onnxModelPath);
}

/**
 * \brief: Writes emotion, rectangle and current time & date to csv file seperated by comma.
 * 
 * \param emotion: Current emotion object which stores rectangle and current frame's emotion.
 * \param csvFile: csv loaded std::ofstream file .
 */

void writeToCSV(emotionRecognizer& emotion, std::ofstream& csvFile)
{
    std::time_t now = time(0);
    char str[26];

    ctime_s(str, sizeof str, &now); // Thread secured ctime.

    csvFile << emotion.getResult() << "," << emotion.getFaceRect() << "," << str << "\n"; // Writing to CSV file.
}

int main()
{
    std::string emotionResult;

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

    std::ofstream csvFile;

    csvFile.open("emotions.csv");

    //Emotion Recognizer object which stores face rectangle and emotion for each detected face.
    emotionRecognizer recognizer;

    // Initializing all cv::Mat objects. There will not be any other created cv::Mat.
    // whole program works with refferences to reduce time complexity by not copying any object.
    // also way reduced space complexity in execution time.
    cv::Mat mainFrame;
    cv::Mat displayedImage;
    cv::Mat processedFrame;
    cv::Mat emotionResultsMatrix;

    std::vector<cv::Rect> faceRects;

    cv::CascadeClassifier faceCascade;

    // Video Capture object, change 0 to %(FILE_PATH) to read from video.
    cv::VideoCapture cap(0);

    //Loading CascadeClassifier and cv::dnn::Net models.
    faceCascade.load(HAARCASCADE_PATH);
    cv::dnn::Net model = loadModel(ONNX_MODEL_PATH);


    while (cap.isOpened())
    {
        

        cap.read(mainFrame);
        mainFrame.copyTo(displayedImage);

        // Starting counting time for each frame taken from camera
        auto start = std::chrono::high_resolution_clock::now();

        // Asynchron threads are used to block future processes until current one finishes
        // This is used to avoid buffer overflow (Assures that another frame will not be taken from camera until
        // desired processes are finished.)
        std::async(std::launch::async, [&] {
            recognizer.detectFacesAndCrop(displayedImage, mainFrame, faceCascade, faceRects);
            });

        // Guard for continuing the program en if no faces are detected.
        if (!faceRects.empty())
        {
            std::async(std::launch::async, [&] {
                recognizer.preprocessFrame(mainFrame);
                });
            

            std::async(std::launch::async, [&] {
                recognizer.predictEmotion(mainFrame, model, emotionResultsMatrix);
                });

            std::async(std::launch::async, [&] {
                recognizer.returnEmotionFromArray(emotionVector, emotionResultsMatrix, minVal, maxVal, minLoc, maxLoc, emotionResult);
                });

            recognizer.setResult(emotionResult);
            recognizer.setFaceRect(faceRects[0]);
            cv::putText(displayedImage, emotionResult, cv::Point(faceRects[0].x, faceRects[0].y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0));
        }
        //writeToCSV(recognizer, csvFile);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Total time taken for single frame " << duration.count() << "\n";
        cv::imshow("Main", displayedImage);
        cv::waitKey(1);
    }     
}
   
    
