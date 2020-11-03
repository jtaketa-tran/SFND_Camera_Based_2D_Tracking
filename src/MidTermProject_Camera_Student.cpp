/* +---------------------------+ */
/* | INCLUDES FOR THIS PROJECT | */
/* +---------------------------+ */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <boost/circular_buffer.hpp>
//#include <boost/circular_buffer/base.hpp>
//#include <boost/circular_buffer/space_optimized.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* +--------------+ */
/* | MAIN PROGRAM | */
/* +--------------+ */
int main(int argc, const char *argv[])
{
    /* +------------------------------------+ */
    /* | INIT VARIABLES AND DATA STRUCTURES | */
    /* +------------------------------------+ */

    // detectors and descriptors to loop over
    const int NUM_DET_TYPES = 7;
    const int NUM_DESC_TYPES = 6;
    const char *DetectorTypes[NUM_DET_TYPES] = {"SIFT", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SHITOMASI"};
    const char *DescriptorTypes[NUM_DESC_TYPES] = {"SIFT", "AKAZE", "BRIEF", "ORB", "FREAK", "BRISK"};

    // matcher and selectors
    string matcherType  = "BF_MATCH"; // BF_MATCH, FLANN_MATCH
    string selectorType = "KNN";      // NN, KNN
    
    // cropping options
    bool cropImage = false; // crop image to preceding vehicle (plus a frame buffer)
    bool cropImage4PlotOnly = true; // for plotting keypoints (not actually cropping the image)
    bool bFocusOnVehicle = true; // shortcut for this study only: discard any keypoints outside a bounding box on the preceding vehicle
    
    // optional: limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false; int maxKeypoints = 50; 
    
    // viewing options
    bool visKeypoints = false; // run routine to visualize and/or save keypoints overlaid on car cookie cuts without option to writeImages
    bool visMatches   = false; // run routine to visualize and/or save adjacent frame keypoint matches with option to writeImages
    bool writeImages  = false; // save images with keypoint and/or match overlays to files
    bool plotImages   = false; // visualize images with keypoint and/or match overlays during run time

    // scoring output file
    bool writeScoring = false; // write performance metrics to an output file for scoring
    string scoringFileName = "cookieCutTimingBF.csv";
    bool writeKPs = false; // write keypoint parameters to a file
    
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    for (int detIdx = 0; detIdx < NUM_DET_TYPES; detIdx++)
    {
        string detectorType = DetectorTypes[detIdx];
        int firstTime = 1; // so that we only plot keypoints once
        int firstTimeCurrFrame = 1;
        for (int descIdx = 0; descIdx < NUM_DESC_TYPES; descIdx++)
        {            
            string descriptorType = DescriptorTypes[descIdx];

            // skip incompatible combinations:
            // - AKAZE descriptors can only be computed on KAZE or AKAZE detectors
            // - ORB descriptors cannot be computed on SIFT detectors
            if (descriptorType == "AKAZE" && detectorType != "AKAZE")
                continue;
            if (descriptorType == "ORB" && detectorType == "SIFT")
                continue;

            // create ring buffer for video frames
            int dataBufferSize = 2; // number of consecutive images that will be held in memory (ring buffer)
            boost::circular_buffer<DataFrame> dataBuffer;
            dataBuffer.set_capacity(dataBufferSize);

            /* +---------------------------+ */
            /* | MAIN LOOP OVER ALL IMAGES | */
            /* +---------------------------+ */
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {        
                if (writeScoring)
                {
                    ofstream logfile(scoringFileName, ios_base::app);
                    logfile << imgIndex << ", ";
                    logfile.close();
                }
                /* +------------------------+ */
                /* | LOAD IMAGE INTO BUFFER | */
                /* +------------------------+ */
                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                ostringstream imgIdx;
                imgIdx << imgStartIndex + imgIndex;
                string delim = "_";
                string imgKpFile = "KeypointsDefault_" + detectorType + delim + imgIdx.str() + imgFileType;
                string imgMatchFile = "MatchesFLANN_" + detectorType + delim + descriptorType + delim + matcherType + delim + selectorType + delim + imgIdx.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                if (cropImage)
                {
                    cv::Rect roi(510, 165, 235, 200); // top left x, top left y, width, height
                    imgGray = imgGray(roi);
                }

                // push images into ring buffer of size dataBufferSize = 2           
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataBuffer.push_back(frame); // push image into data frame buffer

                /* +------------------------+ */
                /* | DETECT IMAGE KEYPOINTS | */
                /* +------------------------+ */
                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image   
                detKeypoints(keypoints, imgGray, detectorType, writeScoring, scoringFileName);

                // only keep keypoints on the preceding vehicle
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle && !cropImage)
                {
                    vector<cv::KeyPoint> vehicleKeypoints;
                    for (auto kp : keypoints)
                    {
                        if (vehicleRect.contains(kp.pt))
                            vehicleKeypoints.push_back(kp);
                    }
                    keypoints = vehicleKeypoints;
                }
                
                // Optional: Limit number of keypoints (helpful for debugging and learning)
                if (bLimitKpts)
                {
                    // there is no response info, so keep the first maxKeypoints as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());                
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // Visualize Keypoint Detections
                if (visKeypoints && firstTime)
                {
                    firstTime == 0; // don't plot keypoints again
                    cv::Mat visImage = imgGray.clone();
                    //cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
                    
                    if (cropImage4PlotOnly)
                    {
                        //cv::Rect roi(510, 165, 235, 200); // top left x, top left y, width, height
                        cv::Rect roi(535, 180, 180, 150);
                        visImage = visImage(roi);
                    }
                    
                    if (plotImages)
                    {
                        string windowName = detectorType;
                        cv::namedWindow(windowName, 6);
                        imshow(windowName, visImage);
                        cv::waitKey(0);
                    }

                    if (writeImages)
                        bool check = imwrite(imgKpFile,visImage);                    
                }
    
                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;

                // record keypoint parameters for each detector type
                if (writeKPs && firstTimeCurrFrame)
                {
                    string kpStatsFile = "keypointCharacter_" + detectorType + ".csv";
                    ofstream keypointFile(kpStatsFile, ios_base::app);

                    if (imgIndex == 0)
                        keypointFile << "Frame, X, Y, Diameter, Orientation, Strength, Octave" << std::endl;

                    for (auto kp : keypoints)
                    {
                        std::cout << detectorType << " " << imgIndex << " " << kp.pt.x << " " << kp.pt.y << " " << kp.size << " " << kp.angle << " " << kp.response << " " << kp.octave << std::endl;
                        keypointFile << imgIndex << ", " << kp.pt.x << ", " << kp.pt.y << ", " << kp.size << ", " << kp.angle << ", " << kp.response << ", " << kp.octave << std::endl;
                    }
                    keypointFile.close();

                    if (imgIndex == imgEndIndex - imgStartIndex)
                        firstTimeCurrFrame = 0;
                }

                /* +------------------------------+ */
                /* | EXTRACT KEYPOINT DESCRIPTORS | */
                /* +------------------------------+ */
                cv::Mat descriptors;
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, writeScoring, scoringFileName);

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                /* +----------------------------+ */
                /* | MATCH KEYPOINT DESCRIPTORS | */
                /* +----------------------------+ */
                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    vector<cv::DMatch> matches;
                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                     (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                      matches, descriptorType, matcherType, selectorType, writeScoring, scoringFileName);

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    // visualize matches between current and previous image
                    if (visMatches)
                    {                
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        if (cropImage4PlotOnly)
                        {
                            cv::Rect roi(510, 165, (imgGray.size().width-510)+510+235, 200); // top left x, top left y, width, height
                            matchImg = matchImg(roi);
                        }

                        if (plotImages)
                        {
                            string windowName = "Matching keypoints between two camera images";
                            cv::namedWindow(windowName, 7);
                            cv::imshow(windowName, matchImg);
                            //cout << "\nPress key to continue to next image" << endl;
                            cv::waitKey(0); // wait for key to be pressed
                        }

                        if (writeImages)
                            bool check = imwrite(imgMatchFile,matchImg);                        
                    }
                }

                if (dataBuffer.size() == 1 && writeScoring) //if this is the first frame
                {
                    // Put in stub for frame 0 match logging for file uniformity
                    ofstream logfile(scoringFileName, ios_base::app);
                    logfile << matcherType << ", " << selectorType << ", 0, 0, 0.00000, 0.00000" << endl;
                    logfile.close();
                }
        
            } // end loop over all images

        } // end loop over descriptor types

    } // end loop over detector types

    return 0;
}
