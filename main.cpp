/**
 * @file:   main.cpp
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 * @date:   Created on 2. Dezember 2012
 * @brief:  Example on how to train your custom HOG detecting vector
 * for use with openCV <code>hog.setSVMDetector(_descriptor)</code>;
 * 
 * For the paper regarding Histograms of Oriented Gradients (HOG), @see http://lear.inrialpes.fr/pubs/2005/DT05/
 * You can populate the positive samples dir with files from the INRIA person detection dataset, @see http://pascal.inrialpes.fr/data/human/
 * 
 * What this program basically does:
 * 1. Read positive and negative training sample image files from specified directories
 * 2. Calculate their HOG features and keep track of their classes (pos, neg)
 * 3. Pass the features and their classes to a machine learning algorithm, e.g. SVMlight (@see http://svmlight.joachims.org/)
 * 4. Use the calculated support vectors and SVM model to calculate a single detecting descriptor vector
 * 
 * Build by issuing:
 * g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF main.o.d -o main.o main.cpp
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_learn.o.d -o svmlight/svm_learn.o svmlight/svm_learn.c
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_hideo.o.d -o svmlight/svm_hideo.o svmlight/svm_hideo.c
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_common.o.d -o svmlight/svm_common.o svmlight/svm_common.c
 * g++ `pkg-config --cflags opencv` -o opencvhogtrainer main.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`
 * 
 * Terms of use:
 * This program is to be used as an example and is provided on an "as-is" basis without any warranties of any kind, either express or implied.
 * Use at your own risk.
 * For used third-party software, refer to their respective terms of use and licensing
 */

#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "svmlight/svmlight.h"

using namespace std;
using namespace cv;

/* Parameter definitions */

// Directory containing positive sample images
static string posSamplesDir = "pos/";
// Directory containing negative sample images
static string negSamplesDir = "neg/";
// Set the file to write the features to
static string featuresFile = "genfiles/features.dat";
// Set the file to write the SVM model to
static string svmModelFile = "genfiles/svmlightmodel.dat";
// Set the file to write the resulting detecting descriptor vector to
static string descriptorVectorFile = "genfiles/descriptorvector.dat";

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(32, 32);
static const Size winStride = Size(8, 8);

/* Helper functions */

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

/**
 * Saves the given descriptor vector to a file
 * @param descriptorVector the descriptor vector to save
 * @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
 * @param fileName
 * @TODO Use _vectorIndices to write correct indices
 */
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
    printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
    /// @WARNING: This is really important, ROS seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
    setlocale(LC_ALL, "C"); // Do not use the system locale
    string separator = " "; // Use blank as default separator between single features
    fstream File;
    float percent;
    File.open(fileName.c_str(), ios::out);
    if (File.is_open()) {
        //        File << "# This file contains the trained descriptor vector" << endl;
        printf("Saving descriptor vector features:\t");
//            printf("\n#features %d", descriptorVector->size());
        storeCursor();
        for (int feature = 0; feature < descriptorVector.size(); ++feature) {
            if ((feature % 10 == 0) || (feature == (descriptorVector.size()-1)) ) {
                percent = ((1 + feature) * 100 / descriptorVector.size());
                printf("%4u (%3.0f%%)", feature, percent);
                fflush(stdout);
                resetCursor();
            }
            //                File << _vectorIndices->at(feature) << ":";
            File << descriptorVector.at(feature) << separator;
        }
        printf("\n");
        File << endl;
        File.flush();
        File.close();
    }
}

/**
 * Saves the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
 * @param descr feature-vectors of training samples
 * @param classBelonging teacher, desired output
 * @param saveToFile filename to save to
 */
static void saveFeatureVectorsInSVMLightCompatibleFormat(const Mat& descr, const Mat& classBelonging, const string& saveToFile) {
    if (descr.rows != classBelonging.rows) {
        printf("Error: Dimensions of training samples (%u) do not match classes vector (%u)!\n", descr.rows, classBelonging.rows);
        exit(EXIT_FAILURE);
    }
    /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
    setlocale(LC_ALL, "C"); // Do not use the system locale
    setlocale(LC_NUMERIC,"C");
    setlocale(LC_ALL, "POSIX");

    float percent;
    fstream File;
    File.open(saveToFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
//        File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << saveToFile.c_str() << endl;
        int sampleClass = 0;
        printf("Saving %d samples:\t\t", descr.rows);
        storeCursor();
        for (int sample = 1; sample <= descr.rows; ++sample) {
            if ( sample % 10 == 0 || sample == descr.rows) {
                percent = (sample * 100 / descr.rows);
                printf("%5u (%3.0f%%)", sample, percent);
                fflush(stdout);
                resetCursor();
            }
            // Convert positive class to +1. and negative class to -1. for svmlight
            sampleClass = (((int)classBelonging.at<float>(sample-1, 0) == 1) ? +1 : -1);
            if (sampleClass == 1) {
                File << "+1"; // +1
            } else {
                File << "-1"; // -1
            }
            for (int feature = 0; feature < descr.cols; ++feature) {
                File << " " << (feature + 1) << ":" << descr.at<float>(sample-1, feature);
            }
            if (sample != descr.rows) {
                File << endl;
            }
        }
        printf("\n");
        File.flush();
        File.close();
    } else {
        printf("Error opening file '%s'!\n", saveToFile.c_str());
    }
}

/**
 * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
 * @param dirName
 * @param fileNames found file names in specified directory
 * @param validExtensions containing the valid file extensions for collection in lower case
 * @return 
 */
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

/**
 * This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() functio
 * @param imageFilename
 * @param descriptorVector
 * @param hog
 */
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
//    printf("Reading image file '%s'\n", imageFilename.c_str());
    /** for imread flags from openCV documentation, 
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. imread),
     * you either do not have a current openCV version (>2.0) 
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
    Mat imageData(imread(imageFilename, 0));
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
//    imageData.release(); // Release the image again after features are extracted
}

/**
 * Main program entry point
 * @param argc
 * @param argv
 * @return EXIT_SUCCESS (0) or EXIT_FAILURE (1)
 */
int main(int argc, char** argv) {

    // <editor-fold defaultstate="collapsed" desc="Init">
    HOGDescriptor hog; // Use standard parameters here
    // Get the files to train from somewhere
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Read image files">
    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
    /// Retrieve the descriptor vectors from the samples
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    // </editor-fold>
    
    // <editor-fold defaultstate="collapsed" desc="Calculate HOG features">
    // Calculating one single HOG feature vector to determine the length of the vector
    vector<float> tmpFeatureVector;
    assert(positiveTrainingImages.size() > 0);
    calculateFeaturesFromInput(positiveTrainingImages.front(), tmpFeatureVector, hog);

    Mat trainingSamplesFeatures = Mat(0, tmpFeatureVector.size(), cv::DataType<float>::type);
    Mat trainingSamplesClasses = Mat(0, 1, cv::DataType<float>::type);
    
    try {
        trainingSamplesFeatures.reserve(overallSamples);
        trainingSamplesClasses.reserve(overallSamples);
    } catch (length_error& le) { // If there are too many training samples for our poor PC
        printf ("Length error, reserving space for Mat of size %lu failed: %s\n", overallSamples, le.what());
        exit(EXIT_FAILURE);
    }

    unsigned long currentFile = 0;
    float percent;
    printf("Reading files:\t");
    // positive samples
    for (vector<string>::const_iterator samplesIt = positiveTrainingImages.begin(); samplesIt != positiveTrainingImages.end(); ++samplesIt) {
        vector<float> featureVector;
        storeCursor();
        ++currentFile;
        if ( currentFile % 10 == 0 || currentFile == overallSamples) {
            percent = (currentFile * 100 / overallSamples);
            printf("%5lu (%3.0f%%)", currentFile, percent);
            fflush(stdout);
            resetCursor();
        }

        calculateFeaturesFromInput(*samplesIt, featureVector, hog);
        if (!featureVector.empty()) {
            trainingSamplesFeatures.push_back(Mat(Mat(featureVector).t()));
            trainingSamplesClasses.push_back<float>( 1.);

        }
    }
    // negative samples
    for (vector<string>::const_iterator samplesIt = negativeTrainingImages.begin(); samplesIt != negativeTrainingImages.end(); ++samplesIt) {
        vector<float> featureVector;
        
        storeCursor();
        ++currentFile;
        if ( currentFile % 10 == 0 || currentFile == overallSamples) {
            percent = (currentFile * 100 / overallSamples);
            printf("%5lu (%3.0f%%)", currentFile, percent);
            fflush(stdout);
            resetCursor();
        }

        calculateFeaturesFromInput(*samplesIt, featureVector, hog);
        if (!featureVector.empty()) {
            trainingSamplesFeatures.push_back(Mat(Mat(featureVector).t()));
            trainingSamplesClasses.push_back<float>(-1.);
        }
    }
    printf("\n");
    // Make sure there are as many training samples as there are class correspondences
    assert(trainingSamplesClasses.rows == trainingSamplesFeatures.rows);
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Pass features to machine learning algorithm">
    /** @TODO Avoid detour via file system, inject feature vectors directly into SVMlight */
    printf("\nSaving extracted calculated features of samples with class to file '%s'\n", featuresFile.c_str());
    saveFeatureVectorsInSVMLightCompatibleFormat(trainingSamplesFeatures, trainingSamplesClasses, featuresFile);

    /// Read in and train the calculated feature vectors with e.g. SVMlight, @see http://svmlight.joachims.org/
    printf("Passing feature vectors to SVMlight (This can take quite some while!)\n");
    SVMlight::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
    SVMlight::getInstance()->train(); // Call the core libsvm training procedure
    printf("Training done, saving model file!\n");
    SVMlight::getInstance()->saveModelToFile(svmModelFile);
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Generate single detecting feature vector from calculated SVM support vectors and SVM model">
    printf("Generating representative single HOG feature vector using svmlight!\n");
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    SVMlight::getInstance()->retrieveSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Clean up">
    trainingSamplesFeatures.release();
    trainingSamplesClasses.release();
    // </editor-fold>

    return EXIT_SUCCESS;
}
