#include <stdio.h>
#include <dirent.h>
#ifdef __MINGW32__
#include <sys/stat.h>
#endif
#include <ios>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>

using namespace std;
using namespace cv;


static string posSamplesNormalDir = "pos/normal/";
static string posSamplesIlluminationDir = "pos/illumination/";

static string negSamplesWaterDir = "neg/water/";
static string negSamplesCoralsDir = "neg/corals/";
static string negSamplesAnimalsDir = "neg/sea_creatures/";
static string negSamplesHardNegativeDir = "neg/hard_negative/";
static string testPosImagesDir = "test_pos/";
static string testNegImagesDir = "test_neg/";

static string foundImagesDir ="found/";

static int hard_negative_start= 1;

static const Size trainingPadding1 = Size(0, 0);
static const Size winStride1 = Size(8, 8);
float posLabel = 1.0;
float negLabel = -1.0;

template <typename T>
string toString(T val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}


//extend CvSVM to get access to weights
class mySVM : public CvSVM
{
public:
    vector<float>
    getWeightVector(const int descriptorSize);
};

//get the weights
vector<float>
mySVM::getWeightVector(const int descriptorSize)
{
    vector<float> svmWeightsVec(descriptorSize+1);
    int numSupportVectors = get_support_vector_count();

    //this is protected, but can access due to inheritance rules 
    const CvSVMDecisionFunc *dec = CvSVM::decision_func;

    const float *supportVector;
    float* svmWeight = &svmWeightsVec[0];

    for (int i = 0; i < numSupportVectors; ++i)
    {
        float alpha = *(dec[0].alpha + i);
        supportVector = get_support_vector(i);
        for(int j=0;j<descriptorSize;j++)
        {
            *(svmWeight + j) += -alpha * *(supportVector+j);
        }
    }
    *(svmWeight + descriptorSize) = - dec[0].rho;

    return svmWeightsVec;
}


static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    } return t;
}

static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

static Mat rotate(Mat src, double angle)
{
    Mat dst;
    Point2f center(src.cols/2.0, src.rows/2.0);    
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    cv::Rect bbox = cv::RotatedRect(center,src.size(), angle).boundingRect();
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;
    warpAffine(src, dst, rot, bbox.size());
    return dst;
}

static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
#ifdef __MINGW32__
	struct stat s;
#endif
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
#ifdef __MINGW32__	
			stat(ep->d_name, &s);
			if (s.st_mode & S_IFDIR) {
				continue;
			}
#else
            if (ep->d_type & DT_DIR) {
                continue;
            }
#endif
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                //printf("Found matching data file '%s'\n", ep->d_name);
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

static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {

    Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);
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
    hog.compute(imageData, featureVector, winStride1, trainingPadding1, locations);
    imageData.release(); // Release the image again after features are extracted
}

static void showDetections(const vector<Point>& found, Mat& imageData) {
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Point r = found[i];
       rectangle(imageData, Rect(r.x-16, r.y-32, 32, 64), Scalar(64, 255, 64), 3);
    }
}

static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}

//go through one more SVM layer
static bool filterDetections(const HOGDescriptor& hog, Mat& imageData, Rect rect, mySVM& SVM, bool output){
    try{
        cv::Mat padded;

        //to handle seg fault when feature is detected at the edge
        int padding = 128;
        padded.create(imageData.rows + 2*padding, imageData.cols + 2*padding, imageData.type());
        padded.setTo(cv::Scalar::all(0));

        imageData.copyTo(padded(Rect(padding, padding, imageData.cols, imageData.rows)));
        //imwrite("found/features/padded.jpg", padded);

        Mat crop= padded(Rect(rect.x+128,rect.y+128,rect.width,rect.height));
        cv::resize(crop,crop,Size(64,128));
        imshow("Cropped",crop);
        vector<float> featureVector;
        featureVector.clear();
        vector<Point> locations;
        Mat feature_matrix;
        int dims=3780;

        GaussianBlur(crop,crop,Size(5,5),0,0);
        hog.compute(crop, featureVector,winStride1, trainingPadding1, locations);
        if(featureVector.size()==dims){
            feature_matrix= Mat(1,dims,CV_32FC1,featureVector.data(),true); 
            imwrite("found/features/feature.jpg", crop);
        }
        else{
            printf("<filterDetection> Error in feature detection\n" );
        }

        int result=SVM.predict(feature_matrix);
        if(result==posLabel){

            if(output){
                imwrite(negSamplesHardNegativeDir+toString(hard_negative_start)+".jpg", crop);
                ++hard_negative_start;
            }

            return true;
        }
        else{
            return false;
        }
    }
    catch(int e){
        printf("<filterDetection> Error caught.\n");
        return -1;
    }
}

static void detectTestSet(const HOGDescriptor& hog, const double hitThreshold, const std::vector<string>& testPosFileNames, const std::vector<string>& testNegFileNames, vector<int>& results, Size winStride, Size padding, double scale, mySVM& SVM){
    unsigned int truePositives =0;
    unsigned int falsePositives =0;
    unsigned int falseNegatives =0;
    unsigned int trueNegatives =0;
    vector<Rect> foundDetectionVertical;

    //Walk over positive test samples
    for (vector<string>::const_iterator testPosIterator = testPosFileNames.begin(); testPosIterator != testPosFileNames.end(); ++testPosIterator) {
        Mat imageData = imread(*testPosIterator, IMREAD_GRAYSCALE);
        const char *file_name= testPosIterator->c_str();
        vector<Rect> filtered_foundDetection;
        hog.detectMultiScale(imageData, foundDetectionVertical, hitThreshold, winStride, padding, scale);
        //GaussianBlur(imageData,imageData,Size(5,5),0,0);
        for (vector<Rect>::const_iterator rectIterator = foundDetectionVertical.begin(); rectIterator != foundDetectionVertical.end(); ++rectIterator){
            if(filterDetections(hog,imageData,*rectIterator, SVM, false)){
                //printf("Feature added\n");
                filtered_foundDetection.push_back(*rectIterator);
            }
        }
        if(filtered_foundDetection.size()>0){
            ++truePositives;
            showDetections(filtered_foundDetection, imageData);
            imwrite(foundImagesDir+file_name, imageData);
        }  
        else {
            ++falseNegatives;
            //print file name of the error
            printf("Failed on %s\n", file_name);
        }
    } 
    // //Walk over negative samples
    for (vector<string>::const_iterator testNegIterator = testNegFileNames.begin(); testNegIterator != testNegFileNames.end(); ++testNegIterator) {
        Mat imageData = imread(*testNegIterator, IMREAD_GRAYSCALE);
        const char *file_name= testNegIterator->c_str();
        vector<Rect> filtered_foundDetection;          
        hog.detectMultiScale(imageData, foundDetectionVertical, hitThreshold, winStride, padding, scale);
        //GaussianBlur(imageData,imageData,Size(5,5),0,0);
        for (vector<Rect>::const_iterator rectIterator = foundDetectionVertical.begin(); rectIterator != foundDetectionVertical.end(); ++rectIterator){
            if(filterDetections(hog,imageData,*rectIterator, SVM, false)){
                filtered_foundDetection.push_back(*rectIterator);
            }
        }
        if(filtered_foundDetection.size()>0){
            showDetections(filtered_foundDetection, imageData);
            ++falsePositives;
            imwrite(foundImagesDir+file_name, imageData);
            printf("Failed on %s\n", file_name);
        }
        else {
            ++trueNegatives;
        }        
    }
    results[0]=truePositives;
    results[1]=trueNegatives;
    results[2]=falsePositives;
    results[3]=falseNegatives;    
     printf("<HOG> Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
}

static void detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames, vector<int>& results) {
    unsigned int truePositives = 0;
    unsigned int trueNegatives = 0;
    unsigned int falsePositives = 0;
    unsigned int falseNegatives = 0;
    vector<Point> foundDetection;
    // Walk over positive training samples, generate images and detect
    for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
        const Mat imageData = imread(*posTrainingIterator, IMREAD_GRAYSCALE);
        //GaussianBlur(imageData,imageData,Size(5,5),0,0);
        hog.detect(imageData, foundDetection, hitThreshold, winStride1, trainingPadding1);
        if (foundDetection.size() > 0) {
            ++truePositives;
            falseNegatives += foundDetection.size() - 1;
        } else {
            ++falseNegatives;
        }
    }
    // Walk over negative training samples, generate images and detect
    for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
        const Mat imageData = imread(*negTrainingIterator, IMREAD_GRAYSCALE);
        //GaussianBlur(imageData,imageData,Size(5,5),0,0);
        hog.detect(imageData, foundDetection, hitThreshold, winStride1, trainingPadding1);
        if (foundDetection.size() > 0) {
            falsePositives += foundDetection.size();
        } else {
            ++trueNegatives;
        }        
    }
    results[0]=truePositives;
    results[1]=trueNegatives;
    results[2]=falsePositives;
    results[3]=falseNegatives;
    printf("<HOG> Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
}

static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData, mySVM& SVM) {
    vector<Rect> found_Vertical;
    vector<Rect> filtered_foundDetection;
    Size padding(Size(8, 8));
    Size winStride(Size(16, 16));
    //medianBlur(imageData,imageData,5);
    hog.detectMultiScale(imageData, found_Vertical, hitThreshold, winStride, padding,1.5);
    for (vector<Rect>::const_iterator rectIterator = found_Vertical.begin(); rectIterator != found_Vertical.end(); ++rectIterator){
        if(filterDetections(hog,imageData,*rectIterator, SVM, false)){
            filtered_foundDetection.push_back(*rectIterator);
        }
    }

    if(filtered_foundDetection.size()>0){
        showDetections(filtered_foundDetection, imageData);
    }

}

static void outputToFile(vector<int> trainingResults, vector<int> testResults, double hitThreshold, float cValue, float weight, Size winStride, Size padding, double scale ){
    ofstream myfile;
    myfile.open("genfiles/hog/detectMultiscale_withIllumination.txt");
    myfile<<"Different HOG results with varying hitThreshold and fixed C Value and class weights.\n";
    myfile<<"With an additional SVM layer after detectMultiscale.\n";
    myfile<<"C Value:\t"<<cValue<<"\nWeights: \t"<<weight<<"\t"<<(1-weight)<<"\nHit Threshold: \t"<<hitThreshold<<"\n\n";
    myfile<<"winStride:\t"<<winStride.height<<"\t"<<winStride.width<<"\nPadding: \t"<<padding.height<<"\t"<<padding.width<<"\nScale: \t\t"<<scale<<"\n\n";
    myfile<<"Training Results:\n\tTrue Positives: "<<trainingResults[0]<<"\n\tTrue Negatives: "<<trainingResults[1]<<"\n\tFalse Positives: "<<trainingResults[2]<<"\n\tFalse Negative: "<<trainingResults[3]<<"\n\n";
    myfile<<"Test Results:\n\tTrue Positives: "<<testResults[0]<<"\n\tTrue Negatives: "<<testResults[1]<<"\n\tFalse Positives: "<<testResults[2]<<"\n\tFalse Negative: "<<testResults[3]<<"\n\n";
    printf("%s\n", "File Outputed" );
    myfile.close();
}

int main(int argc, char** argv) {

    HOGDescriptor hog; // Use standard parameters here
    hog.winSize = Size(64, 128); // Default training images size as used in paper
    // Get the files to train from somewhere
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> testPosImagesSet;
    static vector<string> testNegImagesSet;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");
    validExtensions.push_back("jpeg");

    getFilesInDirectory(posSamplesNormalDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(posSamplesIlluminationDir, positiveTrainingImages, validExtensions);
    //getFilesInDirectory(posSamplesThinnerDir, positiveTrainingImages, validExtensions);
    //getFilesInDirectory(posSamplesFatterDir, positiveTrainingImages, validExtensions);

    getFilesInDirectory(negSamplesWaterDir, negativeTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesCoralsDir, negativeTrainingImages,validExtensions);
    getFilesInDirectory(negSamplesAnimalsDir, negativeTrainingImages, validExtensions);
    //getFilesInDirectory(negSamplesHardNegativeDir, negativeTrainingImages, validExtensions);

    getFilesInDirectory(testPosImagesDir, testPosImagesSet, validExtensions);
    getFilesInDirectory(testNegImagesDir, testNegImagesSet, validExtensions);
    /// Retrieve the descriptor vectors from the samples
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    
    // Make sure there are actually samples to train
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }

    /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
    setlocale(LC_ALL, "C"); // Do not use the system locale
    setlocale(LC_NUMERIC,"C");
    setlocale(LC_ALL, "POSIX");

    cv::Mat cSvmLabels;
    int dims=3780;
    float c=0.085;

    //Feature Matrix
    cv::Mat cSvmTrainingData;

    CvSVMParams params;
    Mat weights = (Mat_<double>(2,1)<<0.6,0.4);
	CvMat weight = weights;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.C = c;
	params.class_weights = &weight;
	
   
   	int positiveCount=0;
   	int i =0;
	for (vector<string>::const_iterator posTrainingIterator = positiveTrainingImages.begin(); posTrainingIterator != positiveTrainingImages.end(); ++posTrainingIterator){
		vector<float> featureVector;
		calculateFeaturesFromInput(*posTrainingIterator, featureVector, hog);
		if(featureVector.size()==dims){
			cSvmTrainingData.push_back(cv::Mat(1,dims,CV_32FC1,featureVector.data(),true));
			cSvmLabels.push_back(posLabel);	
			i+=1;
		}
		positiveCount+=1;
	}
	printf("%d:%s\n",i," positive features appended." );
	printf("%d\n",positiveCount );

	int negativeCount=0;
	int j=0;
	for (vector<string>::const_iterator negTrainingIterator = negativeTrainingImages.begin(); negTrainingIterator != negativeTrainingImages.end(); ++negTrainingIterator){
		vector<float> featureVector;
		calculateFeaturesFromInput(*negTrainingIterator, featureVector, hog);
		if(featureVector.size()==dims){
			cSvmTrainingData.push_back(cv::Mat(1,dims,CV_32FC1,featureVector.data(),true));
			cSvmLabels.push_back(negLabel);
			j+=1;	
		}
		negativeCount+=1;
	}
	printf("%d:%s\n",j,"negative features appended." );
	printf("%d\n",negativeCount );

    mySVM SVM;
    SVM.train(cSvmTrainingData, cSvmLabels, Mat(), Mat(), params);

 	SVM.save("genfiles/cvSVM/svmModel_moreData.yaml");

    vector<float> primal= SVM.getWeightVector(dims);
    hog.setSVMDetector(primal);
    const double hitThreshold= -0.3;
    Size winStride(Size(16,16));
    Size padding(Size(8,8));
    double scale= 1.5;

    vector<int> trainingResults (4);
    vector<int> testResults (4);

    printf("<HOG> Testing training phase using training set as test set.\n");
    detectTrainingSetTest(hog, hitThreshold, positiveTrainingImages, negativeTrainingImages,trainingResults);
    detectTestSet(hog, hitThreshold, testPosImagesSet, testNegImagesSet,testResults, winStride, padding, scale, SVM);
    outputToFile(trainingResults,testResults,hitThreshold,c,0.6,winStride, padding, scale);

    //Testing using video file
    VideoCapture cap("/home/mikacuy/Videos/object_detection.mp4");
    if(!cap.isOpened()) { // check if we succeeded
        printf("Error opening camera!\n");
        return EXIT_FAILURE;
    }
    Mat testImage;
    while ((cvWaitKey(10) & 255) != 27) {
        cap >> testImage; // get a new frame from camera
        cvtColor(testImage, testImage, CV_BGR2GRAY); // Work on grayscale images as trained
        detectTest(hog, hitThreshold, testImage,SVM);
        imshow("HOG custom detection", testImage);
    }


    // printf("Testing custom detection using camera\n");
    // VideoCapture cap(-1); // open the default camera
    // if(!cap.isOpened()) { // check if we succeeded
    //     printf("Error opening camera!\n");
    //     return EXIT_FAILURE;
    // }
    // Mat testImage;
    // while ((cvWaitKey(10) & 255) != 27) {
    //     cap >> testImage; // get a new frame from camera
    //     cvtColor(testImage, testImage, CV_BGR2GRAY); // Work on grayscale images as trained
    //     detectTest(hog, hitThreshold, testImage,SVM);
    //     imshow("HOG custom detection", testImage);
    // }
    // // </editor-fold>

    // return EXIT_SUCCESS;

}
