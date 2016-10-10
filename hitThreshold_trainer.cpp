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

static string testPosImagesDir = "pos_test_small/";
static string testNegImagesDir = "neg_test_small/";

static string foundImagesDir ="found/";


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


static void outputToFile(vector<int> trainingResults, vector<int> testResults, double hitThreshold, float cValue, float weight){
    ofstream myfile;
    myfile.open("genfiles/hitThreshold/with_illumination4.txt");
    myfile<<"Different HOG results with varying hitThreshold and fixed C Value and class weights.\n";
    myfile<<"C Value:\t"<<cValue<<"\nWeights: \t"<<weight<<"\t"<<(1-weight)<<"\nHit Threshold: \t"<<hitThreshold<<"\n\n";
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

    vector<float> primal= SVM.getWeightVector(dims);
    hog.setSVMDetector(primal);

    const double hitThreshold= -0.5;

    vector<int> trainingResults (4);
    vector<int> testResults (4);

    printf("<HOG> Testing training phase using training set as test set.\n");
    detectTrainingSetTest(hog, hitThreshold, positiveTrainingImages, negativeTrainingImages,trainingResults);
    detectTrainingSetTest(hog, hitThreshold, testPosImagesSet, testNegImagesSet, testResults);
    outputToFile(trainingResults,testResults,hitThreshold,c,0.7);

}
