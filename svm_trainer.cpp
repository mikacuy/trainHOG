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


static string posSamplesDir = "pos/";
static string negSamplesWaterDir = "neg/water/";
static string negSamplesCoralsDir = "neg/corals/";
static string negSamplesAnimalsDir = "neg/sea_creatures/";
static string testPosImagesDir = "pos_test_small/";
static string testNegImagesDir = "neg_test_small/";
static string featuresFile = "genfiles/features_trainingSet2.dat";
static string svmModelFile = "genfiles/svmlightmodel_trainingSet2.dat";
static string descriptorVectorFile = "genfiles/descriptorvector_trainingSet2.dat";
static string cvHOGFile = "genfiles/cvHOGClassifier_trainingSet2.yaml";
static string foundImagesDir ="found/";

static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);

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
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}

static void showDetections(const vector<Point>& found, Mat& imageData) {
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Point r = found[i];
       rectangle(imageData, Rect(r.x-16, r.y-32, 32, 64), Scalar(64, 255, 64), 3);
    }
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

    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesWaterDir, negativeTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesCoralsDir, negativeTrainingImages,validExtensions);
    getFilesInDirectory(negSamplesAnimalsDir, negativeTrainingImages, validExtensions);
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

    float posLabel = 1.0;
    float negLabel = -1.0;
    cv::Mat cSvmLabels;
    int dims=3780;
    //Feature Matrix
    cv::Mat cSvmTrainingData;

    CvSVMParams params;
    Mat weights = (Mat_<double>(2,1)<<0.7,0.3);
	CvMat weight = weights;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.C = 0.05;
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

	//for outputing to a file
	ofstream myfile;
  	myfile.open ("genfiles/cvSVM/Cvalues_results5.txt");
  	myfile << "Different results with varying C values in SVM model.\n";
  	myfile << "Class weights: 0.7, 0.3.\n\n";
 	static vector<float> cValues;
 	cValues.push_back(100.0);
 	cValues.push_back(1.0);
 	cValues.push_back(0.2);
 	cValues.push_back(0.15);
 	cValues.push_back(0.1);
 	cValues.push_back(0.09);
 	cValues.push_back(0.08);
 	cValues.push_back(0.075);
 	cValues.push_back(0.07);
 	cValues.push_back(0.065);
 	cValues.push_back(0.05);
 	cValues.push_back(0.04);
 	cValues.push_back(0.02);


 	CvSVM SVM;

 	for (vector<float>::const_iterator CIterator = cValues.begin(); CIterator != cValues.end(); ++CIterator){
 		params.C = *CIterator;
		SVM.train(cSvmTrainingData, cSvmLabels, Mat(), Mat(), params);

		//Test SVM Model alone without HOG detectMultiscale
		unsigned int truePositives =0;	
	    unsigned int falsePositives =0;
	    unsigned int falseNegatives =0;
	    unsigned int trueNegatives =0;
	    float result=0;

	    for (vector<string>::const_iterator posTestIterator = testPosImagesSet.begin(); posTestIterator != testPosImagesSet.end(); ++posTestIterator){
	    	vector<float> featureVector;
	    	Mat feature_matrix;
	    	const char *file_name= posTestIterator->c_str();

	    	calculateFeaturesFromInput(*posTestIterator, featureVector, hog);
			if(featureVector.size()==dims){
				feature_matrix= Mat(1,dims,CV_32FC1,featureVector.data(),true);	
			}
			else{
				printf("Error in feature detection.\n");
			}

	    	result=SVM.predict(feature_matrix);
	    	if (result==posLabel){
	    		++truePositives;
	    	}
	    	else if (result==negLabel){
	    		++falseNegatives;
	    		printf("Failed on %s\n", file_name);
	    	}
	    	else{
	    		printf("Error in prediction.\n");
	    	}
		}

		for (vector<string>::const_iterator negTestIterator = testNegImagesSet.begin(); negTestIterator != testNegImagesSet.end(); ++negTestIterator){
	    	vector<float> featureVector;
	    	Mat feature_matrix;
	    	const char *file_name= negTestIterator->c_str();

	    	calculateFeaturesFromInput(*negTestIterator, featureVector, hog);
	    	if(featureVector.size()==dims){
				feature_matrix= Mat(1,dims,CV_32FC1,featureVector.data(),true);	
			}
			else{
				printf("Error in feature detection.\n");
			}
	    	result=SVM.predict(feature_matrix,false);
	    	if (result==posLabel){
	    		++falsePositives;
	    		printf("Failed on %s\n", file_name);
	    	}
	    	else if (result==negLabel){
	    		++trueNegatives;
	    	}
	    	else{
	    		printf("Error in prediction.\n");
	    	}
		}
		printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
		myfile<<"Cvalue: "<<*CIterator<<"\n";
		myfile<<"Results:\n\tTrue Positives: "<<truePositives<<"\n\tTrue Negatives: "<<trueNegatives<<"\n\tFalse Positives: "<<falsePositives<<"\n\tFalse Negative: "<<falseNegatives<<"\n\n";
		//myfile.write("Cvalue: %f\n", *CIterator);
		//myfile.write("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n\n", truePositives, trueNegatives, falsePositives, falseNegatives); 
 	}

 	SVM.save("genfiles/cvSVM/svmModel2.yaml");
 	myfile.close();

}
