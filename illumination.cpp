#include <dirent.h>
#ifdef __MINGW32__
#include <sys/stat.h>
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <cmath>


using namespace cv;
using namespace std;

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    } return t;
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


/** @function main */
 int main( int argc, char** argv )
 {
  static vector<string> validExtensions;
  validExtensions.push_back("jpg");
  validExtensions.push_back("png");
  validExtensions.push_back("ppm");
  validExtensions.push_back("jpeg");
  static vector<string> positiveTrainingImages;

  static string posSamplesDir="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/normal/";
  static string illuminationFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/illumination/";
   // static string thinnerFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/thinner_improved/";
   // static string fatterFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/fatter_improved/";
  double alpha;
  int beta;

  getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);

  srand(time(NULL));
  int angle;
  Mat src;
  bool negative;
  Mat new_image;

   // /// Load the image
   for (vector<string>::const_iterator posTrainingIterator = positiveTrainingImages.begin(); posTrainingIterator != positiveTrainingImages.end(); ++posTrainingIterator){
       src = imread( *posTrainingIterator, 1 );
       new_image = Mat::zeros( src.size(), src.type() );
       const char *file_name= posTrainingIterator->c_str();

       size_t extensionLocation = string(posTrainingIterator->c_str()).find_last_of(".");
       size_t pathLocation = string(posTrainingIterator->c_str()).find_last_of("/");
       string name = toLowerCase(string(posTrainingIterator->c_str()).substr(pathLocation+1,extensionLocation-1));

       alpha=(rand()%100+50)/100.0; //generate alpha between 0.5 and 1.5;
       beta= rand()%30;
       negative=rand()%2;
       if(negative){
        beta=-1*beta;
       }

       src.convertTo(new_image, -1, alpha, beta);
   //    imwrite(thinnerFolder+name, warp_dst2);
   //    imwrite(fatterFolder+name, warp_dst3);

       string file=illuminationFolder+name;

       imwrite(file,new_image);
    }

    return 0;
  }