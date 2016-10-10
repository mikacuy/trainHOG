#include <dirent.h>
#ifdef __MINGW32__
#include <sys/stat.h>
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string>

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
   static string rotatedFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/rotated_improved/";
   static string thinnerFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/thinner_improved/";
   static string fatterFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/fatter_improved/";

   getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);


   Point2f srcTri[3];
   Point2f dstTri1[3];
   Point2f dstTri2[3];
   Point2f dstTri3[3];

   Mat warp_mat1( 2, 3, CV_32FC1 );
   Mat warp_mat2( 2, 3, CV_32FC1 );
   Mat warp_mat3( 2, 3, CV_32FC1 );

   Mat src, warp_dst1, warp_dst2, warp_dst3;

   /// Load the image
   for (vector<string>::const_iterator posTrainingIterator = positiveTrainingImages.begin(); posTrainingIterator != positiveTrainingImages.end(); ++posTrainingIterator){
      src = imread( *posTrainingIterator, 1 );
      const char *file_name= posTrainingIterator->c_str();

      //get average for border------------------------------
      int border_pixel_count=0;
      int blue_sum=0;
      int green_sum=0;
      int red_sum=0;
      cv::Vec3b intensity1;
      cv::Vec3b intensity2;

      for (int i=0; i<src.rows; i++){
        intensity1= src.at<cv::Vec3b>(0,i);
        intensity2= src.at<cv::Vec3b>(src.cols -1,i);

        blue_sum = blue_sum+intensity1[0]+intensity2[0];
        green_sum = green_sum+intensity1[1]+intensity2[1];
        red_sum = red_sum+intensity1[2]+intensity2[2];

        border_pixel_count+=2;
      }

      for (int i=1; i<src.cols-1; i++){
        intensity1= src.at<cv::Vec3b>(i,0);
        intensity2= src.at<cv::Vec3b>(i,src.rows-1);

        blue_sum = blue_sum+intensity1[0]+intensity2[0];
        green_sum = green_sum+intensity1[1]+intensity2[1];
        red_sum = red_sum+intensity1[2]+intensity2[2];

        border_pixel_count+=2;
      }

      int blue_average= blue_sum/border_pixel_count;
      int green_average= green_sum/border_pixel_count;
      int red_average= red_sum/border_pixel_count;

      Scalar value;
      value= Scalar(blue_average, green_average, red_average);


      //-----------------------------------------------------

      warp_dst1 = Mat::zeros( src.rows, src.cols, src.type() );
      warp_dst2 = Mat::zeros( src.rows, src.cols, src.type() );
      warp_dst3 = Mat::zeros( src.rows, src.cols, src.type() );

         /// Set your 3 points to calculate the  Affine Transform
      srcTri[0] = Point2f( 0,0 );
      srcTri[1] = Point2f( src.cols - 1, 0 );
      srcTri[2] = Point2f( 0, src.rows - 1 );

      //slated bit rotated
      dstTri1[0] = Point2f( src.cols*0.0, src.rows*0.1 );
      dstTri1[1] = Point2f( src.cols*0.6, src.rows*0 );
      dstTri1[2] = Point2f( src.cols*0.3, src.rows-1 );

      //thinner
      dstTri2[0] = Point2f( src.cols*0.15, src.rows*0 );
      dstTri2[1] = Point2f( src.cols*0.85, src.rows*0 );
      dstTri2[2] = Point2f( src.cols*0.0, src.rows-1 );

      //fatter
      dstTri3[0] = Point2f( src.cols*0.0, src.rows*0.15 );
      dstTri3[1] = Point2f( src.cols*0.9, src.rows*0.2 );
      dstTri3[2] = Point2f( src.cols*0, src.rows*0.85 );

         /// Get the Affine Transform
      warp_mat1 = getAffineTransform( srcTri, dstTri1 );
      warp_mat2 = getAffineTransform( srcTri, dstTri2 );
      warp_mat3 = getAffineTransform( srcTri, dstTri3 );

      /// Apply the Affine Transform just found to the src image
      //other option is BORDER_REPLICATE

      warpAffine( src, warp_dst1, warp_mat1, warp_dst1.size(),INTER_LINEAR, BORDER_CONSTANT, value );
      warpAffine( src, warp_dst2, warp_mat2, warp_dst2.size(),INTER_LINEAR, BORDER_CONSTANT, value );
      warpAffine( src, warp_dst3, warp_mat3, warp_dst3.size(),INTER_LINEAR, BORDER_CONSTANT, value );

      size_t extensionLocation = string(posTrainingIterator->c_str()).find_last_of(".");
      size_t pathLocation = string(posTrainingIterator->c_str()).find_last_of("/");
      string name = toLowerCase(string(posTrainingIterator->c_str()).substr(pathLocation+1,extensionLocation-1));

      imwrite(rotatedFolder+name, warp_dst1);
      imwrite(thinnerFolder+name, warp_dst2);
      imwrite(fatterFolder+name, warp_dst3);

      string file=rotatedFolder+name;

      printf("%s\n", file.c_str() );
   }

   return 0;
  }