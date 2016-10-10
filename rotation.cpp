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

#define PI 3.14159265

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

//get biggest sized image (rescaling factor) to fit in the original image size after rotation
double getScale(double rotate_angle){
  double first_angle= atan(1.0/2.0);
  //cout<<"First: "<<first_angle<<'\n';

  if(rotate_angle==180){
    return 1;
  }

  else if(rotate_angle>180){
    rotate_angle=rotate_angle-180;
  }

  if (rotate_angle>90){
    rotate_angle= 180-rotate_angle;
  }

  double second_angle= (180-rotate_angle-first_angle)*PI/180.0;
  //cout<<"Second: "<<second_angle<<'\n';


  double scale=sin(first_angle)/sin(second_angle);
  //cout<<"Scale: "<<scale<<"\n";
  return scale;

}

cv::Scalar getAverageBorder(Mat& src){
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
      return value;
}

void rotate(Mat& image, double angle, const string& filename)
{

    Mat rotated_img(Size(image.size().width, image.size().height), image.type()); 

    Mat resized;
    double scale= getScale(angle);

    resize(image,resized, Size(image.size().width*scale, image.size().height*scale));

    Point2f src_center(resized.cols/2.0F, resized.rows/2.0F);

    Mat rot_matrix = getRotationMatrix2D(src_center, angle, 1.0);
    rot_matrix.at<double>(0,2)+= rotated_img.size().width/2.0 - src_center.x;
    rot_matrix.at<double>(1,2)+= rotated_img.size().height/2.0 - src_center.y;

    //imshow("resize",resized);
    //waitKey(0);
    Scalar border=getAverageBorder(resized);

    warpAffine(resized, rotated_img, rot_matrix, rotated_img.size(), INTER_LINEAR, BORDER_CONSTANT, border);
    imwrite(filename, rotated_img);
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
  static string rotatedFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/rotated/";
   // static string thinnerFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/thinner_improved/";
   // static string fatterFolder="/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/fatter_improved/";

  getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);

  srand(time(NULL));
  int angle;
  Mat src;
   // /// Load the image
   for (vector<string>::const_iterator posTrainingIterator = positiveTrainingImages.begin(); posTrainingIterator != positiveTrainingImages.end(); ++posTrainingIterator){
       src = imread( *posTrainingIterator, 1 );
       const char *file_name= posTrainingIterator->c_str();

       size_t extensionLocation = string(posTrainingIterator->c_str()).find_last_of(".");
       size_t pathLocation = string(posTrainingIterator->c_str()).find_last_of("/");
       string name = toLowerCase(string(posTrainingIterator->c_str()).substr(pathLocation+1,extensionLocation-1));

       angle=rand()%50+20; //generate angle between 10 and 60;

       rotate(src, angle, rotatedFolder+name);
   //    imwrite(thinnerFolder+name, warp_dst2);
   //    imwrite(fatterFolder+name, warp_dst3);

       string file=rotatedFolder+name;

       printf("%d %s\n", angle, file.c_str() );
    }

    // Mat orig_image = imread("/home/mikacuy/OpenCV/opencv2/projects/trainHOG/pos/normal/bali_positive2.jpg", 1);
    // if (orig_image.empty())
    // {
    //     cout << "!!! Couldn't load " << argv[1] << endl;
    //     return -1;
    // }

    // rotate(orig_image, 300);


    return 0;
  }