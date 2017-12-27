//you
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>
#include <vector>
#include <string> 
#include <inttypes.h>
#include <opencv2/ximgproc.hpp>

//me
#include "functions.hpp"

int main( int argc, char** argv)
{
    Mat img, seg_img;
    img = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    //preprocess
    GaussianBlur( img, seg_img, Size( 7, 7), 0, 0 );

    //SLIC
    Ptr<SuperpixelSLIC> myslic;
    myslic = createSuperpixelSLIC(seg_img, SLICO, 10, 220.0f);
    myslic->iterate(30);
    cv::Mat labels(seg_img.size(), CV_32SC1);
    myslic->getLabels(labels);
    Mat out;
    myslic->getLabelContourMask(out, false);

    //split channels
    std::vector<Mat> bgr; //destination array
    split(img,bgr); //split source  

    //count
    int superpixels = myslic->getNumberOfSuperpixels();
    printf("%d\n", superpixels);

    //features.

    //color
    int *color_hist;
    int color_bins = 30;
    color_hist = new int[superpixels*color_bins];
    memset(color_hist, 0, sizeof(int)*superpixels*color_bins);
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int b, g, r, huehuehue;
            int label;
            b = bgr[0].row(i).at<uint8_t>(j);
            g = bgr[1].row(i).at<uint8_t>(j);
            r = bgr[2].row(i).at<uint8_t>(j);  
            huehuehue = get_hue(r,g,b);
            huehuehue /= 12;
            label = labels.row(i).at<int32_t>(j);
            color_hist[label*color_bins + huehuehue]++;
        }
    }

    //texture
    Size gabor_window(64,64);
    vector<Mat> gabors;
    vector<Mat> responses;
    int gabor_bins = 30;
    for( int i = 0; i < gabor_bins; i++)
    {
        Mat resp;
        Mat cur = getGaborKernel(gabor_window, 10, i*(CV_PI/gabor_bins), 3, 1.0, 0, CV_64F);
        gabors.push_back(cur);
        filter2D(img, resp, CV_8U, cur);
        //myshow("asd"+to_string(i), cur);
        responses.push_back(resp);
        myshow("gabor resp"+to_string(i), resp);
    }

    
    int *gabor_hist;
    gabor_hist = new int[superpixels*gabor_bins];
    memset(gabor_hist, 0, sizeof(int)*superpixels*gabor_bins);
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int label;
            label = labels.row(i).at<int32_t>(j);
            for( int k = 0; k < gabor_bins; k++)
            {
                double cur_gabor_resp = (double) responses[k].row(i).at<uint8_t>(j) / 255.0;
                cur_gabor_resp -= 0.5;
                if( abs(cur_gabor_resp) > 0.25)
                {
                    gabor_hist[label*gabor_bins + k]++;
                }
            }
        }
    }

    //cluster.
    //TODO.
    
    //POST
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            if( out.row(i).at<uint8_t>(j))
            {
                bgr[0].row(i).at<uint8_t>(j) = 0;
                bgr[1].row(i).at<uint8_t>(j) = 0;
                bgr[2].row(i).at<uint8_t>(j) = 0;
            }
            else
            {
                Scalar s = random_color(labels.row(i).at<int32_t>(j));
                bgr[0].row(i).at<uint8_t>(j) = s[0];
                //bgr[1].row(i).at<uint8_t>(j) = s[1];
                //bgr[2].row(i).at<uint8_t>(j) = s[2];
            }
        }
    }

    //merge channels
    merge(bgr, img);
    

    //output
    myshow("asd", img);
    waitKey(0);
    return 0;
}
