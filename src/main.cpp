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

void selectSort(int *arr, int n)
{
	int pos_min,temp;

	for (int i=0; i < n-1; i++)
	{
	    pos_min = i;//set pos_min to the current index of array
		for (int j=i+1; j < n; j++)
		{
		    if (arr[j] < arr[pos_min]) pos_min=j;
		}
        if (pos_min != i)
        {
            temp = arr[i];
            arr[i] = arr[pos_min];
            arr[pos_min] = temp;
        }
	}
}

int main( int argc, char** argv)
{
    string fname(argv[1]);
    string a = fname.substr(7,6);
    cout << fname << endl;

    Mat img, seg_img;
    img = imread(fname, CV_LOAD_IMAGE_COLOR);

    FILE *f = fopen(("./data/"+a+".txt").c_str(), "r");
    //preprocess
    GaussianBlur( img, seg_img, Size( 7, 7), 0, 0 );

    //SLIC
    Ptr<SuperpixelSLIC> myslic;
    myslic = createSuperpixelSLIC(seg_img, SLIC, 20, 100.0f);
    myslic->iterate(90);
    cv::Mat labels(seg_img.size(), CV_32SC1);
    myslic->getLabels(labels);
    Mat out;
    myslic->getLabelContourMask(out, false);
    Mat labels_init = labels.clone();

    //split channels
    std::vector<Mat> bgr; //destination array
    split(img,bgr); //split source  

    //count
    int superpixels = myslic->getNumberOfSuperpixels();
    //printf("%d\n", superpixels);

    //merge
    int superpixels_init = superpixels;
    Point p1, p2;
    for( int i = 0; i < 50; i++)
        merge_labels(labels, img, bgr, superpixels, labels_init, superpixels_init, p1, p2);

    Mat new_out = contour_at_sevgilim( labels);
    //POST
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            if( new_out.row(i).at<uint8_t>(j))
            {
                bgr[0].row(i).at<uint8_t>(j) = 0;
                bgr[1].row(i).at<uint8_t>(j) = 0;
                bgr[2].row(i).at<uint8_t>(j) = 0;
            }
            else
            {
                Scalar s = random_color(labels.row(i).at<int32_t>(j));
                //bgr[0].row(i).at<uint8_t>(j) = s[0];
                //bgr[1].row(i).at<uint8_t>(j) = s[1];
                //bgr[2].row(i).at<uint8_t>(j) = s[2];
            }
        }
    }


    //merge channels
    merge(bgr, img);
    rectangle(img, 
              p1, 
              p2, 
              Scalar( 255,0,0), 2);

    int objs = 0;
    int detected = 0;
    while(1)
    {
        int p1x,p1y,p2x,p2y, bs;
        int o = fscanf( f, "%d %d %d %d %d", &p1x, &p1y, &p2x, &p2y, &bs);
        if( o == EOF) break;
        //printf("%d %d %d %d %d \n", p1x,p2x,p1y,p2y, bs);
        objs++;

        vector<int> ptsx;
        ptsx.push_back(p1.x);
        ptsx.push_back(p2.x);
        ptsx.push_back(p1x);
        ptsx.push_back(p2x);
        selectSort(&ptsx[0], 4);

        vector<int> ptsy;
        ptsy.push_back(p1.y);
        ptsy.push_back(p2.y);
        ptsy.push_back(p1y);
        ptsy.push_back(p2y);
        selectSort(&ptsy[0], 4);

        double intersection_area = (ptsx[2] - ptsx[1])*(ptsy[2] - ptsy[1]);
        double union_area = (p2.x-p1.x)*(p2.y-p1.y) + (p2x-p1x)*(p2y-p1y) - intersection_area;
        double iou = intersection_area / union_area;
        if(iou > 0.5) detected++;
    }
    printf("detected: %d\n", detected);
    printf("objects: %d\n", objs);

    //output
    mysave(a, img);
    fclose(f);
    return 0;
}
