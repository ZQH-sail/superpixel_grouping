//you
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
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

    FILE *f = fopen(("./data/"+a+".txt").c_str(), "r+");
    if(!f) printf("null anam");
    //preprocess
    GaussianBlur( img, seg_img, Size( 7, 7), 0, 0 );

    //SLIC
    Ptr<SuperpixelSLIC> myslic;
    myslic = createSuperpixelSLIC(seg_img, SLIC, 20, 8.0f);
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
    int superpixels_init = superpixels;


    //HISTOGRAMS 
    //color
    int *color_hist;
    int temp_bins = 20;
    int color_bins = temp_bins*3;
    color_hist = new int[superpixels_init*color_bins];
    memset(color_hist, 0, sizeof(int)*superpixels_init*color_bins);
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int b, g, r, huehuehue;
            int label;
            b = bgr[0].row(i).at<uint8_t>(j);
            g = bgr[1].row(i).at<uint8_t>(j);
            r = bgr[2].row(i).at<uint8_t>(j);  
            huehuehue = get_hue(r,g,b) / (360/temp_bins);
            int sat = get_saturation(r,g,b) / (100/temp_bins);
            int val = get_value(r,g,b) / (100/temp_bins);
            label = labels_init.row(i).at<int32_t>(j);

            color_hist[label*color_bins + huehuehue]++; 
            color_hist[label*color_bins + temp_bins + sat]++;
            color_hist[label*color_bins + 2*temp_bins + val]++;
        }
    }


    //texture
    Size gabor_window(64,64);
    vector<Mat> gabors;
    vector<Mat> responses;
    int orientations = 8;
    int gbins = 10;
    int gabor_bins = orientations*gbins*3;
    for( int i = 0; i < orientations; i++)
    {
        Mat resp;
        Mat cur = getGaborKernel(gabor_window, 10, i*(CV_PI/gabor_bins), 3, 1.0, 0, CV_64F);
        gabors.push_back(cur);
        filter2D(img, resp, CV_8U, cur);
        //myshow("asd"+to_string(i), cur);
        responses.push_back(resp);
        //myshow("gabor resp"+to_string(i), resp);
    }

    vector<std::vector<Mat> > bgr_gabors; //destination array
    for(int i = 0; i < orientations; i++)
    {
        std::vector<Mat> bgr_gabor;
        split(responses[i],bgr_gabor); //split source  
        bgr_gabors.push_back(bgr_gabor);
    }

    int *gabor_hist;
    gabor_hist = new int[superpixels_init*gabor_bins];
    memset(gabor_hist, 0, sizeof(int)*superpixels_init*gabor_bins);
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int anan;
            anan = labels_init.row(i).at<int32_t>(j);
            for( int k = 0; k < orientations; k++)
            {
                std::vector<Mat> bgr_gabor = bgr_gabors[k];
                //b
                double cur_gabor_resp_b = (double) bgr_gabor[0].row(i).at<uint8_t>(j) / 256.0;
                cur_gabor_resp_b *= 10;
                gabor_hist[anan*gabor_bins + k*gbins + (int)cur_gabor_resp_b]++;
                //g
                double cur_gabor_resp_g = (double) bgr_gabor[1].row(i).at<uint8_t>(j) / 256.0;
                cur_gabor_resp_g *= 10;
                gabor_hist[anan*gabor_bins + k*gbins + orientations*gbins + (int)cur_gabor_resp_g]++;
                //r
                double cur_gabor_resp_r = (double) bgr_gabor[2].row(i).at<uint8_t>(j) / 256.0;
                cur_gabor_resp_r *= 10;
                gabor_hist[anan*gabor_bins + k*gbins + 2*orientations*gbins + (int)cur_gabor_resp_r]++;
                
            }
        }
    }


    //edge
    Mat gray_img;
    cvtColor( img, gray_img, CV_BGR2GRAY );
    Mat edgex, edgey, abs_edgex, abs_edgey;
    Sobel(gray_img, edgex, CV_16S, 1, 0);
    Sobel(gray_img, edgey, CV_16S, 0, 1);
    convertScaleAbs( edgex, abs_edgex );
    convertScaleAbs( edgey, abs_edgey );
    Mat abs_edge;
    addWeighted( abs_edgex, 0.5, abs_edgey, 0.5, 0, abs_edge);

    vector<uint32_t> labels_init_vec;
    vector<uint8_t> abs_edge_vec;
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            labels_init_vec.push_back( labels_init.row(i).at<uint32_t>(j));
            abs_edge_vec.push_back( abs_edge.row(i).at<uint8_t>(j));
        }
    }

    duble* penalties = new duble[superpixels_init*superpixels_init];
    int* adjacency_matrix = new int[superpixels_init*superpixels_init];
    memset(adjacency_matrix, 0, sizeof(int)*superpixels_init*superpixels_init);
    memset(penalties, 0, sizeof(duble)*superpixels_init*superpixels_init);
    
    all_edge_penalties( labels_init_vec, abs_edge_vec, img.rows, img.cols, penalties, adjacency_matrix, superpixels_init);

    double *centerx_init, *centery_init;
    int *countpixels_init;

    centerx_init = new double[superpixels_init];
    centery_init = new double[superpixels_init];
    countpixels_init = new int[superpixels_init];
    memset(centerx_init, 0, sizeof(double)*superpixels_init);
    memset(centery_init, 0, sizeof(double)*superpixels_init);
    memset(countpixels_init, 0, sizeof(int)*superpixels_init);

    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            centerx_init[labels_init.row(i).at<int32_t>(j)] += j;
            centery_init[labels_init.row(i).at<int32_t>(j)] += i;
            
            countpixels_init[labels_init.row(i).at<int32_t>(j)]++;
        }
    }

    for(int i = 0; i < superpixels_init; i++)
    {
        centerx_init[i] /= countpixels_init[i];
        centery_init[i] /= countpixels_init[i];
    }



    //merge
    Point p1, p2;
    for( int i = 0; i < 1000; i++)
    {
        merge_labels(   labels, img, bgr, 
                        superpixels, labels_init, 
                        superpixels_init, p1, p2, 
                        color_hist, gabor_hist, 
                        color_bins, gabor_bins, 
                        adjacency_matrix, penalties, 
                        centerx_init, centery_init, 
                        countpixels_init);

    }

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
                bgr[0].row(i).at<uint8_t>(j) = s[0];
                //bgr[1].row(i).at<uint8_t>(j) = s[1];
                //bgr[2].row(i).at<uint8_t>(j) = s[2];
            }
        }
    }


    //merge channels
    merge(bgr, img);
   

    //lazim oldu

    vector<set<int> > sets(superpixels);
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int set_label = labels.row(i).at<uint32_t>(j);
            int superpix_label = labels.row(i).at<uint32_t>(j);
            sets[set_label].insert(superpix_label);
        }
    }

    vector<vector<int> > mins_maxs(superpixels);


    for(int k = 0; k < superpixels; k++)
    {
        int minx = 99999;
        int miny = 99999;
        int maxx = -1;
        int maxy = -1;
        for( int i = 0; i < img.rows; i++)
        {
            for( int j = 0; j < img.cols; j++)
            {
                if(labels.row(i).at<uint32_t>(j) == k)
                {
                    if(j < minx)
                    {
                        minx = j;
                    }
                    if(j > maxx)
                    {
                        maxx = j;
                    }  
                    if(i < miny)
                    {
                        miny = i;
                    }
                    if(i > maxy)
                    {
                        maxy = i;
                    }

                }
            }
        }

        mins_maxs[k].push_back(minx);
        mins_maxs[k].push_back(miny);
        mins_maxs[k].push_back(maxx);
        mins_maxs[k].push_back(maxy);

    }
    double *centerx, *centery;    
    int *countpixels;

    centerx = new double[superpixels ];
    centery = new double[superpixels ];
    countpixels = new int[superpixels ];
    memset(centerx, 0, sizeof(double)*superpixels );
    memset(centery, 0, sizeof(double)*superpixels );
    memset(countpixels, 0, sizeof(int)*superpixels );

    for( int i = 0; i < superpixels; i++)
    {
        for(std::set<int>::iterator it=sets[i].begin(); it!=sets[i].end(); ++it)
        {
            centerx[i] += centerx_init[*it]*countpixels_init[*it];
            centery[i] += centery_init[*it]*countpixels_init[*it];
            countpixels[i] += countpixels_init[*it];
        }
        centerx[i] /= countpixels[i];
        centery[i] /= countpixels[i];
    }

  

    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            centerx[labels.row(i).at<int32_t>(j)] += j;
            centery[labels.row(i).at<int32_t>(j)] += i;
            
            countpixels[labels.row(i).at<int32_t>(j)]++;
        }
    }

    
    for(int i = 0; i < superpixels; i++)
    {
        centerx[i] /= countpixels[i];
        centery[i] /= countpixels[i];
    }

 
    int objs = 0;
    int detected = 0;
    vector<double> ious;
    vector<int> maxiou_indices;
    while(1)
    {
        int p1x,p1y,p2x,p2y, bs;
        int o = fscanf( f, "%d %d %d %d %d", &p1x, &p1y, &p2x, &p2y, &bs);
        if( o == EOF) break;
        //printf("%d %d %d %d %d \n", p1x,p2x,p1y,p2y, bs);
        objs++;
        double max_iou = -1;
        int max_iou_indx = -1;
        for( int i = 0; i < superpixels; i++)
        {
            Point pp1, pp2;
            pp1.x = mins_maxs[i][0];
            pp1.y = mins_maxs[i][1];

            pp2.x = mins_maxs[i][2];
            pp2.y = mins_maxs[i][3];

            vector<int> ptsx;
            ptsx.push_back(pp1.x);
            ptsx.push_back(pp2.x);
            ptsx.push_back(p1x);
            ptsx.push_back(p2x);
            selectSort(&ptsx[0], 4);

            vector<int> ptsy;
            ptsy.push_back(pp1.y);
            ptsy.push_back(pp2.y);
            ptsy.push_back(p1y);
            ptsy.push_back(p2y);
            selectSort(&ptsy[0], 4);

            double intersection_area = (ptsx[2] - ptsx[1])*(ptsy[2] - ptsy[1]);
            double union_area = (p2.x-p1.x)*(p2.y-p1.y) + (p2x-p1x)*(p2y-p1y) - intersection_area;
            double iou = intersection_area / union_area;
            if(iou > max_iou)
            {
                max_iou = iou;
                max_iou_indx = i;
            }
                
        }

        ious.push_back(max_iou);
        maxiou_indices.push_back(max_iou_indx);

    }

    for( int i = 0; i < objs; i++)
    {
        Point pp1, pp2;
        pp1.x = mins_maxs[maxiou_indices[i]][0];
        pp1.y = mins_maxs[maxiou_indices[i]][1];

        pp2.x = mins_maxs[maxiou_indices[i]][2];
        pp2.y = mins_maxs[maxiou_indices[i]][3];

        rectangle(img, pp1, pp2, Scalar( 255,0,0), 2);
    }

    for(int i = 0; i < objs; i++)
    {
        if(ious[i] >= 0.5) detected++;
    }
    printf("detected: %d\n", detected);
    printf("objects: %d\n", objs);
    printf("superpixels: %d\n", superpixels);

    //output
    fclose(f);
    mysave(a, img);
    return 0;
}
