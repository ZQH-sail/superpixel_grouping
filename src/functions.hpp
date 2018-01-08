#include <opencv2/opencv.hpp>
#include <math.h>
#define duble double
using namespace std;
using namespace cv;
using namespace cv::ximgproc;

void mysave(const String &name, Mat m)
{
    vector<int> params;
    params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    params.push_back(5);
    try
    {
        imwrite(name+".png", m, params);
        cout << name+".png" << endl;
    }
    catch(runtime_error& ex)
    {
        printf("Exception converting image to PNG format: %s\n", ex.what());
    }

}

void myshow(const String &name, Mat m)
{
    cv::imshow(name, m);
    moveWindow(name, 150,150);
}

Scalar random_color( int seed)
{
    return Scalar((21 + (seed*23)) % 255,
                  (83 + (seed*59)) % 255,
                  (137 + (seed*83)) % 255);
}


int get_hue(int r, int g, int b)
{
    float fr, fg, fb;
    float max, min, delta;
    fr = r/255.0;
    fg = g/255.0;
    fb = b/255.0;
    max = (fr>fg?fr:fg);
    max = (max>fb?max:fb);
    min = (fr<fg?fr:fg);
    min = (min<fb?min:fb);
    delta = max-min;
    if( delta < 0.01) return 0;
    if( fr == max)
    {
        return 60*abs(fmod(((fg-fb)/delta), 6.0));
    }
    if( fg == max)
    {
        return 60*(((fb-fr)/delta) + 2);
    }
    if( fb == max)
    {
        return 60*(((fr-fg)/delta) + 4);
    }
}

int get_saturation(int r, int g, int b)
{
    float fr, fg, fb;
    float max, min, delta;
    fr = r/255.0;
    fg = g/255.0;
    fb = b/255.0;
    max = (fr>fg?fr:fg);
    max = (max>fb?max:fb);
    min = (fr<fg?fr:fg);
    min = (min<fb?min:fb);
    delta = max-min;
    
    return delta==0?0:delta/max;
}

int get_value(int r, int g, int b)
{  
    float fr, fg, fb;
    float max, min, delta;
    fr = r/255.0;
    fg = g/255.0;
    fb = b/255.0;
    max = (fr>fg?fr:fg);
    max = (max>fb?max:fb);
    min = (fr<fg?fr:fg);
    min = (min<fb?min:fb);
    return max;
}


double edge_penalty( int a, int b, Mat &labels, Mat &abs_edge)
{
    int vsize = labels.cols*labels.rows;
    int edge_count = 0;
    double result = 0;
    for(int i = 0; i < labels.rows; i++)
    {
        for(int j = 0; j < labels.cols; j++)
        {
            int me = labels.row(i).at<int32_t>(j);
            int left, right, up, down;

            if( j-1 >= 0)
                left = labels.row(i).at<int32_t>(j-1);
            else
                left = -1;

            if( j+1 < labels.cols)
                right = labels.row(i).at<int32_t>(j+1);
            else
                right = -1;

            if( i-1 >= 0)    
                up = labels.row(i-1).at<int32_t>(j);
            else
                up = -1;
            
            if( i+1 < labels.rows)    
                down = labels.row(i+1).at<int32_t>(j);
            else
                down = -1;

            if((me == a && left == b) || (me == b && left == a))
            {
                result += abs_edge.row(i).at<uint8_t>(j);
                edge_count++;
            }
            else if((me == a && right == b) || (me == b && right == a))
            {
                result += abs_edge.row(i).at<uint8_t>(j);
                edge_count++;
            }
            else if((me == a && up == b) || (me == b && up == a))
            {
                result += abs_edge.row(i).at<uint8_t>(j);   
                edge_count++;
            }
            else if((me == a && down == b) || (me == b && down == a))
            {
                result += abs_edge.row(i).at<uint8_t>(j);
                edge_count++;
            }

        }
    }
    return result / edge_count;
}

void all_edge_penalties( vector<uint32_t> &labels, vector<uint8_t> &abs_edge, int rows, int cols, double *penalties, int *adjmat, int superpixels_init)
{
    int* edge_counts = new int[superpixels_init*superpixels_init];
    memset(edge_counts, 0, sizeof(int)*superpixels_init*superpixels_init);
    double result = 0;
    for(int i = 1; i < rows-1; i++)
    {
        for(int j = 1; j < cols-1; j++)
        {
            int me = labels[i*cols + j];
            int left, right, up, down;
            left = labels[i*cols + j - 1];
            right = labels[i*cols + j + 1];
            up = labels[(i-1)*cols + j];
            down = labels[(i+1)*cols + j];

            if( me != right)
            {
                penalties[me*superpixels_init + right] += abs_edge[i*cols + j];
                edge_counts[me*superpixels_init + right]++;
                adjmat[me*superpixels_init + right] = 1;
            }

            if( me != down)
            {
                penalties[me*superpixels_init + down] += abs_edge[i*cols + j];
                edge_counts[me*superpixels_init + down]++;
                adjmat[me*superpixels_init + down] = 1;
            }
        }
    }

    for(int i = 0; i < superpixels_init*superpixels_init; i++)
    {
        if(edge_counts[i])
        {
            penalties[i] /= edge_counts[i];
        }
        else
        {
            penalties[i] = 0;
        }
    }

}

double edge_penalty( int a, int b, vector<uint32_t> &labels, vector<uint8_t> &abs_edge, int rows, int cols)
{
    int edge_count = 0;
    double result = 0;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int me = labels[i*cols + j];
            int left, right, up, down;

            if( j-1 >= 0)
                left = labels[i*cols + j - 1];
            else
                left = -1;

            if( j+1 < cols)
                right = labels[i*cols + j + 1];
            else
                right = -1;

            if( i-1 >= 0)    
                up = labels[(i-1)*cols + j];
            else
                up = -1;
            
            if( i+1 < rows)    
                down = labels[(i+1)*cols + j];
            else
                down = -1;

            if((me == a && left == b) || (me == b && left == a))
            {
                result += abs_edge[i*cols + j];
                edge_count++;
            }
            else if((me == a && right == b) || (me == b && right == a))
            {
                result += abs_edge[i*cols + j];
                edge_count++;
            }
            else if((me == a && up == b) || (me == b && up == a))
            {
                result += abs_edge[i*cols + j];   
                edge_count++;
            }
            else if((me == a && down == b) || (me == b && down == a))
            {
                result += abs_edge[i*cols + j];
                edge_count++;
            }

        }
    }
    return result / edge_count;
}


Mat contour_at_sevgilim( Mat &m)
{
    Mat result( m.rows, m.cols, CV_8S);
    result = 0*result;
    for(int i = 0; i < m.rows-1; i++)
    {
        for(int j = 0; j < m.cols-1; j++)
        {
            if(m.row(i).at<int32_t>(j) != m.row(i).at<int32_t>(j+1))
            {
                result.row(i).at<int8_t>(j) = 1;
            }

            if(m.row(i).at<int32_t>(j) != m.row(i+1).at<int32_t>(j))
            {
                result.row(i).at<int8_t>(j) = 1;
            }
        }
    }

    return result;
}

void refresh_labels(Mat &labels, int &superpixels)
{
    int* visited = new int[superpixels];
    memset(visited, 0, sizeof(int)*superpixels);

    for( int i = 0; i < labels.rows; i++)
    {
        for( int j = 0; j < labels.cols; j++)
        {
            visited[labels.row(i).at<uint32_t>(j)] = 1;
        }
    }

    int count = 0;
    for( int i = 0; i < superpixels; i++)
    {
        if(visited[i])
        {
            visited[i] = count++;
        }

    }

    for( int i = 0; i < labels.rows; i++)
    {
        for( int j = 0; j < labels.cols; j++)
        {
            labels.row(i).at<uint32_t>(j) = visited[labels.row(i).at<uint32_t>(j)];
        }
    }
    //printf("%d\n", count);
    superpixels = count;

    

}

void merge_labels(  Mat &labels, 
                    Mat &img, 
                    std::vector<Mat> &bgr, 
                    int &superpixels, 
                    Mat &labels_init, 
                    int superpixels_init, 
                    Point &p1, 
                    Point &p2, 
                    int *color_hist, 
                    int *gabor_hist, 
                    int color_bins, 
                    int gabor_bins, 
                    int *adjacency_matrix, 
                    double *penalties, double *centerx_init, double *centery_init, int *countpixels_init)
{

    if(superpixels < 10) return;
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

    
    //merge
    int* equivalence;
    equivalence = new int[superpixels];

    double min_dist = INFINITY;
    int min_dist_j = 1;
    int min_dist_i = 0;
    for(int i = 0; i < superpixels; i++)
    {
        for( int j = i+1; j < superpixels; j++)
        {
            double Dct = 0;
            double Dtotal = 0;

            double Ti = sets[i].size();
            double Tj = sets[j].size();
            double alpha = -log2((Ti + Tj) / (double) superpixels_init);
            double ro = 1/(1 + exp((6 - alpha)/0.1 ));


            double Dmax = 0;
            double Dmin = INFINITY;

        
            double graph_dist_x = abs(centerx[i]-centerx[j]);
            double graph_dist_y = abs(centery[i]-centery[j]);

            double graph_dist = 0;
            graph_dist = graph_dist_x*graph_dist_x + graph_dist_y*graph_dist_y;

            for(std::set<int>::iterator it=sets[i].begin(); it!=sets[i].end(); ++it)
            {
                for(std::set<int>::iterator jit=sets[j].begin(); jit!=sets[j].end(); ++jit)  
                {
                    double Dct = 0;

                    for(int k = 0; k < gabor_bins; k++)
                    {
                        double gabor_dist = abs(
                                            gabor_hist[(*it)*gabor_bins + k] - 
                                            gabor_hist[(*jit)*gabor_bins + k]);
                        Dct += gabor_dist;
                    }

                    for(int k = 0; k < color_bins; k++)
                    {
                        double color_dist = abs(
                                            color_hist[(*it)*color_bins + k] - 
                                            color_hist[(*jit)*color_bins + k]);
                        Dct += color_dist;
                    }

                    if( Dct>Dmax)
                    {
                        Dmax = Dct;
                    }

                    if( Dct<Dmin)
                    {
                        Dmin = Dct;
                    }
                }
            }

            double edge_dist = 0;
            int count = 0;
            for(std::set<int>::iterator it=sets[i].begin(); it!=sets[i].end(); ++it)
            {
                for(std::set<int>::iterator jit=sets[j].begin(); jit!=sets[j].end(); ++jit)  
                {
                    if( adjacency_matrix[(*it)*superpixels_init + *jit])
                    {
                        count++;
                        edge_dist += penalties[(*it)*superpixels_init + *jit];
                    }
                }
            }
            if( count)
                edge_dist /= count;
            else
                edge_dist = 0;

            edge_dist *= (gabor_bins+color_bins)*25;
            graph_dist *= (gabor_bins+color_bins)/25;
            //double DL = Dmax + edge_dist + graph_dist;
            double DL = edge_dist + graph_dist;
            double DH = Dmin + 0.4*graph_dist;
            
            Dtotal = ro*DL + (1 - ro)*DH + 2*(countpixels[i] + countpixels[j]);
            
            if(Dtotal < min_dist)
            {
                min_dist = Dtotal;
                min_dist_j = j;
                min_dist_i = i;
            }

        }
        //if(i%100 == 0) printf("%d/%d\n",i,superpixels);

        //equivalence[i] = min_dist_j;

        //if(equivalence[i] == min_dist_j && equivalence[min_dist_j] == i) printf("yo/n");
    }
    //printf("%d-%d\n", min_dist_i, min_dist_j);

    /*
    for(int i = 0; i < superpixels; i++)
    {
        if(equivalence[i] == equivalence[ equivalence[i] ])
            printf("<<<%d \n", i);
    }
*/
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int lbl = labels.row(i).at<int32_t>(j);

            if( lbl == min_dist_j)
                labels.row(i).at<int32_t>(j) = min_dist_i;
        }
    }



    //printf("neydi %d\n",superpixels);
    refresh_labels(labels, superpixels);
    //printf("ne oldu %d\n",superpixels);

    double centerest_dist = INFINITY;
    int centerest_i = 0;
    for(int i = 0; i < superpixels; i++)
    {
        double graph_dist_x = abs(centerx[i]-img.cols/2);
        double graph_dist_y = abs(centery[i]-img.rows/2);

        double graph_dist = 0;
        graph_dist = graph_dist_x*graph_dist_x + graph_dist_y*graph_dist_y;
        if(graph_dist < centerest_dist)
        {
            centerest_dist = graph_dist;
            centerest_i = i;
        }
    }

    //printf("helo\n");

    p1.x = centerx[centerest_i] - img.cols / 5;
    p1.y = centery[centerest_i] - img.rows / 5;

    p2.x = centerx[centerest_i] + img.cols / 5;
    p2.y = centery[centerest_i] + img.rows / 5;

}



