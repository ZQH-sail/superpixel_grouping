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

void showplot(const String &name, Mat xData, int xmin, int xmax, int ymin, int ymax)
{
    Mat plot_result;
    Mat norm;

    normalize(xData, norm, 0, ymax * 3 / 4, NORM_MINMAX, CV_64F);
    Ptr<plot::Plot2d> plot = plot::Plot2d::create(norm);
    plot->setMaxX(xmax);
    plot->setMinX(xmin);
    plot->setMaxY(ymax);
    plot->setMinY(ymin);
    plot->setInvertOrientation(true);
    plot->setShowGrid(false);
    plot->setPlotSize(xData.cols, xData.cols / 3);
    plot->setPlotLineWidth(2);
    plot->setPlotBackgroundColor( Scalar( 50, 50, 50 ) );
    plot->setPlotLineColor( Scalar( 255, 50, 50 ) );
    plot->setPlotAxisColor( Scalar( 50, 50, 200 ) );
    plot->render( plot_result );
    plot->setShowText(false);

    myshow( name, plot_result );
}

void saveplot(const String &name, Mat xData, int xmin, int xmax, int ymin, int ymax)
{
    Mat plot_result;
    Mat norm;

    normalize(xData, norm, 0, ymax * 3 / 4, NORM_MINMAX, CV_64F);
    Ptr<plot::Plot2d> plot = plot::Plot2d::create(norm);
    plot->setMaxX(xmax);
    plot->setMinX(xmin);
    plot->setMaxY(ymax);
    plot->setMinY(ymin);
    plot->setInvertOrientation(true);
    plot->setShowGrid(false);
    plot->setPlotSize(xData.cols, xData.cols / 3);
    plot->setPlotLineWidth(2);
    plot->setPlotBackgroundColor( Scalar( 50, 50, 50 ) );
    plot->setPlotLineColor( Scalar( 255, 50, 50 ) );
    plot->setPlotAxisColor( Scalar( 50, 50, 200 ) );
    plot->render( plot_result );
    plot->setShowText(false);

    mysave( name, plot_result );
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
        return 60*fmod(((fg-fb)/delta), 6.0);
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
    printf("%d\n", count);
    superpixels = count;
}

void merge_labels( Mat &labels, Mat &img, std::vector<Mat> &bgr, int &superpixels, Mat &labels_init, int superpixels_init)
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

    //color
    int *color_hist;
    int color_bins = 30;
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
            huehuehue = get_hue(r,g,b);
            huehuehue /= 12;
            label = labels_init.row(i).at<int32_t>(j);
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
        //myshow("gabor resp"+to_string(i), resp);
    }

    int *gabor_hist;
    gabor_hist = new int[superpixels_init*gabor_bins];
    memset(gabor_hist, 0, sizeof(int)*superpixels_init*gabor_bins);
    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int label;
            label = labels_init.row(i).at<int32_t>(j);
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

    //myshow("edge", abs_edge);
    //printf("%d\n", abs_edge.type());

    //center of MASS
    double *centerx, *centery;
    int *countpixels;

    centerx = new double[superpixels];
    centery = new double[superpixels];
    countpixels = new int[superpixels];
    memset(centerx, 0, sizeof(double)*superpixels);
    memset(centery, 0, sizeof(double)*superpixels);
    memset(countpixels, 0, sizeof(int)*superpixels);

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


    double min_graph_dist = INFINITY;
    for(int i = 0; i < superpixels; i++)
    {
        for( int j = 0; j < superpixels; j++)
        {
            if(j == i) continue;
            double graph_dist_x = abs(centerx[i]-centerx[j]);
            double graph_dist_y = abs(centery[i]-centery[j]);
            double graph_dist = graph_dist_x*graph_dist_x + graph_dist_y*graph_dist_y;

            if(graph_dist < min_graph_dist)
            {
                min_graph_dist = graph_dist;
            }
        }
    }

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
    
    //merge
    int* equivalence;
    equivalence = new int[superpixels];
    for(int i = 0; i < superpixels; i++)
    {
        double min_dist = INFINITY;
        int min_dist_j = i;

        for( int j = 0; j < superpixels; j++)
        {
            if(j == i) continue;
            double Dct = 0;
            double Dtotal = 0;
            double graph_dist_x = abs(centerx[i]-centerx[j]);
            double graph_dist_y = abs(centery[i]-centery[j]);
            double graph_dist = graph_dist_x*graph_dist_x + graph_dist_y*graph_dist_y;


            double Ti = sets[i].size();
            double Tj = sets[j].size();
            double alpha = log2((Ti + Tj) / (double) superpixels_init);
            double ro = 1/(1 + exp((6 - alpha)/0.1 ));


            double Dmax = 0;
            double Dmin = INFINITY;

                
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
            //edge_dist = edge_penalty( i, j, labels_vec, abs_edge_vec, labels.rows, labels.cols);
            int count = 0;
            for(std::set<int>::iterator it=sets[i].begin(); it!=sets[i].end(); ++it)
            {
                for(std::set<int>::iterator jit=sets[j].begin(); jit!=sets[j].end(); ++jit)  
                {
                    if( adjacency_matrix[(*it)*superpixels_init + *jit])
                    {
                        count++;
                        edge_dist += penalties[(*it)*superpixels_init + *jit] / 255;
                    }
                }
            }
            if( count)
                edge_dist /= count;
            else
                edge_dist = 0;

            double DL = Dmax + edge_dist + graph_dist;
            double DH = Dmin + 0.6*graph_dist;
            
            Dtotal = ro*DL + (1 - ro)*DH + 2*(countpixels[i] + countpixels[j]);
        
            
            if(Dtotal < min_dist && Dtotal > 0)
            {
                min_dist = Dtotal;
                min_dist_j = j;
            }
        }
        if(i%100 == 0) printf("%d/%d\n",i,superpixels);

        equivalence[i] = min_dist_j;
    }

    for(int i = 0; i < superpixels; i++)
    {
        if(equivalence[i] == equivalence[ equivalence[i] ])
            printf("<<<%d \n", i);
    }

    for( int i = 0; i < img.rows; i++)
    {
        for( int j = 0; j < img.cols; j++)
        {
            int lbl = labels.row(i).at<int32_t>(j);

            if( lbl > equivalence[lbl])
                labels.row(i).at<int32_t>(j) = equivalence[lbl];
        }
    }

    printf("neydi %d\n",superpixels);
    refresh_labels(labels, superpixels);
    printf("ne oldu %d\n",superpixels);
}



