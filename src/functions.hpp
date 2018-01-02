#include <opencv2/opencv.hpp>
#include <math.h>

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


int* visited;
double edge_penalty_helper( int a, int b, Mat &labels, Mat &abs_edge, int i = 0, int j = 0)
{
    int x1,x2,x3,x4;
    double x0 = 0;
    if( visited[i*labels.cols + j] || 
        i < 0 || 
        j < 0 || 
        i >= labels.rows || 
        j >= labels.cols)
    {
        return 0;
    }

    int me = labels.row(i).at<int32_t>(j);
    int left, right, up, down;
    if( j-1 > 0)
        left = labels.row(i).at<int32_t>(j-1);
    else
        left = -1;

    if( j+1 >= labels.cols)
        right = labels.row(i).at<int32_t>(j+1);
    else
        right = -1;

    if( i-1 > 0)    
        up = labels.row(i-1).at<int32_t>(j);
    else
        up = -1;
    
    if( i+1 >= labels.rows)    
        down = labels.row(i+1).at<int32_t>(j);
    else
        down = -1;


    visited[i*labels.cols + j] = 1;
    if((me == a && left == b) || (me == b && left == a))
    {
        x0 = abs_edge.row(i).at<double>(j);
    }

    if((me == a && right == b) || (me == b && right == a))
    {
        x0 = abs_edge.row(i).at<double>(j);
    }

    if((me == a && up == b) || (me == b && up == a))
    {
        x0 = abs_edge.row(i).at<double>(j);   
    }

    if((me == a && down == b) || (me == b && down == a))
    {
        x0 = abs_edge.row(i).at<double>(j);
    }

    x1 = edge_penalty_helper( a, b, labels, abs_edge, i-1,j);
    x2 = edge_penalty_helper( a, b, labels, abs_edge, i+1,j);
    x3 = edge_penalty_helper( a, b, labels, abs_edge, i,j-1);
    x4 = edge_penalty_helper( a, b, labels, abs_edge, i,j+1);

    return x0+x1+x2+x3+x4;
}

double edge_penalty( int a, int b, Mat &labels, Mat &abs_edge)
{
    int vsize = labels.cols*labels.rows;
    double result;
    visited = new int[vsize];
    memset(visited, 0, vsize);
    result = edge_penalty_helper( a, b, labels, abs_edge);
    
    delete[] visited;
    return result;
}





