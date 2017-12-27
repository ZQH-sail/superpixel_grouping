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