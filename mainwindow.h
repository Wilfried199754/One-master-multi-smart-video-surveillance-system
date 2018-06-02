#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <string>
#include <math.h>
#include <ctype.h>
#include <QMessageBox>
#include <iostream>
#include <QMainWindow>
#include <QImage>
#include <QTimer>     // 设置采集数据的间隔时间
#include <QFileDialog>
#include <QMouseEvent>
#include <QTextEdit>
#include <QPainter>
#include <QDebug>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <QTime>
#include <windows.h>
//包含opencv库头文件
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/video/tracking.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/features2d/features2d.hpp>
using namespace std;
using namespace cv;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void mainVideoNextFrame();
    void slaveVideo1NextFrame();
    void slaveVideo2NextFrame();
    void on_pushButton_clicked();
    void openCamara(int X);      // 打开摄像头
    void closeCamara();     // 关闭摄像头。
    void openFile(int X);  // 打开文件
    void on_pushButton_3_clicked();
    void on_pushButton_2_clicked();
    void cameraCalibrator();
    void Picture();
    void  Correct();
    void on_pushButton_4_clicked();
    QImage  Mat2QImage(cv::Mat cvImg);
    void MovingObjectDetection();
    void getCentroid(Mat frame1);
    void on_pushButton_5_clicked();
    void on_pushButton_6_clicked();
    void mousePressEvent(QMouseEvent *m);//按下
    void Read_Data();
    void on_OpenButton_clicked();
    void closeEvent(QCloseEvent *event);
    void on_pushButton_8_clicked();
    void delaymsec(int msec);

    void on_pushButton_9_clicked();

    void MainWindow::SerialSendCoordinate(QString CoordinateX,QString CoordinateY);

private:
    Ui::MainWindow *ui;
    Mat frame, dframe,frame1,frame2,Pframe;
    VideoCapture capture,capture1,capture2;
    QImage  image,image1,image2;
    QTimer *timer ;
    QTimer *timer1;
    QTimer *timer2;
    double rate; //FPS
    VideoWriter writer;   //make a video record
    Mat imageInput;
    Mat fgMaskMOG2; //通过MOG2方法得到的掩码图像fgmask
    Mat segm;      //frame的副本

    int TargeCoordinateX;
    int TargeCoordinateY;

    QImage    *imag;
    CvCapture *cam;// 视频获取结构， 用来作为视频获取函数的一个参数
    QPoint offset;//储存鼠标指针位置与窗口位置的差值
    QSerialPort *serial;

    int g_nMedianBlurValue=2;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    vector<Point> point_seq;

    Mat selectChannel(Mat src, int channel);
    bool objectDetection(Mat  src, int threshold_vlaue, int areasize, int channel);

    int PictureNum=0;


};

#endif // MAINWINDOW_H
