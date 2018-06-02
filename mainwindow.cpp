#include "mainwindow.h"
#include "ui_mainwindow.h"

#define threshold_diff1 25 //设置简单帧差法阈值
#define threshold_diff2 25 //设置简单帧差法阈值

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->MainVideo->setScaledContents(true);  //fit video to label area
    ui->SlaveVideo1->setScaledContents(true);
    ui->SlaveVideo2->setScaledContents(true);

    //查找可用的串口
    foreach(const QSerialPortInfo &info, QSerialPortInfo::availablePorts())
    {
        QSerialPort serial;
        serial.setPort(info);
        if(serial.open(QIODevice::ReadWrite))
        {
            ui->PortBox->addItem(serial.portName());
            serial.close();
        }
    }
    //设置波特率下拉菜单默认显示第三项
    ui->BaudBox->setCurrentIndex(3);
    qDebug() << tr("SetDone");
}

void MainWindow::delaymsec(int msec)
{
    QTime dieTime = QTime::currentTime().addMSecs(msec);

    while( QTime::currentTime() < dieTime )

        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    closeCamara();//|窗口关闭之前需要的操作~
    this->close();
}

void GetCoordinate(int O11, int O12,int O21,int O22, int temp,int Length)//原坐标系坐标原点O11,O12;圆心O21，O22;角度temp;长度Length
{
    double PI = 3.1415;
    //计算O2坐标系内的坐标
    double x = Length * cos(temp*PI / 180);
    double y = Length * sin(temp*PI / 180);
    x = O21 - O11 + x;
    y = O22 - O12 + y;

}

QImage  MainWindow::Mat2QImage(cv::Mat cvImg)
{
    QImage qImg;
    if(cvImg.channels()==3)                             //3 channels color image
    {

        cv::cvtColor(cvImg,cvImg,CV_BGR2RGB);
        qImg =QImage((const unsigned char*)(cvImg.data),
                     cvImg.cols, cvImg.rows,
                     cvImg.cols*cvImg.channels(),
                     QImage::Format_RGB888);
    }
    else if(cvImg.channels()==1)                    //grayscale image
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                     cvImg.cols,cvImg.rows,
                     cvImg.cols*cvImg.channels(),
                     QImage::Format_Indexed8);
    }
    else
    {
        qImg =QImage((const unsigned char*)(cvImg.data),
                     cvImg.cols,cvImg.rows,
                     cvImg.cols*cvImg.channels(),
                     QImage::Format_RGB888);
    }

    return qImg;

}

void MainWindow::MovingObjectDetection(){
    double rate = 33;
    int delay = 1000 / rate;
    Mat framepro;
    bool flag = false;
    int n=0;
    //namedWindow("image");
    //namedWindow("test");
    while(capture.read(frame))
    {   n++;
        QApplication::processEvents();
        if (false == flag)
        {
            framepro = frame.clone();//将第一帧图像拷贝给framePro
            flag = true;
        }
        else
        {
            absdiff(frame, framepro, dframe);//帧间差分计算两幅图像各个通道的相对应元素的差的绝对值。
            framepro = frame.clone();//将当前帧拷贝给framepro
            threshold(dframe, dframe, 80, 255, CV_THRESH_BINARY);//阈值分割
            //imshow("image", frame);
            //imshow("test", dframe);
            ui->SlaveVideo1->setPixmap(QPixmap::fromImage(Mat2QImage(dframe)));
            //waitKey(delay);
            if(!objectDetection(dframe, 100,100, 1)){
                break;
            };
            //getCentroid(frame);
        }
    }
}

Mat MainWindow::selectChannel(Mat src, int channel)
{///select image channel
    Mat image,gray,hsv;

    image=src.clone(); //not directly operate on source image


    //gray = cvCreateImage( cvGetSize(image), 8, 1 );
    //hsv = cvCreateImage( cvGetSize(image), 8, 3 );
    cvtColor(image,gray,CV_BGR2GRAY);
    cvtColor(image,hsv,CV_BGR2HSV);
    vector<Mat> imageRGBORHSV;
    Mat imageSC;
    switch (channel)
    {
    case 1:
        //cvSplit(image,imageSC,0,0,0);
        split(image,imageRGBORHSV);
        imageSC=imageRGBORHSV[0];
        break;
    case 2:
        //cvSplit(image,0,imageSC,0,0);
        split(image,imageRGBORHSV);
        imageSC=imageRGBORHSV[1];
        break;
    case 3:
        //cvSplit(image,0,0,imageSC,0);
        split(image,imageRGBORHSV);
        imageSC=imageRGBORHSV[2];
        break;
    case 4:
        //cvSplit(hsv,imageSC,0,0,0);
        split(hsv,imageRGBORHSV);
        imageSC=imageRGBORHSV[0];
        break;
    case 5:
        //cvSplit(hsv,0,imageSC,0,0);
        split(hsv,imageRGBORHSV);
        imageSC=imageRGBORHSV[1];
        break;
    case 6:
        //cvSplit(hsv,0,0,imageSC,0);
        split(hsv,imageRGBORHSV);
        imageSC=imageRGBORHSV[2];
        break;
    default:
        //cvCopy( gray, imageSC, 0 );
        imageSC=gray;
    }
    //cvReleaseImage(&image);
    //cvReleaseImage(&hsv);
    //cvReleaseImage(&gray);
    return imageSC;

}

bool MainWindow::objectDetection(Mat  src, int threshold_vlaue, int areasize, int channel)
{/*
  @param[out] success or fail.
  @param[in]  threshold  threshold for segmentation.
  @param[in]  areasize   threshold for selecting large-enough object.
  @param[in]  channel 1(B), 2(G), 3(R), 4(H), 5(S), 6(V), other(GRAY)
  */
    int i;
    //cvCopy(src,displayImage,NULL);
    Mat displayImage=frame.clone();
    //cvClearSeq(point_seq);
    //cvClearSeq(contour);
    //cvClearMemStorage(storage);

    Mat imageSC=selectChannel(src,channel);
    //smooth(imageSC,imageSC,CV_MEDIAN);//图像中值滤波
    medianBlur( imageSC, imageSC, g_nMedianBlurValue*2+1 );//中值滤波
    blur( imageSC, imageSC, Size(3,3) );
    //cvAdaptiveThreshold( gray, gray, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY, 7, 0);
    threshold(imageSC,imageSC,threshold_vlaue,255,CV_THRESH_BINARY);
    if(1)
        threshold(imageSC,imageSC,threshold_vlaue,255,CV_THRESH_BINARY_INV);              //cvNot(imageSC,imageSC);//把元素的每一位取反
    //imageSC->origin = 0;

    dilate(imageSC, imageSC, element);//膨胀
    //CvScalar color = CV_RGB( 155, 155,155 );//灰度图像
    Scalar color = Scalar( 155, 155, 155 );
    vector<vector<Point>> Contours;
    vector<Vec4i> Hierarchy;

    findContours( imageSC, Contours, Hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,  Point(0,0) );
    vector<Moments> mu(Contours.size());
    vector<Point2f> mc(Contours.size());
    Mat drawing=frame;//Mat::zeros(src.size(),CV_8UC3);
    for(int i = 0; i< Contours.size(); i++ ){
        mu[i]=moments(Contours[i],false);
    }
    for(int i = 0; i< Contours.size(); i++ ){
        mc[i]=Point2d(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
    }

    for(int i = 1; i< Contours.size(); i++ ){
        double tmparea=fabs(contourArea(Contours[i]));
        if(tmparea>areasize){
            drawContours(displayImage, Contours, i, color, 2, 8, Hierarchy, 0, Point() );//you can change 1 to CV_FILLED
            if(1){
                //Mat region=Contour[i];
                //CvMoments moments;
                //cvMoments( region, &moments,0 );
                //cvMoments( &contour, &moments,0 );
                // cvDrawContours( cnt_img, _contours, CV_RGB(255,0,0), CV_RGB(0,255,0), _levels, 3, CV_AA, cvPoint(0,0) ); CV_FILLED
                ////////////////////////////////////////////////
                /*float xd,yd;
                int xc,yc;
                float m00, m10, m01, inv_m00;
                Point point;

                m00 = moments.m00;
                m10 = moments.m10;
                m01 = moments.m01;
                inv_m00 = 1. / m00;

                xd =m10 * inv_m00;//一阶矩
                yd =m01 * inv_m00;

                xc=cvRound(xd);//返回和参数最接近的整数
                yc=cvRound(yd);
                i++;

                point.x=xd;
                point.y=yd;
                cvSeqPush(point_seq, &point );
                circle(displayImage,point,5,cvScalar(255,0,0));*/
                //mu=moments(Contours[i],false);
                //mc=Point2d(mu.m10/mu.m00,mu.m01/mu.m00);
                rectangle(drawing, boundingRect(Contours.at(i)), cvScalar(0,255,0));

                char tam[10000];
                sprintf(tam,"(%0.0f,%0.0f,%0.0d,%0.0d)",mc[i].x,mc[i].y,boundingRect(Contours.at(i)).height ,boundingRect(Contours.at(i)).width);
                //tam[0]=mc[i].x;

                TargeCoordinateX=mc[i].x;
                TargeCoordinateY=mc[i].y;
                QString QTargeCoordinateX=QString::fromStdString(to_string(TargeCoordinateX));
                QString QTargeCoordinateY=QString::fromStdString(to_string(TargeCoordinateY));
                SerialSendCoordinate(QTargeCoordinateX,QTargeCoordinateY);
                //ui->plainTextEdit->appendPlainText(QTargeCoordinateX+","+QTargeCoordinateY+"/n");

                cout<<"x "<<mc[i].x<<" y  "<<mc[i].y<<" height="<<boundingRect(Contours.at(i)).height<<" width="<<boundingRect(Contours.at(i)).width<<endl;
                circle(drawing,mc[i],5,cvScalar(255,0,0));
                putText(drawing,tam,Point(mc[i].x,mc[i].y),FONT_HERSHEY_SIMPLEX,0.4,Scalar(255,0,255),1);


            }


        }
    }
    //  cvNamedWindow("Result",0);//创建窗口，设置窗口属性标志，不能手动改变窗口大小
    //  cvMoveWindow("Result",750,0);
    //  cvResizeWindow("Result",300,200); //缩放窗口
    //     cvShowImage("Result",tempImage);
    //  cvWaitKey();
    //  cvDestroyAllWindows();
    ui->SlaveVideo2->setPixmap(QPixmap::fromImage(Mat2QImage(drawing)));
    //imshow("bianyuan",drawing);
    //imshow("imageSC",imageSC);
    //imshow("src",src);
    return false;
}

void MainWindow::getCentroid(Mat frame2)
{   Mat src=frame2;
    Mat src_gray=frame2;
    int thresh = 30;
    //int max_thresh = 255;
    GaussianBlur( src_gray, src_gray, Size(3,3), 0.1, 0, BORDER_DEFAULT );
    blur( src_gray, src_gray, Size(3,3) ); //滤波
    //namedWindow( "image", CV_WINDOW_AUTOSIZE );
    //imshow( "image", src );
    //moveWindow("image",20,20);
    //定义Canny边缘检测图像
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //利用canny算法检测边缘
    Canny( src_gray, canny_output, thresh, thresh*3, 3 );
    namedWindow( "canny", CV_WINDOW_AUTOSIZE );
    imshow( "canny", canny_output );
    moveWindow("canny",550,20);
    //查找轮廓
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //计算轮廓矩
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    {
        mu[i] = moments( contours[i], false );
    }
    //计算轮廓的质心
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    {
        mc[i] = Point2d( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }
    //画轮廓及其质心并显示
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( 255, 0, 0);
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        circle( drawing, mc[i], 5, Scalar( 0, 0, 255), -1, 8, 0 );
        rectangle(drawing, boundingRect(contours.at(i)), cvScalar(0,255,0));
        char tam[100];
        sprintf(tam, "(%0.0f,%0.0f)",mc[i].x,mc[i].y);
        putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(255,0,255),1);
    }
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    moveWindow("Contours",1100,20);
    //waitKey(0);
    //src.release();
    src_gray.release();
    //return 0;

}

//open camera
void MainWindow::openCamara(int X)
{
    switch(X){
    case 0:{
        if (capture.isOpened())
            capture.release();     //decide if capture is already opened; if so,close it
        capture.open(X);           //open the default camera
        if (capture.isOpened())
        {

            capture >> frame;
            if (!frame.empty())
            {

                image = Mat2QImage(frame);
            }
        }
        ui->MainVideo->setPixmap(QPixmap::fromImage(image));
        timer = new QTimer(this);
        timer->setInterval(33);   //set timer match with FPS
        connect(timer, SIGNAL(timeout()), this, SLOT(mainVideoNextFrame()));
        timer->start();
        break;
    }
    case 1:{
        if (capture1.isOpened())
            capture1.release();     //decide if capture is already opened; if so,close it
        capture1.open(X);           //open the default camera
        if (capture1.isOpened())
        {

            capture1 >> frame1;
            if (!frame1.empty())
            {

                image1 = Mat2QImage(frame1);
            }
        }
        ui->SlaveVideo1->setPixmap(QPixmap::fromImage(image1));
        timer1 = new QTimer(this);
        timer1->setInterval(33);   //set timer match with FPS
        connect(timer1, SIGNAL(timeout()), this, SLOT(slaveVideo1NextFrame()));
        timer1->start();
        break;
    }
    case 2:{
        if (capture2.isOpened())
            capture2.release();     //decide if capture is already opened; if so,close it
        capture2.open(X);           //open the default camera
        if (capture2.isOpened())
        {

            capture2 >> frame2;
            if (!frame2.empty())
            {

                image2 = Mat2QImage(frame2);
            }
        }
        ui->SlaveVideo1->setPixmap(QPixmap::fromImage(image2));
        timer2 = new QTimer(this);
        timer2->setInterval(33);   //set timer match with FPS
        connect(timer2, SIGNAL(timeout()), this, SLOT(slaveVideo1NextFrame()));
        timer2->start();
        break;
    }
    }
}

//open file
void MainWindow::openFile(int X)
{
    if (capture.isOpened())
        capture.release();     //decide if capture is already opened; if so,close it
    QString filename =QFileDialog::getOpenFileName(this,tr("Open Video File"),".",tr("Video Files(*.avi *.mp4 *.flv *.mkv)"));
    ui->plainTextEdit->appendPlainText(filename+"\n");
    //textEdit->setPlainText(filename+"\n");
    capture.open(filename.toLocal8Bit().data());
    if (capture.isOpened())
    {
        rate= capture.get(CV_CAP_PROP_FPS);
        capture >> frame;
        if (!frame.empty())
        {

            image = Mat2QImage(frame);
            switch(X){
            case 0:ui->MainVideo->setPixmap(QPixmap::fromImage(image));
                timer = new QTimer(this);
                timer->setInterval(1000/rate);   //set timer match with FPS
                connect(timer, SIGNAL(timeout()), this, SLOT(mainVideoNextFrame()));
                timer->start();
                break;
            case 1:ui->SlaveVideo1->setPixmap(QPixmap::fromImage(image));
                timer = new QTimer(this);
                timer->setInterval(1000/rate);   //set timer match with FPS
                connect(timer, SIGNAL(timeout()), this, SLOT(slaveVideo1NextFrame()));
                timer->start();
                break;
            case 2:ui->SlaveVideo2->setPixmap(QPixmap::fromImage(image));
                timer = new QTimer(this);
                timer->setInterval(1000/rate);   //set timer match with FPS
                connect(timer, SIGNAL(timeout()), this, SLOT(slaveVideo2NextFrame()));
                timer->start();
                break;
            }
        }
    }
}

//close camera
void MainWindow::closeCamara()
{
    timer->stop();         // 停止读取数据。
    capture.release();
    capture1.release();
}


void MainWindow::Picture(){
    capture1 >> Pframe;

    image = Mat2QImage(Pframe);

    QString filename=QString::fromStdString("cam");
    ui->plainTextEdit->appendPlainText(filename+PictureNum+"   Finish!"+"\n");
    imwrite("cam"+PictureNum,Pframe);
    PictureNum++;
    ui->SlaveVideo2->setPixmap(QPixmap::fromImage(image));  // 将图片显示到label上
}

//auto get next frame
void MainWindow::mainVideoNextFrame()
{
    capture >> frame;
    if (!frame.empty())
    {
        image = Mat2QImage(frame);
        ui->MainVideo->setPixmap(QPixmap::fromImage(image));
        this->update();
    }

}

void MainWindow::slaveVideo1NextFrame()
{
    capture1 >> frame1;
    if (!frame1.empty())
    {
        image1 = Mat2QImage(frame1);
        ui->SlaveVideo1->setPixmap(QPixmap::fromImage(image1));
        this->update();
    }

}

void MainWindow::slaveVideo2NextFrame()
{
    capture >> frame;
    if (!frame.empty())
    {
        image = Mat2QImage(frame);
        ui->SlaveVideo2->setPixmap(QPixmap::fromImage(image));
        this->update();
    }

}
//摄像机标定
void MainWindow::cameraCalibrator()
{   openCamara(0);
    vector<Point2f> imageCorners;
    Size boardSize(9, 6);
    Mat image1 = frame;//imread("left01.jpg");
    bool found = findChessboardCorners(image1, boardSize, imageCorners);
    //绘制角点
    drawChessboardCorners(image1, boardSize, imageCorners, found);
    //namedWindow("test");
    QImage TEMP=Mat2QImage(image1);
    ui->SlaveVideo1->setPixmap(QPixmap::fromImage(TEMP));  // 将图片显示到label上
    //imshow("test", image1);//角点如未全部检测出来只是红色圆圈画出角点
    waitKey();
}

void MainWindow::Correct()
{
    VideoCapture inputVideo(0);
    if (!inputVideo.isOpened())
    {
        QMessageBox::information(this, QString::fromLocal8Bit("WRONG"),QString::fromLocal8Bit("Could not open the input video"));

    }
    Mat frame2;
    Mat frameCalibration;

    inputVideo >> frame2;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 4.91087502052584e+02;
    cameraMatrix.at<double>(0, 1) = 2.29475435682419;
    cameraMatrix.at<double>(0, 2) = 3.38159283953158e+02;
    cameraMatrix.at<double>(1, 1) = 4.91857587706177e+02;
    cameraMatrix.at<double>(1, 2) = 2.05663605125609e+02;

    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.320311439187776;
    distCoeffs.at<double>(1, 0) = 0.117708464407889;
    distCoeffs.at<double>(2, 0) = -0.00548954846049678;
    distCoeffs.at<double>(3, 0) = 0.00141925006352090;
    distCoeffs.at<double>(4, 0) = 0;

    Mat view, rview, map1, map2;
    Size imageSize;
    imageSize = frame2.size();
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                            imageSize, CV_16SC2, map1, map2);


    while (1) //Show the image captured in the window and repeat
    {
        inputVideo >> frame2;              // read
        if (frame2.empty()) break;         // check if at end
        remap(frame2, frameCalibration, map1, map2, INTER_LINEAR);
        //QImage TEMP1=Mat2QImage(frame2);
        //ui->SlaveVideo1->setPixmap(QPixmap::fromImage(TEMP1));  // 将图片显示到label上
        //imshow("Origianl", frame2);
        //QImage TEMP2=Mat2QImage(frameCalibration);
        //ui->SlaveVideo2->setPixmap(QPixmap::fromImage(TEMP2));  // 将图片显示到label上
        imshow("Calibration",frameCalibration );
        char key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')break;
    }
}

void MainWindow::on_pushButton_clicked()
{
    openCamara(0);
    timer->start(33);
}

void MainWindow::on_pushButton_3_clicked()
{
    openFile(2);
}

void MainWindow::on_pushButton_2_clicked()
{

    cameraCalibrator();

}

void MainWindow::on_pushButton_4_clicked()
{
    closeCamara();
}

void MainWindow::on_pushButton_5_clicked()
{
    Correct();
}

void MainWindow::on_pushButton_6_clicked()
{

    while (1){
        MovingObjectDetection();
        QApplication::processEvents();
        //objectDetection(dframe, 100,100,  1);
        char key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')break;
    }
}

void MainWindow::mousePressEvent(QMouseEvent *m)
{
    if(m->buttons()==Qt::LeftButton){//如果鼠标按下的是左键
        /*//这里获取指针位置和窗口位置的差值
        offset=m->globalPos()-this->pos();*/
        //QString msg=QString("press:%1,%2").arg(m->pos().x()).arg(m->pos().y());
        //pos()-(centralWidget.pos()+leftImgLabel.pos())（窗口坐标减去父窗口坐标和当前窗口坐标）char_
        //0<=MouxePoint_X<=720,10<=MouxePoint_Y<=570
        QString MouxePoint_X=QString::number(m->pos().x(),10);
        QString MouxePoint_Y=QString::number(m->pos().y(),10);
        int DEC_MouxePoint_X=MouxePoint_X.toInt();
        int DEC_MouxePoint_Y=MouxePoint_Y.toInt();
        QByteArray ba,bb ; // must
        //char* char_MouxePoint_X = NULL;
        ba = MouxePoint_X.toLatin1(); // must
        //char_MouxePoint_X = strdup(ba.data()); //直接拷贝出来就不会乱码了
        char *char_MouxePoint_X=ba.data();//这个会乱码
        bb = MouxePoint_Y.toLatin1(); // must
        char *char_MouxePoint_Y=bb.data();
        if(ui->OpenButton->text()==tr("关闭串口")){
            if((DEC_MouxePoint_X>=0)&&(DEC_MouxePoint_X<=720))
            {
                if((DEC_MouxePoint_Y>=10)&&(DEC_MouxePoint_Y<=570))
                {
                    //serial->write("s");
                    serial->write(char_MouxePoint_X);
                    serial->write(",");
                    serial->write(char_MouxePoint_Y);
                    serial->write("\n");
                    ui->plainTextEdit->appendPlainText(MouxePoint_X+','+MouxePoint_Y+'\n');
                }
            }
        }
    }
}

void MainWindow::SerialSendCoordinate(QString CoordinateX,QString CoordinateY){
    int DEC_CoordinateX=CoordinateX.toInt();
    int DEC_CoordinateY=CoordinateY.toInt();
    QByteArray ba,bb ; // must
    ba = CoordinateX.toLatin1(); // must
    char *char_CoordinateX=ba.data();//这个会乱码
    bb = CoordinateY.toLatin1(); // must
    char *char_CoordinateY=bb.data();
    if(ui->OpenButton->text()==tr("关闭串口")){
        if((DEC_CoordinateX>=0)&&(DEC_CoordinateX<=720))
        {
            if((DEC_CoordinateY>=10)&&(DEC_CoordinateY<=570))
            {
                serial->write(char_CoordinateX);
                serial->write(",");
                serial->write(char_CoordinateY);
                serial->write("\n");
                ui->plainTextEdit->appendPlainText(CoordinateX+','+CoordinateY+'\n');
            }
        }
    }

}

void MainWindow::Read_Data()
{
    QByteArray buf;
    buf = serial->readAll();
    if(!buf.isEmpty())
    {
        QString str = ui->plainTextEdit->toPlainText();
        str+=tr(buf);
        ui->plainTextEdit->clear();
        ui->plainTextEdit->appendPlainText(str);
    }
    buf.clear();
}

void MainWindow::on_OpenButton_clicked()
{
    if(ui->OpenButton->text()==tr("打开串口"))
    {
        serial = new QSerialPort;
        //设置串口名
        serial->setPortName(ui->PortBox->currentText());
        //打开串口
        serial->open(QIODevice::WriteOnly);
        //设置波特率
        serial->setBaudRate(ui->BaudBox->currentText().toInt());
        //设置数据位数
        switch(ui->BitNumBox->currentIndex())
        {
        case 8: serial->setDataBits(QSerialPort::Data8); break;
        default: break;
        }
        //设置奇偶校验
        switch(ui->ParityBox->currentIndex())
        {
        case 0: serial->setParity(QSerialPort::NoParity); break;
        default: break;
        }
        //设置停止位
        switch(ui->StopBox->currentIndex())
        {
        case 1: serial->setStopBits(QSerialPort::OneStop); break;
        case 2: serial->setStopBits(QSerialPort::TwoStop); break;
        default: break;
        }
        //设置流控制
        serial->setFlowControl(QSerialPort::NoFlowControl);
        //关闭设置菜单使能
        ui->PortBox->setEnabled(false);
        ui->BaudBox->setEnabled(false);
        ui->BitNumBox->setEnabled(false);
        ui->ParityBox->setEnabled(false);
        ui->StopBox->setEnabled(false);
        ui->OpenButton->setText(tr("关闭串口"));
        //连接信号槽
        QObject::connect(serial, &QSerialPort::readyRead, this, &MainWindow::Read_Data);
    }

    else
    {
        //关闭串口
        serial->clear();
        serial->close();
        serial->deleteLater();
        //恢复设置使能
        ui->PortBox->setEnabled(true);
        ui->BaudBox->setEnabled(true);
        ui->BitNumBox->setEnabled(true);
        ui->ParityBox->setEnabled(true);
        ui->StopBox->setEnabled(true);
        ui->OpenButton->setText(tr("打开串口"));
    }
}

void MainWindow::on_pushButton_8_clicked()
{
    openCamara(1);
    timer1->start(33);
}

void MainWindow::on_pushButton_9_clicked()
{
    Picture();
}
