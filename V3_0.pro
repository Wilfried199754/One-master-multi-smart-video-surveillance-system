#-------------------------------------------------
#
# Project created by QtCreator 2018-03-27T20:11:10
#
#-------------------------------------------------

QT       += core gui
QT       += serialport
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = V3_0
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH += D:\OpenCV3.2.0\opencv\build\include \
   D:\OpenCV3.2.0\opencv\build\include\opencv  \
   D:\OpenCV3.2.0\opencv\build\include\opencv2
LIBS += D:\OpenCV3.2.0\opencv\build\x64\vc14\lib\opencv_world320d.lib


SOURCES += \
        main.cpp \
        mainwindow.cpp


HEADERS += \
        mainwindow.h


FORMS += \
        mainwindow.ui

DISTFILES +=
