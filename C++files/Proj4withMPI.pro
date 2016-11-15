TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp
INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -larmadillo -lmpi

INCLUDEPATH += /usr/local/bin/mpic++

# MPI Settings
QMAKE_CXX = /usr/local/bin/mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc
