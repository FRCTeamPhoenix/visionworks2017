PROJECT = ImgProc

PYTHON_VERSION = 2.7

OPENCV_LIB = `pkg-config --libs opencv`
OPENCV_CFLAGS = `pkg-config --cflags opencv`

CFLAGS = -c -fPIC $(OPENCV_CFLAGS) -I/usr/include/python$(PYTHON_VERSION)
LFLAGS = -L/usr/local/cuda-6.5/lib $(OPENCV_LIB) -lpython$(PYTHON_VERSION) -lboost_python

TARGET = $(PROJECT).so
SRC = $(PROJECT).cpp conversion.cpp conversion.h
OBJ = $(PROJECT).o conversion.o


shared: $(OBJ)
	g++ -shared $(OBJ) $(LFLAGS) -o $(TARGET)

%.o: %.cpp
	g++ $(CFLAGS) $^ -o $@

clean:
	rm -f $(OBJ)
