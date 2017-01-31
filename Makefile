PYTHON_VERSION = 2.7

OPENCV_LIB = `pkg-config --libs opencv`
OPENCV_CFLAGS = `pkg-config --cflags opencv`

CFLAGS = -c -fPIC $(OPENCV_CFLAGS)
LFLAGS = -L/usr/local/cuda-6.5/lib $(OPENCV_LIB) -lpython$(PYTHON_VERSION)

TARGET = ImgProc
SRC = ImgProc.cpp conversion.cpp conversion.h
OBJ = ImgProc.o conversion.o


shared: $(OBJ)
	g++ -shared $(OBJ) $(LFLAGS) -o $(TARGET).so

$(OBJ): $(SRC)
	g++ $(CFLAGS) $(SRC) -o $(OBJ)

clean:
	rm -f $(OBJ)
