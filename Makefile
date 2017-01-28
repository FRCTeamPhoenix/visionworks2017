PYTHON_VERSION = 2.7

OPENCV_LIB = `pkg-config --libs opencv`
OPENCV_CFLAGS = `pkg-config --cflags opencv`

CFLAGS = -c -fPIC $(OPENCV_CFLAGS)

TARGET = ImgProc
SRC = ImgProc.cpp
OBJ = ImgProc.o


shared: $(OBJ)
	g++ -shared $(OBJ) $(OPENCV_LIB) -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(OBJ):
	g++ $(CFLAGS) $(SRC) -o $(OBJ)

clean:
	rm -f $(OBJ)
