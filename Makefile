PYTHON_VERSION = 2.7

OPENCV_LIB = `pkg-config --libs opencv`
OPENCV_CFLAGS = `pkg-config --cflags opencv`

CFLAGS = -c -fPIC $(OPENCV_CFLAGS)

TARGET = ImgProc
SRC = ImgProc.cpp conversion.cpp conversion.h
OBJ = ImgProc.o conversion.o


shared: $(OBJ)
	g++ -shared $(OBJ) $(OPENCV_LIB) -lpython$(PYTHON_VERSION) -o $(TARGET).so

%.o: %.cpp %.h
	g++ $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJ)
