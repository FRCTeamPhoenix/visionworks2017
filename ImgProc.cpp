#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudafilters.hpp>
#include <python2.7/Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace cv;

static PyObject* ImgProc_processFrameGPU(PyObject* self, PyObject* args)
{
    printf("Hello from cpp\n");

    PyArrayObject* frame;
    if (!PyArg_ParseTuple(args, "O", &frame))
    {
        return NULL;
    }

    uint8_t thresh_min_vals[] = {127, 127, 127};
    uint8_t thresh_max_vals[] = {255, 255, 255};
    //int morph_kernel_erode[] = {3, 3};
    //int morph_kernel_dilate[] = {25, 25};
    Size morph_kernel_erode(3, 3);
    Size morph_kernel_dilate(25, 25);

    // numpy array created from a PIL image is 3-dimensional:
    // height x width x num_channels (num_channels being 3 for RGB)
    assert (PyArray_NDIM(frame) == 3 && PyArray_SHAPE(frame)[2] == 3 /*Incorrect dimensions or colors*/);
    
    // Extract the metainformation and the data.
    int rows = PyArray_SHAPE(frame)[0];
    int cols = PyArray_SHAPE(frame)[1];
    void *src_frame_data = PyArray_DATA(frame);
    
    // Construct the Mat object and use it.
    Mat cframe(rows, cols, CV_8UC3, src_frame_data);
    
    static gpu::GpuMat gframe, gmask;
    gframe.upload(cframe);
    printf("Data uploaded to GPU\n");
    /*
    // convert to HSV and download back to cv_frame for thresholding
    gpu::cvtColor(gframe, gframe, CV_BGR2HSV);
    gframe.download(cframe);
    
    Mat cmask(rows, cols, CV_8UC1);
    uint8_t* frame_data = cframe.data;
    int frame_stride = cframe.step;

    uint8_t* mask_data = cmask.data;
    int mask_stride = cmask.step;

    // threshold (move this to CUDA kernel?)
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            uint8_t h = frame_data[3 * (j + i * frame_stride)];
            uint8_t s = frame_data[3 * (j + i * frame_stride) + 1];
            uint8_t v = frame_data[3 * (j + i * frame_stride) + 2]; 
            bool h_inrange = h > thresh_min_vals[0] && h < thresh_max_vals[0];
            bool s_inrange = s > thresh_min_vals[1] && s < thresh_max_vals[1];
            bool v_inrange = v > thresh_min_vals[2] && v < thresh_max_vals[2];
            
            mask_data[j + i * mask_stride] = h_inrange & s_inrange & v_inrange;
        }
    }
    gmask.upload(cmask); 
    
    // do morph stuff to remove noise (slow, has to be on GPU)
    Mat kernel_erode = getStructuringElement(MORPH_ELLIPSE, morph_kernel_erode);
    gpu::erode(gmask, gmask, kernel_erode);
    Mat kernel_dilate = getStructuringElement(MORPH_ELLIPSE, morph_kernel_dilate);
    gpu::dilate(gmask, gmask, kernel_dilate);
    
    // edge detect (done in C++ to make returning values easier lol) 
    gmask.download(cmask);
    std::vector< std::vector<Mat> > contours;
    
    findContours(cmask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    */
    // TODO: return frame? return contours? which is easier
    return Py_BuildValue("i", 0);
}

static PyMethodDef ImgProcMethods[] = {
    {"processFrameGPU",  ImgProc_processFrameGPU, METH_VARARGS,
     "Process a frame on the GPU"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initImgProc(void)
{
    (void) Py_InitModule("ImgProc", ImgProcMethods);
}
