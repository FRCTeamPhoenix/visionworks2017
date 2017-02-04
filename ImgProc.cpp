#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <python2.7/Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include "conversion.h"

namespace py = boost::python;
using namespace cv;

static PyObject* ImgProc_processFrameGPU(PyObject* self, PyObject* args)
{
    //printf("Hello from cpp\n");
    NDArrayConverter cvt;    

    PyArrayObject* frame;
    if (!PyArg_ParseTuple(args, "O", &frame))
    {
        return NULL;
    }

    uint8_t thresh_min_vals[] = {127, 127, 127};
    uint8_t thresh_max_vals[] = {255, 255, 255};
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
    
    static gpu::GpuMat gframe;
    static gpu::GpuMat gmask(rows, cols, CV_8UC1);
    static gpu::GpuMat gchannels[3];

    gframe.upload(cframe);
    //printf("Data uploaded to GPU\n");
    
    // convert to HSV and download back to cv_frame for thresholding
    gpu::cvtColor(gframe, gframe, CV_BGR2HSV);
    //printf("Pre-download\n");
    gframe.download(cframe);
    
    //printf("Post-download\n");

    Mat cmask = Mat::zeros(rows, cols, CV_8UC1);

    gpu::split(gframe, gchannels);
    gpu::threshold(gchannels[0], gchannels[0], thresh_min_vals[0], thresh_max_vals[0], THRESH_BINARY);
    gpu::threshold(gchannels[1], gchannels[1], thresh_min_vals[1], thresh_max_vals[1], THRESH_BINARY);
    gpu::threshold(gchannels[2], gchannels[2], thresh_min_vals[2], thresh_max_vals[2], THRESH_BINARY);
    
    gpu::bitwise_and(gchannels[0], gchannels[1], gmask);
    gpu::bitwise_and(gchannels[2], gmask, gmask);    

    //printf("After threshold\n");
    gmask.upload(cmask); 

    //printf("After upload");
    // do morph stuff to remove noise (slow, has to be on GPU)
    
    Mat kernel_erode = getStructuringElement(MORPH_ELLIPSE, morph_kernel_erode);
    gpu::erode(gmask, gmask, kernel_erode);
    Mat kernel_dilate = getStructuringElement(MORPH_ELLIPSE, morph_kernel_dilate);
    gpu::dilate(gmask, gmask, kernel_dilate);
    
    //printf("After kernel\n");
    gmask.download(cmask);
    
    //printf("After download\n");
    //static int dims[] = {rows, cols};
    //PyObject* np_array = PyArray_SimpleNewFromData(2, &dims[0], NPY_UINT8, &cmask.data[0]);
    //printf("After conversion to np array\n");
    
    //findContours(cmask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    

    //printf("End cpp processing\n");
    // TODO: return frame? return contours? which is easier
    return cvt.toNDArray(cmask);
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
