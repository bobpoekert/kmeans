#include <Python.h>
#include <numpy/arrayobject.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#include "kmeans.h"


template<typename T>
thrust::host_vector<T> numpy_to_thrust(PyArrayObject *data) {

    int n_dims = PyArray_NDIM(data);
    if (n_dims != sizeof(T)) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        return 0;
    }

    npy_intp buffer_size = PyArray_NBYTES(data);
    T *data_buffer = (T *) PyArray_DATA(data);

    thrust::host_vector<T> host_data(buffer_size);
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;

}

PyObject *py_kmeans_uint32(
        int iterations,
        int k,
        int debug,
        PyArrayObject *data,
        PyArrayObject *labels,
        double threshold) {

    int n_dims = PyArray_NDIM(data);
    if (n_dims != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        return 0;
    }

    npy_intp item_size = PyArray_ITEMSIZE(data);
    if (item_size != sizeof(uint32_t)) {
        PyErr_BadArgument("expected array of uint32_t, got something else");
        return 0;
    }

    npy_intp height = PyArray_DIM(data, 0);
    npy_intp width = PyArray_DIM(data, 1);
    
    npy_intp buffer_size = PyArray_NBYTES(data);
    uint32_t *data_buffer = (uint32_t *) PyArray_DATA(data);

    thrust::host_vector<uint32_t> host_buffer(buffer_size);
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    thrust::device_vector<uint32_t> device_buffer = host_buffer;

    

}
