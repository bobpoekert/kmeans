
#include <Python.h>
#include <numpy/arrayobject.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#include "kmeans.h"


thrust::host_vector<int> numpy_to_thrust_int(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(int)) {
        PyErr_BadValue("wrong array type (expecting int)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    int *data_buffer = (int *) PyArray_DATA(data);

    thrust::host_vector<int> host_data(buffer_size / sizeof(int));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}



thrust::host_vector<int8_t> numpy_to_thrust_int8_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(int8_t)) {
        PyErr_BadValue("wrong array type (expecting int8_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    int8_t *data_buffer = (int8_t *) PyArray_DATA(data);

    thrust::host_vector<int8_t> host_data(buffer_size / sizeof(int8_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_int8_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<int8_t> host_data = numpy_to_thrust_int8_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<int8_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<int8_t> centroids(k * d);
        thrust::device_vecotr<int8_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<int8_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<float64_t> numpy_to_thrust_float64_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(float64_t)) {
        PyErr_BadValue("wrong array type (expecting float64_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    float64_t *data_buffer = (float64_t *) PyArray_DATA(data);

    thrust::host_vector<float64_t> host_data(buffer_size / sizeof(float64_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_float64_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<float64_t> host_data = numpy_to_thrust_float64_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<float64_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<float64_t> centroids(k * d);
        thrust::device_vecotr<float64_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<float64_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<uint16_t> numpy_to_thrust_uint16_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(uint16_t)) {
        PyErr_BadValue("wrong array type (expecting uint16_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    uint16_t *data_buffer = (uint16_t *) PyArray_DATA(data);

    thrust::host_vector<uint16_t> host_data(buffer_size / sizeof(uint16_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_uint16_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<uint16_t> host_data = numpy_to_thrust_uint16_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<uint16_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<uint16_t> centroids(k * d);
        thrust::device_vecotr<uint16_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<uint16_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<uint8_t> numpy_to_thrust_uint8_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(uint8_t)) {
        PyErr_BadValue("wrong array type (expecting uint8_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    uint8_t *data_buffer = (uint8_t *) PyArray_DATA(data);

    thrust::host_vector<uint8_t> host_data(buffer_size / sizeof(uint8_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_uint8_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<uint8_t> host_data = numpy_to_thrust_uint8_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<uint8_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<uint8_t> centroids(k * d);
        thrust::device_vecotr<uint8_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<uint8_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<int64_t> numpy_to_thrust_int64_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(int64_t)) {
        PyErr_BadValue("wrong array type (expecting int64_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    int64_t *data_buffer = (int64_t *) PyArray_DATA(data);

    thrust::host_vector<int64_t> host_data(buffer_size / sizeof(int64_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_int64_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<int64_t> host_data = numpy_to_thrust_int64_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<int64_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<int64_t> centroids(k * d);
        thrust::device_vecotr<int64_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<int64_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<float32_t> numpy_to_thrust_float32_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(float32_t)) {
        PyErr_BadValue("wrong array type (expecting float32_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    float32_t *data_buffer = (float32_t *) PyArray_DATA(data);

    thrust::host_vector<float32_t> host_data(buffer_size / sizeof(float32_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_float32_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<float32_t> host_data = numpy_to_thrust_float32_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<float32_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<float32_t> centroids(k * d);
        thrust::device_vecotr<float32_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<float32_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<int16_t> numpy_to_thrust_int16_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(int16_t)) {
        PyErr_BadValue("wrong array type (expecting int16_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    int16_t *data_buffer = (int16_t *) PyArray_DATA(data);

    thrust::host_vector<int16_t> host_data(buffer_size / sizeof(int16_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_int16_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<int16_t> host_data = numpy_to_thrust_int16_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<int16_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<int16_t> centroids(k * d);
        thrust::device_vecotr<int16_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<int16_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<uint64_t> numpy_to_thrust_uint64_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(uint64_t)) {
        PyErr_BadValue("wrong array type (expecting uint64_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    uint64_t *data_buffer = (uint64_t *) PyArray_DATA(data);

    thrust::host_vector<uint64_t> host_data(buffer_size / sizeof(uint64_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_uint64_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<uint64_t> host_data = numpy_to_thrust_uint64_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<uint64_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<uint64_t> centroids(k * d);
        thrust::device_vecotr<uint64_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<uint64_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<uint32_t> numpy_to_thrust_uint32_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(uint32_t)) {
        PyErr_BadValue("wrong array type (expecting uint32_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    uint32_t *data_buffer = (uint32_t *) PyArray_DATA(data);

    thrust::host_vector<uint32_t> host_data(buffer_size / sizeof(uint32_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_uint32_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<uint32_t> host_data = numpy_to_thrust_uint32_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<uint32_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<uint32_t> centroids(k * d);
        thrust::device_vecotr<uint32_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<uint32_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<int32_t> numpy_to_thrust_int32_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(int32_t)) {
        PyErr_BadValue("wrong array type (expecting int32_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    int32_t *data_buffer = (int32_t *) PyArray_DATA(data);

    thrust::host_vector<int32_t> host_data(buffer_size / sizeof(int32_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_int32_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<int32_t> host_data = numpy_to_thrust_int32_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<int32_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<int32_t> centroids(k * d);
        thrust::device_vecotr<int32_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<int32_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}


thrust::host_vector<float16_t> numpy_to_thrust_float16_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(float16_t)) {
        PyErr_BadValue("wrong array type (expecting float16_t)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    float16_t *data_buffer = (float16_t *) PyArray_DATA(data);

    thrust::host_vector<float16_t> host_data(buffer_size / sizeof(float16_t));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}


PyObject *py_kmeans_float16_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int data_typenum = PyArray_TYPE(data);
    int labels_typenum = PyArray_TYPE(labels);
    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        PyErr_BadArgument("data must be rows of vectors (2d matrix)");
        throw 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    Py_BEGIN_ALLOW_THREADS
    try {
        thrust::host_vector<float16_t> host_data = numpy_to_thrust_float16_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<float16_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<float16_t> centroids(k * d);
        thrust::device_vecotr<float16_t> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<float16_t> host_centroids = centroids;

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids::data());

        return res_centroids;


    } catch(...) {
        return 0;
    } finally {
        Py_END_ALLOW_THREADS
    }

}

extern "C" {
    static PyObject *py_kmeans(PyObject *self, PyObject *args) {
        PyObject *data;
        PyObject *labels;
        int iterations;
        int k;
        double threshold;

        if (!PyArgs_ParseTuple(args, "ooiid", &data, &labels, &iterations, &k, &threshold)) {
            return 0;
        }

        PyArrayObject *data_arr = (PyArrayObject *) data;
        PyArrayObject *labels_arr = (PyArrayObject *) labels;

        switch(PyArray_TYPE(data_arr)) {
            
                case 1:
                    return py_kmeans_int8_t(iterations, k, data, labels, threshold);
        

                case 12:
                    return py_kmeans_float64_t(iterations, k, data, labels, threshold);
        

                case 4:
                    return py_kmeans_uint16_t(iterations, k, data, labels, threshold);
        

                case 2:
                    return py_kmeans_uint8_t(iterations, k, data, labels, threshold);
        

                case 7:
                    return py_kmeans_int64_t(iterations, k, data, labels, threshold);
        

                case 11:
                    return py_kmeans_float32_t(iterations, k, data, labels, threshold);
        

                case 3:
                    return py_kmeans_int16_t(iterations, k, data, labels, threshold);
        

                case 8:
                    return py_kmeans_uint64_t(iterations, k, data, labels, threshold);
        

                case 6:
                    return py_kmeans_uint32_t(iterations, k, data, labels, threshold);
        

                case 5:
                    return py_kmeans_int32_t(iterations, k, data, labels, threshold);
        

                case 23:
                    return py_kmeans_float16_t(iterations, k, data, labels, threshold);
        
            default:
                PyErr_BadValue("unknown array type");
                return 0;
        }
    }


    static PyMethodDef KmeansMethods[] = {
        {"kmeans", py_kmeans, METH_VARARGS, "run kmeans clustering using CUDA"},
        {NULL, NULL, 0, NULL}
    };

    PyMODINIT_FUNC initkmeans(void) {
        (void) Py_InitModule("kmeans", KmeansMethods);
    }
}
    