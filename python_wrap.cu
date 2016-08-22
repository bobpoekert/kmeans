
#include <stdio.h>
#include <Python.h>
#include <cuda_runtime.h>
#include <numpy/arrayobject.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#include <execinfo.h>
#include "kmeans.h"

typedef float float32_t;
typedef double float64_t;

#define argument_error(message) PyErr_SetString(PyExc_TypeError, message);

PyObject *pyarray_from_data(size_t n_dims, npy_intp *sizes, int typenum, size_t itemsize, void *data) {
    size_t buffer_size = 0;
    for (int i=0; i < n_dims; i++) {
        buffer_size += sizes[i];
    }
    buffer_size *= itemsize;
    void *buffer = malloc(buffer_size);
    if (!buffer) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate result buffer");
        return 0;
    }
    memcpy(buffer, data, buffer_size);
    PyArrayObject *res = (PyArrayObject *) PyArray_SimpleNewFromData(n_dims, sizes, typenum, buffer);
    if (!res) {
        free(buffer);
        return 0;
    }
    Py_INCREF(res);
    if (PyArray_ITEMSIZE(res) != itemsize) {
        PyErr_SetString(PyExc_RuntimeError, "invalid item size");
        Py_DECREF(res);
        free(buffer);
        return 0;
    }
    PyArray_ENABLEFLAGS(res, NPY_ARRAY_OWNDATA);
    return (PyObject *) res;
}




thrust::host_vector<int> numpy_to_thrust_int(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(int)) {
        argument_error("wrong array type (expecting int)");
        throw 0;
    }

    npy_intp buffer_size = PyArray_NBYTES(data);
    int *data_buffer = (int *) PyArray_DATA(data);

    thrust::host_vector<int> host_data(buffer_size / sizeof(int));
    thrust::copy(data_buffer, data_buffer + (buffer_size / sizeof(int)), host_data.begin());

    return host_data;
}


namespace kmeans {

thrust::host_vector<float64_t> numpy_to_thrust_float64_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(float64_t)) {
        argument_error("wrong array type (expecting float64_t)");
        throw 0;
    }

    npy_intp buffer_size = PyArray_NBYTES(data);
    float64_t *data_buffer = (float64_t *) PyArray_DATA(data);

    thrust::host_vector<float64_t> host_data(buffer_size / sizeof(float64_t));
    thrust::copy(data_buffer, data_buffer + (buffer_size / sizeof(float64_t)), host_data.begin());

    return host_data;
}


PyObject *py_kmeans_float64_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        argument_error("data must be rows of vectors (2d matrix)");
        return 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    try {

        thrust::host_vector<float64_t> host_data = numpy_to_thrust_float64_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<float64_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<float64_t> device_centroids(k * d);
        thrust::device_vector<float64_t> device_distances(n);

        kmeans::kmeans(
            iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
            true, false, threshold);

        thrust::host_vector<float64_t> host_centroids(device_centroids);

        thrust::host_vector<int> host_labels_res(device_labels);

        if (host_centroids.size() != k*d) {
            PyErr_SetString(PyExc_RuntimeError, "incorrect size for host_centroids!!!!(internal error)");
            return 0;
        }

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;

        PyObject *res_centroids = pyarray_from_data(2, res_dims, 12, sizeof(float64_t), host_centroids.data());
        if (!res_centroids) return 0;

        npy_intp label_dims[1];
        label_dims[0] = d;
        PyObject *res_labels = pyarray_from_data(1, label_dims, 5, sizeof(int), host_labels_res.data());
        if (!res_labels) return 0;

        return Py_BuildValue("(O,O)", res_centroids, res_labels);
    } catch(thrust::system::system_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch(std::bad_alloc &e) {
        PyErr_SetString(PyExc_MemoryError, "Out of GPU memory");
        return 0;
    } catch(std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "Caught unexpected C++ exception");
        return 0;
    }
}
}

namespace kmeans {

thrust::host_vector<float32_t> numpy_to_thrust_float32_t(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(float32_t)) {
        argument_error("wrong array type (expecting float32_t)");
        throw 0;
    }

    npy_intp buffer_size = PyArray_NBYTES(data);
    float32_t *data_buffer = (float32_t *) PyArray_DATA(data);

    thrust::host_vector<float32_t> host_data(buffer_size / sizeof(float32_t));
    thrust::copy(data_buffer, data_buffer + (buffer_size / sizeof(float32_t)), host_data.begin());

    return host_data;
}


PyObject *py_kmeans_float32_t(
    int iterations,
    int k,
    PyArrayObject *data,
    PyArrayObject *labels,
    double threshold) {

    int ndim = PyArray_NDIM(data);

    if (ndim != 2) {
        argument_error("data must be rows of vectors (2d matrix)");
        return 0;
    }

    int n = PyArray_DIM(data, 0);
    int d = PyArray_DIM(data, 1);

    try {

        thrust::host_vector<float32_t> host_data = numpy_to_thrust_float32_t(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<float32_t> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<float32_t> device_centroids(k * d);
        thrust::device_vector<float32_t> device_distances(n);

        kmeans::kmeans(
            iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
            true, false, threshold);

        thrust::host_vector<float32_t> host_centroids(device_centroids);

        thrust::host_vector<int> host_labels_res(device_labels);

        if (host_centroids.size() != k*d) {
            PyErr_SetString(PyExc_RuntimeError, "incorrect size for host_centroids!!!!(internal error)");
            return 0;
        }

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;

        PyObject *res_centroids = pyarray_from_data(2, res_dims, 11, sizeof(float32_t), host_centroids.data());
        if (!res_centroids) return 0;

        npy_intp label_dims[1];
        label_dims[0] = d;
        PyObject *res_labels = pyarray_from_data(1, label_dims, 5, sizeof(int), host_labels_res.data());
        if (!res_labels) return 0;

        return Py_BuildValue("(O,O)", res_centroids, res_labels);
    } catch(thrust::system::system_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch(std::bad_alloc &e) {
        PyErr_SetString(PyExc_MemoryError, "Out of GPU memory");
        return 0;
    } catch(std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "Caught unexpected C++ exception");
        return 0;
    }
}
}

extern "C" {
#define check(cond, message)         if (cond) {            PyErr_SetString(PyExc_RuntimeError, message);            Py_XDECREF(data_arr);            Py_XDECREF(labels_arr);            return 0;        }
    static PyObject *py_kmeans(PyObject *self, PyObject *args) {
        PyArrayObject *data_arr = 0;
        PyArrayObject *labels_arr = 0;
        int iterations;
        int k;
        double threshold;

        if (!PyArg_ParseTuple(args, "O!O!iid",
            &PyArray_Type, &data_arr,
            &PyArray_Type, &labels_arr, &iterations, &k, &threshold)) {
            return 0;
        }

        check(!PyArray_ISCARRAY(data_arr), "data is not C contiguous")
        check(!PyArray_ISCARRAY(labels_arr), "labels is not C contiguous")
        check(PyArray_TYPE(labels_arr) != 5, "labels is not np.int32")

        PyObject *res = 0;

        switch(PyArray_TYPE(data_arr)) {
            
                case 12:
                    res =  kmeans::py_kmeans_float64_t(iterations, k, data_arr, labels_arr, threshold);
                    break;
        

                case 11:
                    res =  kmeans::py_kmeans_float32_t(iterations, k, data_arr, labels_arr, threshold);
                    break;
        
            default:
                argument_error("unknown array type");
        }

        Py_XDECREF(data_arr);
        Py_XDECREF(labels_arr);

        return res;
    }

    PyObject *device_count(PyObject *self) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        return PyInt_FromLong((long) device_count);
    }

    PyObject *device_props(PyObject *self, PyObject *args) {
        int device_id = 0;
        if (!PyArg_ParseTuple(args, "i", &device_id)) {
            return 0;
        }
        cudaDeviceProp props;
        if(cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
            PyErr_SetString(PyExc_IndexError, "invalid device id");
            return 0;
        }

        return PyString_FromStringAndSize((char *) &props, sizeof(cudaDeviceProp));
    }

    PyObject *set_device(PyObject *self, PyObject *args) {
        int device_id = 0;
        if (!PyArg_ParseTuple(args, "i", &device_id)) {
            return 0;
        }
        if (cudaSetDevice(device_id) != cudaSuccess) {
            PyErr_SetString(PyExc_RuntimeError, "failed to set device id");
            return 0;
        }
        Py_RETURN_NONE;
    }

    static PyMethodDef KmeansMethods[] = {
        {"kmeans", py_kmeans, METH_VARARGS, "run kmeans clustering using CUDA"},
        {"device_props", device_props, METH_VARARGS, "takes a CUDA device id, and returns a dict of info about it"},
        {"set_device", set_device, METH_VARARGS, "set the id of the CUDA device to use"},
        {"device_count", (PyCFunction) device_count, METH_NOARGS, "returns the number of CUDA-capable devices installed"},
        {NULL, NULL, 0, NULL}
    };

    PyMODINIT_FUNC init_kmeans(void) {
        (void) Py_InitModule("_kmeans", KmeansMethods);
        import_array();
    }
}
    