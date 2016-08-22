import numpy as np

prelude = '''
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


'''

def numpy_to_thrust(T):
    return '''
thrust::host_vector<%(T)s> numpy_to_thrust_%(T)s(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(%(T)s)) {
        argument_error("wrong array type (expecting %(T)s)");
        throw 0;
    }

    npy_intp buffer_size = PyArray_NBYTES(data);
    %(T)s *data_buffer = (%(T)s *) PyArray_DATA(data);

    thrust::host_vector<%(T)s> host_data(buffer_size / sizeof(%(T)s));
    thrust::copy(data_buffer, data_buffer + (buffer_size / sizeof(%(T)s)), host_data.begin());

    return host_data;
}
''' % locals()

def kmeans(T):
    return '''
namespace kmeans {
%(to_thrust_def)s

PyObject *py_kmeans_%(T)s(
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

        thrust::host_vector<%(T)s> host_data = %(to_thrust)s(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<%(T)s> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<%(T)s> device_centroids(k * d);
        thrust::device_vector<%(T)s> device_distances(n);

        kmeans::kmeans(
            iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
            true, false, threshold);

        thrust::host_vector<%(T)s> host_centroids(device_centroids);

        thrust::host_vector<int> host_labels_res(device_labels);

        if (host_centroids.size() != k*d) {
            PyErr_SetString(PyExc_RuntimeError, "incorrect size for host_centroids!!!!(internal error)");
            return 0;
        }

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;

        PyObject *res_centroids = pyarray_from_data(2, res_dims, %(typenum)s, sizeof(%(T)s), host_centroids.data());
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
}''' % dict(
        T=T, typenum=typenum(T),
        to_thrust_def=numpy_to_thrust(T), to_thrust='numpy_to_thrust_%s' % T)

npy_types = (
    'float32',
    'float64'
)

npy_types = set(map(np.dtype, npy_types))

def typenum(T):
    return np.dtype(T.split('_')[0]).num

def dispatcher():
    arglist = 'iterations, k, data_arr, labels_arr, threshold'

    template = '''
extern "C" {
#define check(cond, message) \
        if (cond) {\
            PyErr_SetString(PyExc_RuntimeError, message);\
            Py_XDECREF(data_arr);\
            Py_XDECREF(labels_arr);\
            return 0;\
        }
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
        check(PyArray_TYPE(labels_arr) != %(int32_id)s, "labels is not np.int32")

        PyObject *res = 0;

        switch(PyArray_TYPE(data_arr)) {
            %(cases)s
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
    '''

    function_definitions = [numpy_to_thrust('int')]
    cases = []
    for dtype in npy_types:
        type_id = dtype.num
        type_cname = '%s_t' % dtype.name
        function_definitions.append(kmeans(type_cname))
        cases.append('''
                case %d:
                    res =  kmeans::py_kmeans_%s(%s);
                    break;
        ''' % (type_id, type_cname, arglist))

    return '%s\n%s' % ('\n'.join(function_definitions),
            template % {'cases':'\n'.join(cases), 'int32_id':np.dtype('int32').num})

def run(outf):
    outf.write(prelude)
    outf.write('\n')
    outf.write(dispatcher())


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as outf:
        run(outf)
