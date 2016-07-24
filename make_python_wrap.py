import numpy as np

prelude = '''
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#include "kmeans.h"

typedef float float32_t;
typedef double float64_t;

#define argument_error(message) PyErr_SetString(PyExc_TypeError, message);

'''

def numpy_to_thrust(T):
    return '''
thrust::host_vector<%(T)s> numpy_to_thrust_%(T)s(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(%(T)s)) {
        argument_error("wrong array type (expecting %(T)s)");
        throw 0;
    }

    npy_intp buffer_size = PyArray_NBYTES(data);
    printf("\\n%%d\\n", buffer_size);
    fflush(stdout);
    %(T)s *data_buffer = (%(T)s *) PyArray_DATA(data);

    thrust::host_vector<%(T)s> host_data(buffer_size / sizeof(%(T)s));
    for (size_t i=0; i < (buffer_size / sizeof(%(T)s)); i++) {
        host_data[i] = data_buffer[i];
    }

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

    int data_typenum = PyArray_TYPE(data);
    int ndim = PyArray_NDIM(data);
    void *host_centroids_ptr;

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
        host_centroids_ptr = thrust::raw_pointer_cast(host_centroids.data());

        npy_intp res_dims[2];
        res_dims[0] = k;
        res_dims[1] = d;
        PyObject *res_centroids = PyArray_SimpleNewFromData(2, res_dims, data_typenum, host_centroids_ptr);

        return res_centroids;
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
}''' % dict(T=T, to_thrust_def=numpy_to_thrust(T), to_thrust='numpy_to_thrust_%s' % T)

npy_types = (
    #'int8',
    #'int16',
    #'int32',
    #'int64',
    #'uint8',
    #'uint16',
    #'uint32',
    #'uint64',
    'float32',
    'float64'
)

npy_types = set(map(np.dtype, npy_types))

def dispatcher():
    arglist = 'iterations, k, data_arr, labels_arr, threshold'

    template = '''
extern "C" {
    static PyObject *py_kmeans(PyObject *self, PyObject *args) {
        printf("0");
        PyArrayObject *data_arr;
        PyArrayObject *labels_arr;
        int iterations;
        int k;
        double threshold;

        printf("%%p %%p\\n", PyArray_Type, &PyArray_Type);
        fflush(stdout);
        if (!PyArg_ParseTuple(args, "O!O!iid",
            &PyArray_Type, &data_arr,
            &PyArray_Type, &labels_arr, &iterations, &k, &threshold)) {
            return 0;
        }
        printf("b\\n");
        fflush(stdout);

        fflush(stdout);
        if (data_arr->ob_type != &PyArray_Type) {
            printf("2\\n");
            fflush(stdout);
            argument_error("data is not a numpy array");
            Py_XDECREF(data_arr);
            Py_XDECREF(labels_arr);
            return 0;
        }

        printf("c");
        fflush(stdout);

        if (labels_arr->ob_type != &PyArray_Type) {
            argument_error("labels is not a numpy array");
            Py_XDECREF(data_arr);
            Py_XDECREF(labels_arr);
            return 0;
        }

        printf("d");
        fflush(stdout);

        PyObject *res = 0;

        switch(PyArray_TYPE(data_arr)) {
            %s
            default:
                argument_error("unknown array type");
        }

        printf("e");
        fflush(stdout);

        Py_XDECREF(data_arr);
        Py_XDECREF(labels_arr);

        return res;
    }

    PyObject *device_count(PyObject *self) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        return PyInt_FromLong((long) device_count);
    }

    static PyMethodDef KmeansMethods[] = {
        {"kmeans", py_kmeans, METH_VARARGS, "run kmeans clustering using CUDA"},
        {"device_count", (PyCFunction) device_count, METH_NOARGS, "returns the number of CUDA-capable devices installed"},
        {NULL, NULL, 0, NULL}
    };

    PyMODINIT_FUNC initkmeans(void) {
        (void) Py_InitModule("kmeans", KmeansMethods);
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

    return '%s\n%s' % ('\n'.join(function_definitions), template % '\n'.join(cases))

def run(outf):
    outf.write(prelude)
    outf.write('\n')
    outf.write(dispatcher())


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as outf:
        run(outf)
