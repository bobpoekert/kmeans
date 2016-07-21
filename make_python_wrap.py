import numpy as np

prelude = '''
#include <Python.h>
#include <numpy/arrayobject.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#include "kmeans.h"

typedef float float32_t;
typedef double float64_t;

'''

def numpy_to_thrust(T):
    return '''
thrust::host_vector<%(T)s> numpy_to_thrust_%(T)s(PyArrayObject *data) {

    if (PyArray_ITEMSIZE(data) != sizeof(%(T)s)) {
        PyErr_BadValue("wrong array type (expecting %(T)s)");
        throw 0;
    }

    int n_dims = PyArray_NDIM(data);

    npy_intp buffer_size = PyArray_NBYTES(data);
    %(T)s *data_buffer = (%(T)s *) PyArray_DATA(data);

    thrust::host_vector<%(T)s> host_data(buffer_size / sizeof(%(T)s));
    for (size_t i=0; i < buffer_size; i++) {
        host_buffer[i] = data_buffer[i];
    }

    return host_data;
}
''' % locals()

def kmeans(T):
    return '''
%(to_thrust_def)s

PyObject *py_kmeans_%(T)s(
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
        thrust::host_vector<%(T)s> host_data = %(to_thrust)s(data);
        thrust::host_vector<int> host_labels = numpy_to_thrust_int(labels);

        thrust::device_vector<%(T)s> device_data = host_data;
        thrust::device_vector<int> device_labels = host_labels;

        thrust::device_vector<%(T)s> centroids(k * d);
        thrust::device_vecotr<%(T)s> distances(n);

        kmeans(
        iterations, n, d, k, device_data, device_labels, device_centroids, device_distances,
        true, false, threshold);

        thrust::host_vector<%(T)s> host_centroids = centroids;

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

}''' % dict(T=T, to_thrust_def=numpy_to_thrust(T), to_thrust='numpy_to_thrust_%s' % T)

npy_types = (
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float32',
    'float64'
)

npy_types = set(map(np.dtype, npy_types))

def dispatcher():
    arglist = 'iterations, k, data, labels, threshold'

    template = '''
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

        if (!PyArray_Check(data)) {
            PyErr_BadValue("data is not a numpy array");
            return 0;
        }

        if (!PyArray_Check(labels)) {
            PyErr_BadValue("labels is not a numpy array");
            return 0;
        }

        PyArrayObject *data_arr = (PyArrayObject *) data;
        PyArrayObject *labels_arr = (PyArrayObject *) labels;

        switch(PyArray_TYPE(data_arr)) {
            %s
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
    '''

    function_definitions = [numpy_to_thrust('int')]
    cases = []
    for dtype in npy_types:
        type_id = dtype.num
        type_cname = '%s_t' % dtype.name
        function_definitions.append(kmeans(type_cname))
        cases.append('''
                case %d:
                    return py_kmeans_%s(%s);
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
