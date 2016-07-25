import numpy as np

prelude = '''
#include <stdio.h>
#include <Python.h>
#include <cuda_runtime.h>
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
}''' % dict(T=T, to_thrust_def=numpy_to_thrust(T), to_thrust='numpy_to_thrust_%s' % T)

npy_types = (
    'float32',
    'float64'
)

npy_types = set(map(np.dtype, npy_types))

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

        return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:(i,i,i), s:(i,i,i), s:(i,i), s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i}",
            "global_mem", (int) props.totalGlobalMem,
            "shared_mem_per_block", (int) props.sharedMemPerBlock,
            "regs_per_block", (int) props.regsPerBlock,
            "warp_size", (int) props.warpSize,
            "mem_pitch", (int) props.memPitch,
            "max_threads_per_block", (int) props.maxThreadsPerBlock,
            "max_threads_dim",
                (int) (props.maxThreadsDim[0]), (int) (props.maxThreadsDim[1]), (int) (props.maxThreadsDim[2]),
            "max_grid_size",
                (int) (props.maxGridSize[0]), (int) (props.maxGridSize[1]), (int) (props.maxGridSize[2]),
            "version", (int) props.major, (int) props.minor,
            "clock_rate", (int) props.clockRate,
            "texture_alignment", (int) props.textureAlignment,
            "device_overlap", (int) props.deviceOverlap,
            "multi_processor_count", (int) props.multiProcessorCount,
            "kernel_exec_timeout_enabled", (int) props.kernelExecTimeoutEnabled,
            "integrated", (int) props.integrated,
            "can_map_host_memory", (int) props.canMapHostMemory,
            "compute_mode", (int) props.computeMode,
            "concurrent_kernels", (int) props.concurrentKernels,
            "ecc_enabled", (int) props.ECCEnabled,
            "tcc_driver", (int) props.tccDriver);

    }

    static PyMethodDef KmeansMethods[] = {
        {"kmeans", py_kmeans, METH_VARARGS, "run kmeans clustering using CUDA"},
        {"device_props", device_props, METH_VARARGS, "takes a CUDA device id, and returns a dict of info about it"},
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
