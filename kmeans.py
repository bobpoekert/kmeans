import pycuda.autoinit, pycuda.driver
from pycuda.compiler import SourceModule
import numpy as np
import os, re, random

current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

dtype_names = {
        np.int32:'long',
        np.int16:'short',
        np.int64:'long long',
        np.int8:'char',
        np.uint8:'unsigned char',
        np.uint16:'unsigned short',
        np.uint32:'unsigned long',
        np.uint64:'unsigned long long',
        np.float32:'float',
        np.float64:'double'}

def c_dtype(dtype):
    try:
        return dtype_names[dtype]
    except KeyError:
        raise TypeError('no type specialization available for type: %r' % dtype)

def interpolate(template, _locals):
    return re.sub(r'{{(.*?)}}', lambda m: eval(m.group(1), globals(), _locals), template)

def to_vec(dtypev, pointer, res, size):
    try:
        dtype = c_dtype(dtypev)
    except:
        dtype = dtypev
    size_varname = '_%d' % int(random.random() * 1000)
    return interpolate('''
        size_t {{size_varname}} = {{size}};
        thrust::device_ptr<{{dtype}}> t{{pointer}} = thrust::device_pointer_cast({{pointer}});
        thrust::device_vector<{{dtype}}> {{res}}({{size_varname}});
        thrust::copy({{res}}.begin(), {{res}}.end(), t{{pointer}});
    ''', locals())

class KmeansKernel(object):

    def __init__(self, dtypev):
        self.dtype = dtypev
        dtype = c_dtype(dtypev)
        code = interpolate('''
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include "kmeans.h"

extern "C" {

    void pykmeans(int *params, {{dtype}} *data, int *labels, {{dtype}} *centroids, {{dtype}} *distances) {

        int iterations = params[0];
        int n = params[1];
        int d = params[2];
        int k = params[3];

        {{to_vec(dtype, 'data', 'data_thrust', 'n*d')}}
        {{to_vec('int', 'labels', 'labels_thrust', 'n')}}
        {{to_vec(dtype, 'centroids', 'centroids_thrust', 'k*d')}}
        {{to_vec(dtype, 'distances', 'distances_thrust', 'n')}}

        kmeans::kmeans(iterations, n, d, k, data_thrust, labels_thrust, centroids_thrust, distances_thrust);

        thrust::copy(centroids_thrust.begin(), centroids_thrust.end(), centroids);
        thrust::copy(labels_thrust.begin(), labels_thrust.end(), labels);
        thrust::copy(distances_thrust.begin(), distances_thrust.end(), distances);


    }
}

    ''', locals())
        self.module = SourceModule(code, no_extern_c=True, include_dirs=[current_dir], arch='sm_35')
        self.thunk = self.module.get_function('pykmeans')

    def __call__(self, k, data, labels=None, iterations=50):
        if len(data.shape) != 2:
            raise ValueError('data must be a 2d array (rows of vectors)')
        if data.dtype != self.dtype:
            raise ValueError('data is of wrong type specialization (%s, should be %s)' % (
                data.dtype.name, self.dtype.name))
        n = data.shape[0]
        d = data.shape[1]
        centroids = np.empty((k, d), dtype=data.dtype)
        if labels is None:
            labels = np.random.randint(k, size=n, dtype=np.int32)
        labels = labels.astype(np.int32)
        distances = np.empty(n, dtype=data.dtype)
        params = np.array([iterations, n, d, k], dtpye=np.int32)
        self.thunk(params, data, labels, centroids, distances)
        return {'labels':labels, 'centroids':centroids, 'distances':distances}

_kmeans_kernels = {}
def kmeans(k, data, **kwargs):
    dtype = data.dtype
    if dtype.char not in _kmeans_kernels:
        _kmeans_kernels[dtype.char] = KmeansKernel(dtype)
    return _kmeans_kernels[dtype.char](k, data, **kwargs)
