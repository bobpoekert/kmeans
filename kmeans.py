import _kmeans
import cstruct
import numpy as np

class CudaDeviceProps(cstruct.CStruct):

    __byte_order__ = cstruct.LITTLE_ENDIAN
    __struct__ =  '''
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemSupported;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
'''

def set_device(n):
    assert n <= _kmeans.device_count()
    _kmeans.set_device(n)

def device_count():
    return _kmeans.device_count()

def device_props(device_id):
    res = CudaDeviceProps()
    res.unpack(_kmeans.device_props(device_id))
    return res

def raw_kmeans(points, k, iterations=50, threshold=0.000001):
    assert len(points.shape) == 2
    #label_arr = np.empty(points.shape[0], dtype=np.int32)
    label_arr = np.random.randint(k, size=points.shape[0])
    label_arr = np.require(label_arr, dtype=np.int32, requirements=['C_CONTIGUOUS', 'WRITEABLE', 'ALIGNED'])
    points = np.require(points, dtype=np.float32, requirements=['C_CONTIGUOUS', 'WRITEABLE', 'ALIGNED'])
    res = _kmeans.kmeans(points, label_arr, iterations, k, threshold)
    print label_arr.shape, label_arr.dtype
    #print np.amax(label_arr)
    return res






