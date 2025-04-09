// kernels/8_bit.cl
__kernel void calculateHistogram(__global const unsigned char* image,
                                __global int* histogram,
                                const int totalPixels,
                                const int numBins,
                                const int maxValue) {
    __local int localHist[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int groupSize = get_local_size(0);

    for (int i = lid; i < numBins; i += groupSize) {
        localHist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < totalPixels) {
        unsigned char pixelValue = image[gid];
        int bin = (pixelValue * numBins) / (maxValue + 1);
        atomic_add(&localHist[bin], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < numBins; i += groupSize) {
        if (localHist[i] > 0) {
            atomic_add(&histogram[i], localHist[i]);
        }
    }
}

__kernel void prefixSum(__global int* input,
                        __global int* output,
                        const int n) {
    __local int temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    if (gid < n) {
        temp[lid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = 1; d < n; d *= 2) {
        int index = 2 * d * (lid + 1) - 1;
        if (index < n && index < 256) {
            temp[index] += temp[index - d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0 && n <= 256) {
        temp[n - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = n / 2; d > 0 && d < 256; d /= 2) {
        int index = 2 * d * (lid + 1) - 1;
        if (index < n && index < 256) {
            int t = temp[index];
            temp[index] += temp[index - d];
            temp[index - d] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        output[gid] = temp[lid];
    }
}

__kernel void hillisSteeleScan(__global int* input,
                               __global int* output,
                               const int n) {
    __local int temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    if (gid < n) {
        temp[lid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < n && offset < 256; offset *= 2) {
        int val = 0;
        if (lid >= offset) {
            val = temp[lid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (gid < n) {
            temp[lid] += val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        output[gid] = temp[lid];
    }
}

__kernel void normalizeLUT(__global int* cumulativeHistogram,
                           __global int* lut,
                           const int totalPixels,
                           const int numBins,
                           const int maxValue) {
    int gid = get_global_id(0);
    if (gid < numBins) {
        lut[gid] = (totalPixels > 0) ? (int)((float)cumulativeHistogram[gid] / totalPixels * maxValue) : 0;
    }
}

__kernel void applyLUT(__global const unsigned char* inputImage,
                       __global const int* lut,
                       __global unsigned char* outputImage,
                       const int totalPixels,
                       const int numBins) {
    int gid = get_global_id(0);
    if (gid < totalPixels) {
        unsigned char pixelValue = inputImage[gid];
        int bin = (pixelValue * numBins) / 256;
        outputImage[gid] = (unsigned char)lut[bin];
    }
}
