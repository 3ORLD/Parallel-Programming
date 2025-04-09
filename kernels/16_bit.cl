// kernels/16_bit.cl
__kernel void calculateHistogram16(__global const unsigned short* image,
                                   __global int* histogram,
                                   const int totalPixels,
                                   const int numBins,
                                   const int maxValue) {
    __local int localHist[256]; // Fixed size for simplicity; adjust if numBins > 256
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int groupSize = get_local_size(0);

    // Initialize local histogram
    for (int i = lid; i < min(numBins, 256); i += groupSize) {
        localHist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute local histogram
    if (gid < totalPixels) {
        unsigned short pixelValue = image[gid];
        int bin = (int)(((float)pixelValue * numBins) / (maxValue + 1.0f));
        if (bin < 256) {
            atomic_add(&localHist[bin], 1);
        } else {
            atomic_add(&histogram[bin], 1); // Direct global update for bins >= 256
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce to global histogram
    for (int i = lid; i < min(numBins, 256); i += groupSize) {
        if (localHist[i] > 0) {
            atomic_add(&histogram[i], localHist[i]);
        }
    }
}

__kernel void prefixSum16(__global int* input,
                          __global int* output,
                          const int n) {
    __local int temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    if (gid < n) {
        temp[lid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = 1; d < n && d < 256; d *= 2) {
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

__kernel void hillisSteeleScan16(__global int* input,
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

__kernel void normalizeLUT16(__global int* cumulativeHistogram,
                             __global int* lut,
                             const int totalPixels,
                             const int numBins,
                             const int maxValue) {
    int gid = get_global_id(0);
    if (gid < numBins) {
        lut[gid] = (totalPixels > 0) ? (int)((float)cumulativeHistogram[gid] / totalPixels * maxValue) : 0;
    }
}

__kernel void applyLUT16(__global const unsigned short* inputImage,
                         __global const int* lut,
                         __global unsigned short* outputImage,
                         const int totalPixels,
                         const int numBins) {
    int gid = get_global_id(0);
    if (gid < totalPixels) {
        unsigned short pixelValue = inputImage[gid];
        int bin = (int)(((float)pixelValue * numBins) / 65536.0f);
        outputImage[gid] = (unsigned short)lut[bin];
    }
}
