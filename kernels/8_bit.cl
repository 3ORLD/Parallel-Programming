__kernel void calculateHistogram(__global const unsigned short* image,
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
        unsigned short pixelValue = image[gid];
        int bin = (numBins == 256) ? pixelValue : (pixelValue * numBins) / (maxValue + 1);
        if (bin < numBins) {
            atomic_add(&localHist[bin], 1);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < numBins; i += groupSize) {
        if (localHist[i] > 0) {
            atomic_add(&histogram[i], localHist[i]);
        }
    }
}
// Blelloch prefix sum for cumulative histogram (exclusive scan)
__kernel void prefixSum(__global int* input,
                       __global int* output,
                       const int n) {
    // Local memory for scan (assumes n <= 256)
    __local int temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    // Load input into local memory (shift for exclusive scan)
    if (gid < n) {
        temp[lid] = (gid > 0) ? input[gid - 1] : 0;
    } else {
        temp[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Upsweep phase
    for (int d = 1; d < n; d *= 2) {
        int index = (lid + 1) * 2 * d - 1;
        if (index < n && index < 256) {
            temp[index] += temp[index - d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Downsweep phase
    if (lid == 0) {
        temp[n - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = n / 2; d > 0; d /= 2) {
        int index = (lid + 1) * 2 * d - 1;
        if (index < n && index < 256) {
            int t = temp[index];
            temp[index] += temp[index - d];
            temp[index - d] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result to output
    if (gid < n) {
        output[gid] = temp[lid];
    }
}

// Hillis-Steele scan for alternative cumulative histogram (exclusive scan)
__kernel void hillisSteeleScan(__global int* input,
                               __global int* output,
                               const int n) {
    // Local memory for scan (assumes n <= 256)
    __local int temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    // Load input into local memory (shift for exclusive scan)
    if (gid < n) {
        temp[lid] = (gid > 0) ? input[gid - 1] : 0;
    } else {
        temp[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Iterative scan
    for (int offset = 1; offset < n; offset *= 2) {
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

    // Write result to output
    if (gid < n) {
        output[gid] = temp[lid];
    }
}

// Normalizes cumulative histogram to create LUT
__kernel void normalizeLUT(__global int* cumulativeHistogram,
                          __global int* lut,
                          const int totalPixels,
                          const int numBins,
                          const int maxValue) {
    int gid = get_global_id(0);
    if (gid < numBins) {
        // Use integer arithmetic to avoid floating-point issues
        lut[gid] = (totalPixels > 0) ? (cumulativeHistogram[gid] * maxValue) / totalPixels : 0;
        // Clamp to [0, maxValue] for safety
        lut[gid] = min(max(lut[gid], 0), maxValue);
    }
}

__kernel void applyLUT(__global const unsigned short* inputImage,
                      __global const int* lut,
                      __global unsigned short* outputImage,
                      const int totalPixels,
                      const int numBins) {
    int gid = get_global_id(0);
    if (gid < totalPixels) {
        unsigned short pixelValue = inputImage[gid];
        int bin = (numBins == 256) ? pixelValue : (pixelValue * numBins) / 256;
        if (bin < numBins) {
            outputImage[gid] = (unsigned short)lut[bin];
        } else {
            outputImage[gid] = 0; // Fallback
        }
    }
}
