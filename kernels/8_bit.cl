__kernel void calculateHistogram(__global const unsigned char* image, 
                                __global int* histogram,
                                const int totalPixels) {
    // Get global ID
    int gid = get_global_id(0);
    // Clear histogram in local memory to avoid race conditions
    // This is a simplified approach - in a full solution we'd use local memory
    if (gid < 256) {
        histogram[gid] = 0;
    }

    // Make sure all work items see the cleared histogram
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Process pixels
    if (gid < totalPixels) {
        unsigned char pixelValue = image[gid];
        atomic_inc(&histogram[pixelValue]);
    }
}

__kernel void prefixSum(__global int* input, 
                       __global int* output,
                       const int n) {
    __local int temp[256];
    int gid = get_global_id(0);
    if (gid < n) {
        temp[gid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 1; stride < n; stride *= 2) {

        int index = (gid + 1) * 2 * stride - 1;

        if (index < n) {
            temp[index] += temp[index - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gid == 0) {
        temp[n - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = n / 2; stride > 0; stride /= 2) {
        int index = (gid + 1) * 2 * stride - 1;
        if (index < n) {
            int t = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } 
    if (gid < n) {
        output[gid] = temp[gid];
    }
}

__kernel void addOriginalValues(__global int* scannedHistogram, 
                              __global int* originalHistogram,
                              const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        scannedHistogram[gid] += originalHistogram[gid];
    }
}

__kernel void normalizeLUT(__global int* cumulativeHistogram, 
                          __global int* lut,
                          const int totalPixels,
                          const int numBins) {
    int gid = get_global_id(0);
    if (gid < numBins) {
        lut[gid] = (int)((float)cumulativeHistogram[gid] / totalPixels * 255);
    }
}

__kernel void applyLUT(__global const unsigned char* inputImage, 
                      __global const int* lut,
                      __global unsigned char* outputImage,
                      const int totalPixels) {
    int gid = get_global_id(0);
    if (gid < totalPixels) {
        unsigned char pixelValue = inputImage[gid];
        outputImage[gid] = (unsigned char)lut[pixelValue];
    }
}
