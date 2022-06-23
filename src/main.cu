#include <cuda.h>
#include <iostream>

std::ostream &operator<<(std::ostream &out, cudaDeviceProp const &props) {
    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    out << "Device name: " << props.name << std::endl;
    out << "Multiprocessor count: " << props.multiProcessorCount << std::endl;
    out << "Clock rate (GHz): " << props.clockRate / 1E6 << std::endl;
    out << "Concurrent kernel execution: " << (props.concurrentKernels ? "yes" : "no") << std::endl;
    out << "Compute capability: " << props.major << props.minor << std::endl;
    out << "Max threads dim: [" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", "
        << props.maxThreadsDim[2] << "]\n";
    out << "Max threads/block: " << props.maxThreadsPerBlock << std::endl;
    out << "Max threads per multiprocessor: " << props.maxThreadsPerMultiProcessor << std::endl;
    out << "Warp size (threads): " << props.warpSize << std::endl;
    out << "Global memory (GiB): " << (double)props.totalGlobalMem / (1024 * 1024 * 1024) << std::endl;
    out << "Global memory peak clock rate (GHz): " << props.memoryClockRate / 1E6 << std::endl;
    out << "Global Memory bus width (bytes): " << props.memoryBusWidth / 8 << std::endl;
    // (DDR = factor of 2), (mem clock in kHz = factor of 1000), (bus width in bits = factor of 1/8/1024^3),
    out << "Theoretical memory throughput (GiB/s): "
        << (double)2 * 1000 * props.memoryClockRate * props.memoryBusWidth / 8 / (1024 * 1024 * 1024) << std::endl;
    out << "Shared memory/block (KiB): " << props.sharedMemPerBlock / 1024 << std::endl;
    out << "Async engine count: " << props.asyncEngineCount << std::endl;
    out << "Global L1 cache support: " << (props.globalL1CacheSupported ? "yes" : "no") << std::endl;
    out << "Local L1 cache support: " << (props.localL1CacheSupported ? "yes" : "no") << std::endl;
    out << "L2 cache size (MiB): " << props.l2CacheSize / 1024 << std::endl;

    return out;
}

int main(int argc, char *argv[]) {
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    if (!n_devices) {
        std::cerr << "No cuda devices found\n";
        return 1;
    }

    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, device_id);
    std::cout << device_props;

    return 0;
}
