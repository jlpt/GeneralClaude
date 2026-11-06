// Minimal CPU stand-in for the CUDA runtime, used only by the headless test
// harness (see ../run_headless.sh). Kernels run sequentially: one "thread" at
// a time, block by block. That is NOT how a GPU orders work, so anything that
// depends on thread ordering (the existing activation race in propagateKernel)
// behaves differently here. The programmed sensor -> motor circuit does not
// depend on ordering, which is what the harness is meant to check.
#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

#define __global__
#define __device__
#define __host__

struct ShimDim3 { unsigned int x = 0, y = 0, z = 0; };
inline ShimDim3 blockIdx, threadIdx, blockDim, gridDim;

enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;

// Every allocation is remembered so the harness can peek at simulator state
// (allocation 0 = activations, 1 = refractory counters, 2 = refractory reset values).
struct ShimAlloc { void* ptr; size_t bytes; };
inline std::vector<ShimAlloc> g_shimAllocs;

inline cudaError_t cudaMalloc(void** p, size_t bytes) {
    *p = std::calloc(1, bytes);
    if (!*p) { std::fprintf(stderr, "shim: out of memory (%zu bytes)\n", bytes); std::exit(1); }
    g_shimAllocs.push_back({ *p, bytes });
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error (CPU shim)"; }

inline void shimOnDeviceToHost(const void* src, size_t bytes);  // defined in raylib.h shim
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    std::memcpy(dst, src, bytes);
    if (kind == cudaMemcpyDeviceToHost) shimOnDeviceToHost(src, bytes);
    return cudaSuccess;
}

inline float atomicAdd(float* addr, float v) { float old = *addr; *addr = old + v; return old; }
inline int atomicAdd(int* addr, int v) { int old = *addr; *addr = old + v; return old; }

// run_headless.sh rewrites `k<<<grid, block>>>(args)` into `SHIM_LAUNCH(k, grid, block)(args)`.
template <class F> struct ShimLauncher {
    F f; unsigned int grid, block;
    template <class... A> void operator()(A... a) const {
        gridDim.x = grid; blockDim.x = block;
        for (unsigned int b = 0; b < grid; ++b) {
            blockIdx.x = b;
            for (unsigned int t = 0; t < block; ++t) { threadIdx.x = t; f(a...); }
        }
    }
};
template <class F> ShimLauncher<F> shimLaunch(F f, long long grid, long long block) {
    return { f, (unsigned int)grid, (unsigned int)block };
}
#define SHIM_LAUNCH(k, g, b) shimLaunch(k, (g), (b))
