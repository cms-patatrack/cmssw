#include <chrono>
#include <thread>
#include <future>

#include "HeterogeneousCore/CUDAServices/interface/cuInitAsync.h"

CUresult cuInitAsync(unsigned int flags, double timeout) {
    std::promise<CUresult> handle;
    std::thread worker([&handle, flags]() {
        // call cuInit() in a separate thread
        handle.set_value(cuInit(flags));
    });
    auto result = handle.get_future();
    auto status = result.wait_for(std::chrono::duration<double>(timeout));
    if (status == std::future_status::ready) {
        // wait for the worker thread to complete
        worker.join();
        return result.get();
    } else {
        // let the worker thread sleep (or busy-wait) as long as the CUDA subsystem is busy
        worker.detach();
        return CUDA_ERROR_NOT_INITIALIZED;
    }
}
