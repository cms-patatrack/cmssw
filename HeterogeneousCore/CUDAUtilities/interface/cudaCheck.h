#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

namespace cudautils {

  [[noreturn]] inline void abortOnCudaError(
      const char* file, int line, const char* cmd, const char* error, const char* message) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "cudaCheck(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    throw std::runtime_error(out.str());
  }

}  // namespace cudautils

inline bool cudaCheck_(const char* file, int line, const char* cmd, CUresult result) {
  if (__builtin_expect(result == CUDA_SUCCESS, true))
    return true;

  const char* error;
  const char* message;
  cuGetErrorName(result, &error);
  cuGetErrorString(result, &message);
  cudautils::abortOnCudaError(file, line, cmd, error, message);
  return false;
}

inline bool cudaCheck_(const char* file, int line, const char* cmd, cudaError_t result) {
  if (__builtin_expect(result == cudaSuccess, true))
    return true;

  const char* error = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  cudautils::abortOnCudaError(file, line, cmd, error, message);
  return false;
}

#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

namespace cudautils {

  template <typename T>
  [[noreturn]] inline void abortOnCudaErrorVerbose(
      const char* file, int line, const char* cmd, const char* error, const char* message, T const& description) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "cudaCheck(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    out << description << "\n";
    throw std::runtime_error(out.str());
  }

}  // namespace cudautils

template <typename T>
inline bool cudaCheckVerbose_(const char* file, int line, const char* cmd, CUresult result, T const& description) {
  if (__builtin_expect(result == CUDA_SUCCESS, true))
    return true;

  const char* error;
  const char* message;
  cuGetErrorName(result, &error);
  cuGetErrorString(result, &message);
  cudautils::abortOnCudaErrorVerbose(file, line, cmd, error, message, description);
  return false;
}

template <typename T>
inline bool cudaCheckVerbose_(const char* file, int line, const char* cmd, cudaError_t result, T const& description) {
  if (__builtin_expect(result == cudaSuccess, true))
    return true;

  const char* error = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  cudautils::abortOnCudaErrorVerbose(file, line, cmd, error, message, description);
  return false;
}

#define cudaCheckVerbose(ARG, DESC) (cudaCheckVerbose_(__FILE__, __LINE__, #ARG, (ARG), DESC))

#endif  // HeterogeneousCore_CUDAUtilities_cudaCheck_h
