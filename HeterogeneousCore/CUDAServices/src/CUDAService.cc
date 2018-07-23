#include <cuda.h>
#include <cuda/api_wrappers.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAServices/interface/cuInitAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/getCudaDrvErrorString.h"

CUDAService::CUDAService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  bool enabled = iConfig.getUntrackedParameter<bool>("enabled");
  double timeout = iConfig.getUntrackedParameter<double>("timeout");

  if (not enabled) {
    edm::LogInfo("CUDAService") << "CUDAService disabled by configuration";
    return;
  }

  // Call cuInit() in a separate thread, and fail if it does not manage to initialise the CUDA driver before the timeout.
  // This may happen e.g. if the CUDA driver is blocked by a running cuda-gdb.
  // Use the driver API rather than the runtime API because calls to `cudaGetErrorName()` and `cudaGetErrorString()` will
  // also block, while calls to `cuGetErrorName()` and `cuGetErrorString()` will work.
  auto ret = cuInitAsync(0, timeout);
  if (CUDA_ERROR_NOT_INITIALIZED == ret) {
    edm::LogWarning("CUDAService") << "Time out after " << timeout << "s while trying to initialize the CUDA driver API, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
    return;
  }
  if (CUDA_SUCCESS != ret) {
    edm::LogWarning("CUDAService") << "Failed to initialize the CUDA driver API by calling cuInit, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "cuInit succeeded";

  ret = cuDeviceGetCount(&numberOfDevices_);
  if (CUDA_SUCCESS != ret) {
    edm::LogWarning("CUDAService") << "Failed to call cuDeviceGetCount from CUDA driver API, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "cuDeviceGetCount succeeded, found " << numberOfDevices_ << " devices";
  if (numberOfDevices_ < 1) {
    edm::LogWarning("CUDAService") << "Number of devices < 1, disabling CUDAService";
    return;
  }

  auto numberOfStreamsPerDevice = iConfig.getUntrackedParameter<unsigned int>("numberOfStreamsPerDevice");
  if (numberOfStreamsPerDevice > 0) {
    numberOfStreamsTotal_ = numberOfStreamsPerDevice * numberOfDevices_;
    edm::LogSystem("CUDAService") << "Number of edm::Streams per CUDA device has been set to " << numberOfStreamsPerDevice << ". With " << numberOfDevices_ << " CUDA devices, this means total of " << numberOfStreamsTotal_ << " edm::Streams for all CUDA devices."; // TODO: eventually silence to LogDebug
  }

  computeCapabilities_.reserve(numberOfDevices_);
  for(int i=0; i<numberOfDevices_; ++i) {
    int major, minor;
    ret = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, i);
    if (CUDA_SUCCESS != ret) {
      edm::LogWarning("CUDAService") << "Failed to call cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) for device " << i << " from CUDA driver API, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
      return;
    }
    ret = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, i);
    if (CUDA_SUCCESS != ret) {
      edm::LogWarning("CUDAService") << "Failed to call cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) for device " << i << " from CUDA driver API, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
      return;
    }

    edm::LogInfo("CUDAService") << "Device " << i << " compute capability major " << major << " minor " << minor;
    computeCapabilities_.emplace_back(major, minor);
  }

  edm::LogInfo("CUDAService") << "CUDAService fully initialized";
  enabled_ = true;
}

void CUDAService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enabled", true)->setComment("Enable the CUDAService.");
  desc.addUntracked<double>("timeout", 1.)->setComment("Seconds after which the initialisation will time out, and disable the CUDAService.");
  desc.addUntracked<unsigned int>("numberOfStreamsPerDevice", 0)->setComment("Upper limit of the number of edm::Streams that will run on a single CUDA GPU device. The remaining edm::Streams will be run only on other devices (for time being this means CPU in practice). The value '0' means 'unlimited', a value >= 1 imposes the limit.");

  descriptions.add("CUDAService", desc);
}

int CUDAService::deviceWithMostFreeMemory() const {
  size_t freeMem = 0;
  int devId = -1;
  for(int i=0; i<numberOfDevices_; ++i) {
    // TODO: understand why the api-wrappers version gives same value for all devices
    /*
    auto device = cuda::device::get(i);
    auto mem = device.memory.amount_free();
    */
    size_t free, tot;
    cudaSetDevice(i);
    cudaMemGetInfo(&free, &tot);
    auto mem = free;
    edm::LogPrint("CUDAService") << "Device " << i << " free memory " << mem;
    if (mem > freeMem) {
      freeMem = mem;
      devId = i;
    }
  }
  return devId;
}

void CUDAService::setCurrentDevice(int device) const {
  cuda::device::current::set(device);
}

int CUDAService::getCurrentDevice() const {
  return cuda::device::current::get().id();
}
