#include "HeterogeneousCore/CUDAServices/interface/supportedCudaDevices.h"

int main() {
  return supportedCudaDevices().empty() ? EXIT_FAILURE : EXIT_SUCCESS;
}
