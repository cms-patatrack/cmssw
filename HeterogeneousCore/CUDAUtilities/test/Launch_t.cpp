#include "HeterogeneousCore/CUDAUtilities/interface/requireCUDADevices.h"

#include "Launch_t.h"

#ifdef LaunchInCU
  void wrapperInCU();
#endif


int main() {

  requireCUDADevices();

  printf("in Main\n");
  printEnv();

  wrapper();

#ifdef LaunchInCU
  wrapperInCU();
#endif

  return 0;
}
