#ifdef CUDA_KERNELS_ON_CPU
#undef CUDA_KERNELS_ON_CPU
#endif

#include "Launch_t.h"

#ifdef LaunchInCU
  void wrapperInCU() {
    printf("in cu wrapper\n");
    wrapper();
  }
#endif
