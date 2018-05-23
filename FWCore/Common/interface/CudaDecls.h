#ifndef FWCore_Common_interface_CudaDecls_h
#define FWCore_Common_interface_CudaDecls_h

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif

#endif // FWCore_Common_interface_CudaDecls_h
