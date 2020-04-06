#ifndef RecoLocalCalo_EcalRecAlgos_src_KernelHelpers_h
#define RecoLocalCalo_EcalRecAlgos_src_KernelHelpers_h

namespace ecal {
  namespace reconstruction {
  
  __device__ uint32_t hashedIndexEB(uint32_t id);
  
  __device__ uint32_t hashedIndexEE(uint32_t id);
  
  
  __device__ int laser_monitoring_region_EB(uint32_t id);
  
  __device__ int laser_monitoring_region_EE(uint32_t id);
  
  }  // namespace reconstruction
}  // namespace ecal

#endif // RecoLocalCalo_EcalRecAlgos_src_KernelHelpers_h


