// system include files
#include <cmath>
#include <memory>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "cudavectors.h"

class ConvertToCartesianVectorsCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit ConvertToCartesianVectorsCUDA(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDA() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CartesianVectors = std::vector<math::XYZVectorF>;
  using CylindricalVectors = std::vector<math::RhoEtaPhiVectorF>;

  void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  cms::cuda::host::unique_ptr<cudavectors::CartesianVector[]> output_buffer_;
  edm::EDGetTokenT<CylindricalVectors> input_;
  edm::EDPutTokenT<CartesianVectors> output_;
};

ConvertToCartesianVectorsCUDA::ConvertToCartesianVectorsCUDA(const edm::ParameterSet& config)
    : input_(consumes<CylindricalVectors>(config.getParameter<edm::InputTag>("input"))) {
  output_ = produces<CartesianVectors>();
}

 
void ConvertToCartesianVectorsCUDA::acquire(const edm::Event& event, const edm::EventSetup& setup,
                                                           edm::WaitingTaskWithArenaHolder waitingTaskHolder) {

  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(waitingTaskHolder)};

  auto const& input = event.get(input_);
  auto elements = input.size();

  // allocate memory on the GPU for the cylindrical and cartesian vectors
 
  auto gpu_input = cms::cuda::make_device_unique<cudavectors::CylindricalVector[]>(elements, ctx.stream());
  auto gpu_product = cms::cuda::make_device_unique<cudavectors::CartesianVector[]>(elements, ctx.stream());
  auto cpu_input = cms::cuda::make_host_noncached_unique<cudavectors::CylindricalVector[]>(elements, cudaHostAllocWriteCombined);
  output_buffer_ = cms::cuda::make_host_unique<cudavectors::CartesianVector[]>(elements, ctx.stream());
 
  // copy the input data to the GPU
  
  std::memcpy(cpu_input.get(), input.data(), sizeof(cudavectors::CylindricalVector) * elements);
  cudaCheck(cudaMemcpyAsync(gpu_input.get(), cpu_input.get(), sizeof(cudavectors::CylindricalVector) * elements, cudaMemcpyHostToDevice, ctx.stream()));

   // convert the vectors from cylindrical to cartesian coordinates, on the GPU
  
   cudavectors::convertWrapper(gpu_input.get(), gpu_product.get(), elements, ctx.stream());
  
  // copy the result from the GPU
  
 cudaCheck(cudaMemcpyAsync(output_buffer_.get(), gpu_product.get(), sizeof(cudavectors::CartesianVector) * elements, cudaMemcpyDeviceToHost, ctx.stream()));
  
 // free the GPU memory
   // no need of explicit free operation 
} 
void ConvertToCartesianVectorsCUDA::produce(edm::Event& event, const edm::EventSetup& setup) {
  //no need for a CUDA context here, because there are no CUDA operations
 
   auto const& input = event.get(input_);
   auto elements = input.size();
  
  // instantiate the event product, copy the results from the output buffer, and free it
   auto product = std::make_unique<CartesianVectors>(elements);
   std::memcpy((void*) product->data(), output_buffer_.get(), sizeof(cudavectors::CartesianVector) * elements);
   output_buffer_.reset();
   event.put(output_, std::move(product));
}

void ConvertToCartesianVectorsCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}
// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDA);
