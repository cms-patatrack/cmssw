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
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include "cudavectors.h"

class ConvertToCartesianVectorsCUDA : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectorsCUDA(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDA() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CartesianVectors = std::vector<math::XYZVectorF>;
  using CylindricalVectors = std::vector<math::RhoEtaPhiVectorF>;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CylindricalVectors> input_;
  edm::EDPutTokenT<CartesianVectors> output_;
};

ConvertToCartesianVectorsCUDA::ConvertToCartesianVectorsCUDA(const edm::ParameterSet& config)
    : input_(consumes<CylindricalVectors>(config.getParameter<edm::InputTag>("input"))) {
  output_ = produces<CartesianVectors>();
}

void ConvertToCartesianVectorsCUDA::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<CartesianVectors>(elements);

  // allocate memory on the GPU for the cylindrical and cartesian vectors
  auto gpu_input = cudautils::make_device_unique<cudavectors::CylindricalVector[]>(elements, cudaStreamDefault);
  auto gpu_product = cudautils::make_device_unique<cudavectors::CartesianVector[]>(elements, cudaStreamDefault);

  // allocate memory on the CPU for the transfer buffer
  auto cpu_input = cudautils::make_host_noncached_unique<cudavectors::CylindricalVector[]>(elements, cudaHostAllocWriteCombined);
  std::memcpy(cpu_input.get(), input.data(), sizeof(cudavectors::CylindricalVector) * elements);

  // copy the input data to the GPU
  cudaCheck(cudaMemcpy(gpu_input.get(), cpu_input.get(), sizeof(cudavectors::CylindricalVector) * elements, cudaMemcpyHostToDevice));

  // convert the vectors from cylindrical to cartesian coordinates, on the GPU
  cudavectors::convertWrapper(gpu_input.get(), gpu_product.get(), elements);

  // copy the result from the GPU
  cudaCheck(cudaMemcpy(product->data(), gpu_product.get(), sizeof(cudavectors::CartesianVector) * elements, cudaMemcpyDeviceToHost));

  event.put(output_, std::move(product));
}

void ConvertToCartesianVectorsCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDA);
