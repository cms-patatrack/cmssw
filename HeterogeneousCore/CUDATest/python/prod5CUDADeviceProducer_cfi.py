import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDACore.cudaDeviceChooserProducer_cfi import cudaDeviceChooserProducer as _cudaDeviceChooserProducer
prod5CUDADeviceProducer = _cudaDeviceChooserProducer.clone()
