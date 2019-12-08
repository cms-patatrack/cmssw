#include "RecHitSoATest.h"
template
void analyzeImpl<cudaCompat::GPUTraits>(TrackingRecHit2DHeterogeneous<cudaCompat::GPUTraits> const & gHits, cudaStream_t stream);

