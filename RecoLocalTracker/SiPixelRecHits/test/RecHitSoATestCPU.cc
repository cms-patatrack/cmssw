#include "RecHitSoATest.h"
template
void analyzeImpl<cudaCompat::CPUTraits>(TrackingRecHit2DHeterogeneous<cudaCompat::CPUTraits> const & gHits, cudaStream_t stream);

