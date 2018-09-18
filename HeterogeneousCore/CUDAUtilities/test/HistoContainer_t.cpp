#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<random>
#include<limits>

template<typename T, int NBINS=128, int S=8*sizeof(T), int DELTA=1000>
void go() {
  std::mt19937 eng;

  int rmin=std::numeric_limits<T>::min();
  int rmax=std::numeric_limits<T>::max();
  if (NBINS!=128) {
    rmin=0;
    rmax=NBINS*2-1;
  }



  std::uniform_int_distribution<T> rgen(rmin,rmax);

  
  constexpr int N=12000;
  T v[N];

  using Hist = HistoContainer<T,NBINS,8,S>;
  std::cout << "HistoContainer " << Hist::nbits() << ' ' << Hist::nbins() << ' ' << Hist::binSize() << ' ' << (rmax-rmin)/Hist::nbins() << std::endl;
  std::cout << "bins " << int(Hist::bin(0)) << ' ' <<  int(Hist::bin(rmin)) << ' ' << int(Hist::bin(rmax)) << std::endl;  

  Hist h;
  for (int it=0; it<5; ++it) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);
    if (it==2) for (long long j = N/2; j < N/2+2*Hist::binSize(); j++) v[j]=4;
    h.zero();
    for (long long j = 0; j < N; j++) h.fill(v[j],j);
  
    std::cout << "nspills " << h.nspills << std::endl;    

    auto verify = [&](uint32_t i, uint32_t j, uint32_t k, uint32_t t1, uint32_t t2) {
      assert(t1<N); assert(t2<N);
      if (i!=j && T(v[t1]-v[t2])<=0) std::cout << "for " << i <<':'<< v[k] <<" failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    for (uint32_t i=0; i<Hist::nbins(); ++i) {
      if (0==h.n[i]) continue;
      auto k= *h.begin(i);
      assert(k<N);
      auto kl = NBINS!=128 ? h.bin(std::max(rmin,v[k]-DELTA)) : h.bin(v[k]-T(DELTA));
      auto kh =	NBINS!=128 ? h.bin(std::min(rmax,v[k]+DELTA)) : h.bin(v[k]+T(DELTA));
      if(NBINS==128) { assert(kl!=i);  assert(kh!=i); }
      if(NBINS!=128) { assert(kl<=i);  assert(kh>=i); }
      // std::cout << kl << ' ' << kh << std::endl;
      for (auto j=h.begin(kl); j<h.end(kl); ++j) verify(i,kl, k,k,(*j));
      for (auto	j=h.begin(kh); j<h.end(kh); ++j) verify(i,kh, k,(*j),k);
    }
  }

  for (long long j = 0; j < N; j++) {
    auto b0 = h.bin(v[j]);
    auto stot = h.endSpill()-h.beginSpill();
    int w=0;
    int tot=0;
    auto ftest = [&](int k) {
       assert(k>=0 && k<N);
       tot++;
    };
    forEachInBins(h,v[j],w,ftest);
    int rtot = h.end(b0)-h.begin(b0) + stot;
    assert(tot==rtot);
    w=1; tot=0;
    forEachInBins(h,v[j],w,ftest);
    int bp = b0+1;
    int bm = b0-1;
    if (bp<int(h.nbins())) rtot += h.end(bp)-h.begin(bp);
    if (bm>=0) rtot += h.end(bm)-h.begin(bm);
    assert(tot==rtot);
    w=2; tot=0;
    forEachInBins(h,v[j],w,ftest);
    bp++;
    bm--;
    if (bp<int(h.nbins())) rtot += h.end(bp)-h.begin(bp);
    if (bm>=0) rtot += h.end(bm)-h.begin(bm);
    assert(tot==rtot);
  }

}

int main() {
  go<int16_t>();
  go<uint8_t,128,8,4>();
  go<uint16_t,313/2,9,4>();


  return 0;
}
