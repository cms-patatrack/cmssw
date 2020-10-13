#include "CUDADataFormats/TrackingRecHit/interface/SiPixelStatus.h"

#include<cassert>

int main() {

  assert(sizeof(SiPixelStatus)==sizeof(uint8_t));

  SiPixelStatus status;
  status.isOneX = true;
  status.isOneY = false;


  status.isBigX = 1;
  status.isBigY = 0;


  assert(status.isOneX);
  assert(false==(!status.isOneX));
  assert(1==status.isOneX);
  assert(!status.isOneY);
  assert(0==status.isOneY);

  assert(status.isBigX);
  assert(false==(!status.isBigX));
  assert(1==status.isBigX);
  assert(!status.isBigY);
  assert(0==status.isBigY);


  return 0;
}
