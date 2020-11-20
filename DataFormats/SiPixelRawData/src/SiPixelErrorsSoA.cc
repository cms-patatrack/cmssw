#include "DataFormats/SiPixelRawData/interface/SiPixelErrorsSoA.h"

#include <cassert>

SiPixelErrorsSoA::SiPixelErrorsSoA(size_t nErrors, const SiPixelErrorCompact *error, const SiPixelFormatterErrors *err)
    : error_(error, error + nErrors), formatterErrors_(err) {
  assert(error_.size() == nErrors);
}
