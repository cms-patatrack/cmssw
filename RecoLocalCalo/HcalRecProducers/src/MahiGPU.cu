#include <Eigen/Dense>

#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"
#include "DataFormats/Math/interface/EigenComputations.h"

// nvcc not able to parse this guy (whatever is inlcuded from it)....
//#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"

#include "MahiGPU.h"

#ifdef HCAL_MAHI_GPUDEBUG
#define DETID_TO_DEBUG 1125647428
#endif

namespace hcal {
  namespace mahi {

    // Assume: same number of samples for HB and HE
    // TODO: add/validate restrict (will increase #registers in use by the kernel)
    __global__ void kernel_prep1d_sameNumberOfSamples(float* amplitudes,
                                                      float* noiseTerms,
                                                      float* outputEnergy,
                                                      float* outputChi2,
                                                      uint16_t const* dataf01HE,
                                                      uint16_t const* dataf5HB,
                                                      uint16_t const* dataf3HB,
                                                      uint32_t const* idsf01HE,
                                                      uint32_t const* idsf5HB,
                                                      uint32_t const* idsf3HB,
                                                      uint32_t const stridef01HE,
                                                      uint32_t const stridef5HB,
                                                      uint32_t const stridef3HB,
                                                      uint32_t const nchannelsf01HE,
                                                      uint32_t const nchannelsf5HB,
                                                      uint8_t const* npresamplesf5HB,
                                                      int8_t* soiSamples,
                                                      float* method0Energy,
                                                      float* method0Time,
                                                      uint32_t* outputdid,
                                                      uint32_t const nchannels,
                                                      uint32_t const* recoParam1Values,
                                                      uint32_t const* recoParam2Values,
                                                      float const* qieCoderOffsets,
                                                      float const* qieCoderSlopes,
                                                      int const* qieTypes,
                                                      float const* pedestalWidths,
                                                      float const* effectivePedestalWidths,
                                                      float const* pedestals,
                                                      float const* effectivePedestals,
                                                      bool const useEffectivePedestals,
                                                      int const* sipmTypeValues,
                                                      float const* fcByPEValues,
                                                      float const* parLin1Values,
                                                      float const* parLin2Values,
                                                      float const* parLin3Values,
                                                      float const* gainValues,
                                                      float const* respCorrectionValues,
                                                      int const maxDepthHB,
                                                      int const maxDepthHE,
                                                      int const maxPhiHE,
                                                      int const firstHBRing,
                                                      int const lastHBRing,
                                                      int const firstHERing,
                                                      int const lastHERing,
                                                      int const nEtaHB,
                                                      int const nEtaHE,
                                                      int const sipmQTSShift,
                                                      int const sipmQNTStoSum,
                                                      int const firstSampleShift,
                                                      uint32_t const offsetForHashes,
                                                      float const ts4Thresh) {
      // indices + runtime constants
      auto const sample = threadIdx.x;
      int32_t const nsamplesExpected = blockDim.x;
      auto const lch = threadIdx.y;
      auto const gch = lch + blockDim.y * blockIdx.x;
      auto const nchannels_per_block = blockDim.y;
      auto const linearThPerBlock = threadIdx.x + threadIdx.y * blockDim.x;

      // remove
      if (gch >= nchannels)
        return;

      // initialize all output buffers
      if (sample == 0) {
        outputdid[gch] = 0;
        method0Energy[gch] = 0;
        method0Time[gch] = 0;
        outputEnergy[gch] = 0;
        outputChi2[gch] = 0;
      }

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_SINGLECHANNEL
      if (gch > 0)
        return;
#endif
#endif

      // configure shared mem
      extern __shared__ char smem[];
      float* shrEnergyM0PerTS = reinterpret_cast<float*>(smem);
      float* shrChargeMinusPedestal = shrEnergyM0PerTS + nsamplesExpected * nchannels_per_block;
      float* shrMethod0EnergyAccum = shrChargeMinusPedestal + nsamplesExpected * nchannels_per_block;
      float* shrEnergyM0TotalAccum = shrMethod0EnergyAccum + nchannels_per_block;
      unsigned long long int* shrMethod0EnergySamplePair =
          reinterpret_cast<unsigned long long int*>(shrEnergyM0TotalAccum + nchannels_per_block);
      if (sample == 0) {
        shrMethod0EnergyAccum[lch] = 0;
        shrMethod0EnergySamplePair[lch] = __float_as_uint(std::numeric_limits<float>::min());
        shrEnergyM0TotalAccum[lch] = 0;
      }

      // offset output
      auto* amplitudesForChannel = amplitudes + nsamplesExpected * gch;
      auto* noiseTermsForChannel = noiseTerms + nsamplesExpected * gch;
      auto const nchannelsf015 = nchannelsf01HE + nchannelsf5HB;

      // get event input quantities
      auto const stride = gch < nchannelsf01HE ? stridef01HE : (gch < nchannelsf015 ? stridef5HB : stridef3HB);
      auto const nsamples = gch < nchannelsf01HE ? compute_nsamples<Flavor01>(stride)
                                                 : (gch < nchannelsf015 ? compute_nsamples<Flavor5>(stride)
                                                                        : compute_nsamples<Flavor3>(stride));

#ifdef HCAL_MAHI_GPUDEBUG
      assert(nsamples == nsamplesExpected);
#endif

      auto const id = gch < nchannelsf01HE
                          ? idsf01HE[gch]
                          : (gch < nchannelsf015 ? idsf5HB[gch - nchannelsf01HE] : idsf3HB[gch - nchannelsf015]);
      auto const did = HcalDetId{id};
      auto const adc =
          gch < nchannelsf01HE
              ? adc_for_sample<Flavor01>(dataf01HE + stride * gch, sample)
              : (gch < nchannelsf015 ? adc_for_sample<Flavor5>(dataf5HB + stride * (gch - nchannelsf01HE), sample)
                                     : adc_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample));
      auto const capid =
          gch < nchannelsf01HE
              ? capid_for_sample<Flavor01>(dataf01HE + stride * gch, sample)
              : (gch < nchannelsf015 ? capid_for_sample<Flavor5>(dataf5HB + stride * (gch - nchannelsf01HE), sample)
                                     : capid_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample));

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
      if (id != DETID_TO_DEBUG)
        return;
#endif
#endif

      // compute hash for this did
      auto const hashedId =
          did.subdetId() == HcalBarrel
              ? hcal::reconstruction::did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
              : hcal::reconstruction::did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE) +
                    offsetForHashes;

      // conditions based on the hash
      // FIXME: remove hardcoded values
      auto const qieType = qieTypes[hashedId] > 0 ? 1 : 0;  // 2 types at this point
      auto const* qieOffsets = qieCoderOffsets + hashedId * HcalQIECodersGPU::numValuesPerChannel;
      auto const* qieSlopes = qieCoderSlopes + hashedId * HcalQIECodersGPU::numValuesPerChannel;
      auto const* pedestalsForChannel = pedestals + hashedId * 4;
      auto const* pedestalWidthsForChannel = useEffectivePedestals && (gch < nchannelsf01HE || gch >= nchannelsf015)
                                                 ? effectivePedestalWidths + hashedId * 4
                                                 : pedestalWidths + hashedId * 4;
      auto const* gains = gainValues + hashedId * 4;
      auto const gain = gains[capid];
      auto const gain0 = gains[0];
      auto const respCorrection = respCorrectionValues[hashedId];
      auto const pedestal = pedestalsForChannel[capid];
      auto const pedestalWidth = pedestalWidthsForChannel[capid];
      // if needed, only use effective pedestals for f01
      auto const pedestalToUseForMethod0 = useEffectivePedestals && (gch < nchannelsf01HE || gch >= nchannelsf015)
                                               ? effectivePedestals[hashedId * 4 + capid]
                                               : pedestal;
      auto const sipmType = sipmTypeValues[hashedId];
      auto const fcByPE = fcByPEValues[hashedId];
      auto const recoParam1 = recoParam1Values[hashedId];
      auto const recoParam2 = recoParam2Values[hashedId];

#ifdef HCAL_MAHI_GPUDEBUG
      printf("qieType = %d qieOffset0 = %f qieOffset1 = %f qieSlope0 = %f qieSlope1 = %f\n",
             qieType,
             qieOffsets[0],
             qieOffsets[1],
             qieSlopes[0],
             qieSlopes[1]);
#endif

      // compute charge
      auto const charge = hcal::reconstruction::compute_coder_charge(qieType, adc, capid, qieOffsets, qieSlopes);

      shrChargeMinusPedestal[linearThPerBlock] = charge - pedestal;
      if (gch < nchannelsf01HE) {
        // NOTE: assume that soi is high only for a single guy!
        //   which must be the case. cpu version does not check for that
        //   if that is not the case, we will see that with cuda mmecheck
        auto const soibit = soibit_for_sample<Flavor01>(dataf01HE + stride * gch, sample);
        if (soibit == 1)
          soiSamples[gch] = sample;
      } else if (gch >= nchannelsf015) {
        auto const soibit = soibit_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample);
        if (soibit == 1)
          soiSamples[gch] = sample;
      }
      __syncthreads();
      int32_t const soi = gch < nchannelsf01HE
                              ? soiSamples[gch]
                              : (gch < nchannelsf015 ? npresamplesf5HB[gch - nchannelsf01HE] : soiSamples[gch]);
      //int32_t const soi = gch >= nchannelsf01HE
      //    ? npresamplesf5HB[gch - nchannelsf01HE]
      //    : soiSamples[gch];
      // this is here just to make things uniform...
      if (gch >= nchannelsf01HE && gch < nchannelsf015 && sample == 0)
        soiSamples[gch] = npresamplesf5HB[gch - nchannelsf01HE];

      //
      // compute various quantities (raw charge and tdc stuff)
      // NOTE: this branch will be divergent only for a single warp that
      // sits on the boundary when flavor 01 channels end and flavor 5 start
      //
      float rawCharge;
#ifdef COMPUTE_TDC_TIME
      float tdcTime;
#endif  // COMPUTE_TDC_TIME
      auto const dfc = hcal::reconstruction::compute_diff_charge_gain(
          qieType, adc, capid, qieOffsets, qieSlopes, gch < nchannelsf01HE || gch >= nchannelsf015);
      if (gch >= nchannelsf01HE && gch < nchannelsf015) {
        // flavor 5
        rawCharge = charge;
#ifdef COMPUTE_TDC_TIME
        tdcTime = HcalSpecialTimes::UNKNOWN_T_NOTDC;
#endif  // COMPUTE_TDC_TIME
      } else {
        // flavor 0 or 1 or 3
        // conditions needed for sipms
        auto const parLin1 = parLin1Values[sipmType - 1];
        auto const parLin2 = parLin2Values[sipmType - 1];
        auto const parLin3 = parLin3Values[sipmType - 1];

        int const first = std::max(soi + sipmQTSShift, 0);
        int const last = std::min(soi + sipmQNTStoSum, nsamplesExpected);
        float sipmq = 0.0f;
        for (auto ts = first; ts < last; ts++)
          sipmq += shrChargeMinusPedestal[threadIdx.y * nsamplesExpected + ts];
        auto const effectivePixelsFired = sipmq / fcByPE;
        auto const factor =
            hcal::reconstruction::compute_reco_correction_factor(parLin1, parLin2, parLin3, effectivePixelsFired);
        rawCharge = (charge - pedestal) * factor + pedestal;
#ifdef COMPUTE_TDC_TIME
        if (gch < nchannelsf01HE)
          tdcTime = HcalSpecialTimes::getTDCTime(tdc_for_sample<Flavor01>(dataf01HE + stride * gch, sample));
        else if (gch >= nchannelsf015)
          tdcTime =
              HcalSpecialTimes::getTDCTime(tdc_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample));
#endif  // COMPUTE_TDC_TIME

#ifdef HCAL_MAHI_GPUDEBUG
        printf("first = %d last = %d sipmQ = %f factor = %f rawCharge = %f\n", first, last, sipmq, factor, rawCharge);
#endif
      }

      // compute method 0 quantities
      // TODO: need to apply containment
      // TODO: need to apply time slew
      // TODO: for < run 3, apply HBM legacy energy correction
      auto const nsamplesToAdd = recoParam1 < 10 ? recoParam2 : (recoParam1 >> 14) & 0xF;
      auto const startSampleTmp = soi + firstSampleShift;
      auto const startSample = startSampleTmp < 0 ? 0 : startSampleTmp;
      auto const endSample = startSample + nsamplesToAdd < nsamples ? startSample + nsamplesToAdd : nsamples;
      // NOTE: gain is a small number < 10^-3, multiply it last
      auto const energym0_per_ts = gain * ((rawCharge - pedestalToUseForMethod0) * respCorrection);
      auto const energym0_per_ts_gain0 = gain0 * ((rawCharge - pedestalToUseForMethod0) * respCorrection);
      // store to shared mem
      shrEnergyM0PerTS[lch * nsamplesExpected + sample] = energym0_per_ts;
      atomicAdd(&shrEnergyM0TotalAccum[lch], energym0_per_ts_gain0);

#ifdef HCAL_MAHI_GPUDEBUG
      printf(
          "id = %u sample = %d gch = %d hashedId = %u adc = %u capid = %u\n"
          "   charge = %f rawCharge = %f dfc = %f pedestal = %f\n"
          "   gain = %f respCorrection = %f energym0_per_ts = %f\n",
          id,
          sample,
          gch,
          hashedId,
          adc,
          capid,
          charge,
          rawCharge,
          dfc,
          pedestalToUseForMethod0,
          gain,
          respCorrection,
          energym0_per_ts);
      printf(
          "startSample = %d endSample = %d param1 = %u param2 = %u\n", startSample, endSample, recoParam1, recoParam2);
#endif

      if (sample >= startSample && sample < endSample) {
        atomicAdd(&shrMethod0EnergyAccum[lch], energym0_per_ts);
        // pack sample, energy as 64 bit value
        unsigned long long int old = shrMethod0EnergySamplePair[lch], assumed;
        unsigned long long int val =
            (static_cast<unsigned long long int>(sample) << 32) + __float_as_uint(energym0_per_ts);
        do {
          assumed = old;
          // decode energy, sample values
          //int const current_sample = (assumed >> 32) & 0xffffffff;
          float const current_energy = __uint_as_float(assumed & 0xffffffff);
          if (energym0_per_ts > current_energy)
            old = atomicCAS(&shrMethod0EnergySamplePair[lch], assumed, val);
          else
            break;
        } while (assumed != old);
      }
      __syncthreads();

      // NOTE: must take soi, as values for that thread are used...
      if (sample == soi) {
        auto const method0_energy = shrMethod0EnergyAccum[lch];
        auto const val = shrMethod0EnergySamplePair[lch];
        int const max_sample = (val >> 32) & 0xffffffff;
        float const max_energy = __uint_as_float(val & 0xffffffff);
        float const max_energy_1 =
            max_sample < nsamples - 1 ? shrEnergyM0PerTS[lch * nsamplesExpected + max_sample + 1] : 0.f;
        float const position = nsamplesToAdd < nsamples ? max_sample - soi : max_sample;
        auto const sum = max_energy + max_energy_1;
        // FIXME: for full comparison with cpu method 0  timing,
        // need to correct by slew
        // requires an accumulator -> more shared mem -> omit here unless
        // really needed
        float const time =
            max_energy > 0.f && max_energy_1 > 0.f ? 25.f * (position + max_energy_1 / sum) : 25.f * position;

        // store method0 quantities to global mem
        outputdid[gch] = id;
        method0Energy[gch] = method0_energy;
        method0Time[gch] = time;

#ifdef HCAL_MAHI_GPUDEBUG
        printf("tsTOT = %f tstrig = %f ts4Thresh = %f\n", shrEnergyM0TotalAccum[lch], energym0_per_ts_gain0, ts4Thresh);
#endif

        // check as in cpu version if mahi is not needed
        // FIXME: KNOWN ISSUE: observed a problem when rawCharge and pedestal
        // are basically equal and generate -0.00000...
        // needs to be treated properly
        if (!(shrEnergyM0TotalAccum[lch] > 0 && energym0_per_ts_gain0 >= ts4Thresh)) {
          // do not need to run mahi minimization
          //outputEnergy[gch] = 0; energy already inited to 0
          outputChi2[gch] = -9999.f;
        }

#ifdef HCAL_MAHI_GPUDEBUG
        printf("method0_energy = %f max_sample = %d max_energy = %f time = %f\n",
               method0_energy,
               max_sample,
               max_energy,
               time);
#endif
      }

      //
      // preparations for mahi fit
      //
      auto const amplitude = rawCharge - pedestalToUseForMethod0;
      auto const noiseADC = (1. / std::sqrt(12)) * dfc;
      auto const noisePhoto = amplitude > pedestalWidth ? std::sqrt(amplitude * fcByPE) : 0.f;
      auto const noiseTerm = noiseADC * noiseADC + noisePhoto * noisePhoto + pedestalWidth * pedestalWidth;

#ifdef HCAL_MAHI_GPUDEBUG
      printf(
          "charrge(%d) = %f pedestal(%d) = %f dfc(%d) = %f pedestalWidth(%d) = %f noiseADC(%d) = %f noisPhoto(%d) = "
          "%f\n",
          sample,
          rawCharge,
          sample,
          pedestalToUseForMethod0,
          sample,
          dfc,
          sample,
          pedestalWidth,
          sample,
          noiseADC,
          sample,
          noisePhoto);
#endif

      // store to global memory
      amplitudesForChannel[sample] = amplitude;
      noiseTermsForChannel[sample] = noiseTerm;
    }

    // TODO: need to add an array of offsets for pulses (a la activeBXs...)
    // Assume for now 8 pulses
    __global__ void kernel_prep_pulseMatrices_sameNumberOfSamples(float* pulseMatrices,
                                                                  float* pulseMatricesM,
                                                                  float* pulseMatricesP,
                                                                  int const* pulseOffsets,
                                                                  float const* amplitudes,
                                                                  uint32_t const* idsf01HE,
                                                                  uint32_t const* idsf5HB,
                                                                  uint32_t const* idsf3HB,
                                                                  uint32_t const nchannelsf01HE,
                                                                  uint32_t const nchannelsf5HB,
                                                                  uint32_t const nchannelsTotal,
                                                                  int8_t const* soiSamples,
                                                                  uint32_t const* recoPulseShapeIds,
                                                                  float const* acc25nsVecValues,
                                                                  float const* diff25nsItvlVecValues,
                                                                  float const* accVarLenIdxMinusOneVecValues,
                                                                  float const* diffVarItvlIdxMinusOneVecValues,
                                                                  float const* accVarLenIdxZeroVecValues,
                                                                  float const* diffVarItvlIdxZeroVecValues,
                                                                  float const meanTime,
                                                                  float const timeSigmaSiPM,
                                                                  float const timeSigmaHPD,
                                                                  int const maxDepthHB,
                                                                  int const maxDepthHE,
                                                                  int const maxPhiHE,
                                                                  int const firstHBRing,
                                                                  int const lastHBRing,
                                                                  int const firstHERing,
                                                                  int const lastHERing,
                                                                  int const nEtaHB,
                                                                  int const nEtaHE,
                                                                  uint32_t const offsetForHashes,
                                                                  bool const applyTimeSlew,
                                                                  float const tzeroTimeSlew,
                                                                  float const slopeTimeSlew,
                                                                  float const tmaxTimeSlew) {
      // indices
      auto const ipulse = threadIdx.y;
      auto const npulses = blockDim.y;
      auto const sample = threadIdx.x;
      auto const nsamples = blockDim.x;
      auto const lch = threadIdx.z;
      auto const gch = lch + blockIdx.x * blockDim.z;
      auto const nchannelsf015 = nchannelsf01HE + nchannelsf5HB;

      if (gch >= nchannelsTotal)
        return;

      // conditions
      auto const id = gch < nchannelsf01HE
                          ? idsf01HE[gch]
                          : (gch < nchannelsf015 ? idsf5HB[gch - nchannelsf01HE] : idsf3HB[gch - nchannelsf015]);
      //auto const id = gch >= nchannelsf01HE
      //    ? idsf5HB[gch - nchannelsf01HE]
      //    : idsf01HE[gch];
      auto const deltaT = gch >= nchannelsf01HE && gch < nchannelsf015 ? timeSigmaHPD : timeSigmaSiPM;
      auto const did = DetId{id};
      auto const hashedId =
          did.subdetId() == HcalBarrel
              ? hcal::reconstruction::did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
              : hcal::reconstruction::did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE) +
                    offsetForHashes;
      auto const recoPulseShapeId = recoPulseShapeIds[hashedId];
      auto const* acc25nsVec = acc25nsVecValues + recoPulseShapeId * hcal::reconstruction::maxPSshapeBin;
      auto const* diff25nsItvlVec = diff25nsItvlVecValues + recoPulseShapeId * hcal::reconstruction::maxPSshapeBin;
      auto const* accVarLenIdxMinusOneVec =
          accVarLenIdxMinusOneVecValues + recoPulseShapeId * hcal::reconstruction::nsPerBX;
      auto const* diffVarItvlIdxMinusOneVec =
          diffVarItvlIdxMinusOneVecValues + recoPulseShapeId * hcal::reconstruction::nsPerBX;
      auto const* accVarLenIdxZeroVec = accVarLenIdxZeroVecValues + recoPulseShapeId * hcal::reconstruction::nsPerBX;
      auto const* diffVarItvlIdxZeroVec =
          diffVarItvlIdxZeroVecValues + recoPulseShapeId * hcal::reconstruction::nsPerBX;

      // offset output arrays
      auto* pulseMatrix = pulseMatrices + nsamples * npulses * gch;
      auto* pulseMatrixM = pulseMatricesM + nsamples * npulses * gch;
      auto* pulseMatrixP = pulseMatricesP + nsamples * npulses * gch;

      // amplitude per ipulse
      int const soi = soiSamples[gch];
      int const pulseOffset = pulseOffsets[ipulse];
      auto const amplitude = amplitudes[gch * nsamples + pulseOffset + soi];

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
      if (id != DETID_TO_DEBUG)
        return;
#endif
#endif

#ifdef HCAL_MAHI_GPUDEBUG
      if (sample == 0 && ipulse == 0) {
        for (int i = 0; i < 8; i++)
          printf("amplitude(%d) = %f\n", i, amplitudes[gch * nsamples + i]);
        printf("acc25nsVec and diff25nsItvlVec for recoPulseShapeId = %u\n", recoPulseShapeId);
        for (int i = 0; i < 256; i++) {
          printf("acc25nsVec(%d) = %f diff25nsItvlVec(%d) = %f\n", i, acc25nsVec[i], i, diff25nsItvlVec[i]);
        }
        printf("accVarLenIdxZEROVec and accVarLenIdxMinusOneVec\n");
        for (int i = 0; i < 25; i++) {
          printf("accVarLenIdxZEROVec(%d) = %f accVarLenIdxMinusOneVec(%d) = %f\n",
                 i,
                 accVarLenIdxZeroVec[i],
                 i,
                 accVarLenIdxMinusOneVec[i]);
        }
        printf("diffVarItvlIdxZEROVec and diffVarItvlIdxMinusOneVec\n");
        for (int i = 0; i < 25; i++) {
          printf("diffVarItvlIdxZEROVec(%d) = %f diffVarItvlIdxMinusOneVec(%d) = %f\n",
                 i,
                 diffVarItvlIdxZeroVec[i],
                 i,
                 diffVarItvlIdxMinusOneVec[i]);
        }
      }
#endif

      auto t0 = meanTime;
      if (applyTimeSlew) {
        if (amplitude <= 1.0f)
          t0 += hcal::reconstruction::compute_time_slew_delay(1.0, tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew);
        else
          t0 += hcal::reconstruction::compute_time_slew_delay(amplitude, tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew);
      }
      auto const t0m = -deltaT + t0;
      auto const t0p = deltaT + t0;

#ifdef HCAL_MAHI_GPUDEBUG
      if (sample == 0 && ipulse == 0) {
        printf("time values: %f %f %f\n", t0, t0m, t0p);
      }

      if (sample == 0 && ipulse == 0) {
        for (int i = 0; i < 10; i++) {
          auto const value = hcal::reconstruction::compute_pulse_shape_value(t0,
                                                                             i,
                                                                             0,
                                                                             acc25nsVec,
                                                                             diff25nsItvlVec,
                                                                             accVarLenIdxMinusOneVec,
                                                                             diffVarItvlIdxMinusOneVec,
                                                                             accVarLenIdxZeroVec,
                                                                             diffVarItvlIdxZeroVec);
          printf("pulse(%d) = %f\n", i, value);
        }
        printf("\n");
        for (int i = 0; i < 10; i++) {
          auto const value = hcal::reconstruction::compute_pulse_shape_value(t0p,
                                                                             i,
                                                                             0,
                                                                             acc25nsVec,
                                                                             diff25nsItvlVec,
                                                                             accVarLenIdxMinusOneVec,
                                                                             diffVarItvlIdxMinusOneVec,
                                                                             accVarLenIdxZeroVec,
                                                                             diffVarItvlIdxZeroVec);
          printf("pulseP(%d) = %f\n", i, value);
        }
        printf("\n");
        for (int i = 0; i < 10; i++) {
          auto const value = hcal::reconstruction::compute_pulse_shape_value(t0m,
                                                                             i,
                                                                             0,
                                                                             acc25nsVec,
                                                                             diff25nsItvlVec,
                                                                             accVarLenIdxMinusOneVec,
                                                                             diffVarItvlIdxMinusOneVec,
                                                                             accVarLenIdxZeroVec,
                                                                             diffVarItvlIdxZeroVec);
          printf("pulseM(%d) = %f\n", i, value);
        }
      }
#endif

      // FIXME: shift should be treated properly,
      // here assume 8 time slices and 8 samples
      auto const shift = 4 - soi;  // as in cpu version!

      // auto const offset = ipulse - soi;
      // auto const idx = sample - offset;
      int32_t const idx = sample - pulseOffset;
      auto const value = idx >= 0 && idx < nsamples
                             ? hcal::reconstruction::compute_pulse_shape_value(t0,
                                                                               idx,
                                                                               shift,
                                                                               acc25nsVec,
                                                                               diff25nsItvlVec,
                                                                               accVarLenIdxMinusOneVec,
                                                                               diffVarItvlIdxMinusOneVec,
                                                                               accVarLenIdxZeroVec,
                                                                               diffVarItvlIdxZeroVec)
                             : 0;
      auto const value_t0m = idx >= 0 && idx < nsamples
                                 ? hcal::reconstruction::compute_pulse_shape_value(t0m,
                                                                                   idx,
                                                                                   shift,
                                                                                   acc25nsVec,
                                                                                   diff25nsItvlVec,
                                                                                   accVarLenIdxMinusOneVec,
                                                                                   diffVarItvlIdxMinusOneVec,
                                                                                   accVarLenIdxZeroVec,
                                                                                   diffVarItvlIdxZeroVec)
                                 : 0;
      auto const value_t0p = idx >= 0 && idx < nsamples
                                 ? hcal::reconstruction::compute_pulse_shape_value(t0p,
                                                                                   idx,
                                                                                   shift,
                                                                                   acc25nsVec,
                                                                                   diff25nsItvlVec,
                                                                                   accVarLenIdxMinusOneVec,
                                                                                   diffVarItvlIdxMinusOneVec,
                                                                                   accVarLenIdxZeroVec,
                                                                                   diffVarItvlIdxZeroVec)
                                 : 0;

      // store to global
      pulseMatrix[ipulse * nsamples + sample] = value;
      ;
      pulseMatrixM[ipulse * nsamples + sample] = value_t0m;
      pulseMatrixP[ipulse * nsamples + sample] = value_t0p;
    }

    // TODO: add active bxs
    template <typename MatrixType, typename VectorType>
    __device__ void fnnls(MatrixType const& AtA,
                          VectorType const& Atb,
                          VectorType& solution,
                          int& npassive,
                          calo::multifit::ColumnVector<VectorType::RowsAtCompileTime, int>& pulseOffsets,
                          calo::multifit::MapSymM<float, VectorType::RowsAtCompileTime>& matrixL,
                          double const eps,
                          int const maxIterations) {
      // constants
      constexpr auto NPULSES = VectorType::RowsAtCompileTime;

      // to keep track of where to terminate if converged
      Eigen::Index w_max_idx_prev = 0;
      float w_max_prev = 0;
      auto eps_to_use = eps;
      bool recompute = false;

      // used throughout
      VectorType s;
      float reg_b[NPULSES];
      //float matrixLStorage[MapSymM<float, NPULSES>::total];
      //MapSymM<float, NPULSES> matrixL{matrixLStorage};

      int iter = 0;
      while (true) {
        if (iter > 0 || npassive == 0) {
          auto const nactive = NPULSES - npassive;
          // exit if there are no more pulses to constrain
          if (nactive == 0)
            break;

          // compute the gradient
          //w.tail(nactive) = Atb.tail(nactive) - (AtA * solution).tail(nactive);
          Eigen::Index w_max_idx;
          float w_max = -std::numeric_limits<float>::max();
          for (int icol = npassive; icol < NPULSES; icol++) {
            auto const icol_real = pulseOffsets(icol);
            auto const atb = Atb(icol_real);
            float sum = 0;
#pragma unroll
            for (int counter = 0; counter < NPULSES; counter++)
              sum += counter > icol_real ? AtA(counter, icol_real) * solution(counter)
                                         : AtA(icol_real, counter) * solution(counter);

            auto const w = atb - sum;
            if (w > w_max) {
              w_max = w;
              w_max_idx = icol - npassive;
            }
          }

          // check for convergence
          if (w_max < eps_to_use || w_max_idx == w_max_idx_prev && w_max == w_max_prev)
            break;

          if (iter >= maxIterations)
            break;

          w_max_prev = w_max;
          w_max_idx_prev = w_max_idx;

          // move index to the right part of the vector
          w_max_idx += npassive;

          Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(w_max_idx));
          ++npassive;
        }

        // inner loop
        while (true) {
          if (npassive == 0)
            break;

          //s.head(npassive)
          //auto const& matrixL =
          //    AtA.topLeftCorner(npassive, npassive)
          //        .llt().matrixL();
          //.solve(Atb.head(npassive));
          if (recompute || iter == 0)
            calo::multifit::compute_decomposition_forwardsubst_with_offsets(
                matrixL, AtA, reg_b, Atb, npassive, pulseOffsets);
          else
            calo::multifit::update_decomposition_forwardsubst_with_offsets(
                matrixL, AtA, reg_b, Atb, npassive, pulseOffsets);

          // run backward substituion
          s(npassive - 1) = reg_b[npassive - 1] / matrixL(npassive - 1, npassive - 1);
          for (int i = npassive - 2; i >= 0; --i) {
            float total = 0;
            for (int j = i + 1; j < npassive; j++)
              total += matrixL(j, i) * s(j);

            s(i) = (reg_b[i] - total) / matrixL(i, i);
          }

          // done if solution values are all positive
          if (s.head(npassive).minCoeff() > 0.f) {
            for (int i = 0; i < npassive; i++) {
              auto const i_real = pulseOffsets(i);
              solution(i_real) = s(i);
            }
            //solution.head(npassive) = s.head(npassive);
            recompute = false;
            break;
          }

          // there were negative values -> have to recompute the whole decomp
          recompute = true;

          auto alpha = std::numeric_limits<float>::max();
          Eigen::Index alpha_idx = 0, alpha_idx_real = 0;
          for (int i = 0; i < npassive; i++) {
            if (s[i] <= 0.) {
              auto const i_real = pulseOffsets(i);
              auto const ratio = solution[i_real] / (solution[i_real] - s[i]);
              if (ratio < alpha) {
                alpha = ratio;
                alpha_idx = i;
                alpha_idx_real = i_real;
              }
            }
          }

          // upadte solution
          for (int i = 0; i < npassive; i++) {
            auto const i_real = pulseOffsets(i);
            solution(i_real) += alpha * (s(i) - solution(i_real));
          }
          //solution.head(npassive) += alpha *
          //    (s.head(npassive) - solution.head(npassive));
          solution[alpha_idx_real] = 0;
          --npassive;

          Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(alpha_idx));
        }

        // as in cpu
        ++iter;
        if (iter % 10 == 0)
          eps_to_use *= 10;
      }
    }

    template <int NSAMPLES, int NPULSES>
    __forceinline__ __device__ void update_covariance(
        calo::multifit::ColumnVector<NPULSES> const& resultAmplitudesVector,
        calo::multifit::MapSymM<float, NSAMPLES>& covarianceMatrix,
        Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrix,
        Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrixM,
        Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrixP) {
#pragma unroll
      for (int ipulse = 0; ipulse < NPULSES; ipulse++) {
        auto const resultAmplitude = resultAmplitudesVector(ipulse);
        if (resultAmplitude == 0)
          continue;

#ifdef HCAL_MAHI_GPUDEBUG
        printf("pulse cov array for ibx = %d and offset %d\n", ipulse, offset);
#endif

        // preload a column
        float pmcol[NSAMPLES], pmpcol[NSAMPLES], pmmcol[NSAMPLES];
#pragma unroll
        for (int counter = 0; counter < NSAMPLES; counter++) {
          pmcol[counter] = __ldg(&pulseMatrix.coeffRef(counter, ipulse));
          pmpcol[counter] = __ldg(&pulseMatrixP.coeffRef(counter, ipulse));
          pmmcol[counter] = __ldg(&pulseMatrixM.coeffRef(counter, ipulse));
        }

        auto const ampl2 = resultAmplitude * resultAmplitude;
#pragma unroll
        for (int col = 0; col < NSAMPLES; col++) {
          auto const valueP_col = pmpcol[col];
          auto const valueM_col = pmmcol[col];
          auto const value_col = pmcol[col];
          auto const tmppcol = valueP_col - value_col;
          auto const tmpmcol = valueM_col - value_col;

          // diagonal
          auto tmp_value = 0.5 * (tmppcol * tmppcol + tmpmcol * tmpmcol);
          covarianceMatrix(col, col) += ampl2 * tmp_value;

// FIXME: understand if this actually gets unrolled
#pragma unroll
          for (int row = col + 1; row < NSAMPLES; row++) {
            float const valueP_row = pmpcol[row];  //pulseMatrixP(j, ipulseReal);
            float const value_row = pmcol[row];    //pulseMatrix(j, ipulseReal);
            float const valueM_row = pmmcol[row];  //pulseMatrixM(j, ipulseReal);

            float tmpprow = valueP_row - value_row;
            float tmpmrow = valueM_row - value_row;

            auto const covValue = 0.5 * (tmppcol * tmpprow + tmpmcol * tmpmrow);

            covarianceMatrix(row, col) += ampl2 * covValue;
          }
        }
      }
    }

    template <int NSAMPLES, int NPULSES>
    __global__ void kernel_minimize(float* outputEnergy,
                                    float* outputChi2,
                                    float const* __restrict__ inputAmplitudes,
                                    float const* __restrict__ pulseMatrices,
                                    float const* __restrict__ pulseMatricesM,
                                    float const* __restrict__ pulseMatricesP,
                                    int const* __restrict__ pulseOffsetValues,
                                    float const* __restrict__ noiseTerms,
                                    int8_t const* __restrict__ soiSamples,
                                    float const* __restrict__ pedestalWidths,
                                    float const* __restrict__ effectivePedestalWidths,
                                    bool const useEffectivePedestals,
                                    uint32_t const* __restrict__ idsf01HE,
                                    uint32_t const* __restrict__ idsf5HB,
                                    uint32_t const* __restrict__ idsf3HB,
                                    float const* __restrict__ gainValues,
                                    float const* __restrict__ respCorrectionValues,
                                    uint32_t const nchannelsf01HE,
                                    uint32_t const nchannelsf5HB,
                                    uint32_t const nchannelsTotal,
                                    uint32_t const offsetForHashes,
                                    int const maxDepthHB,
                                    int const maxDepthHE,
                                    int const maxPhiHE,
                                    int const firstHBRing,
                                    int const lastHBRing,
                                    int const firstHERing,
                                    int const lastHERing,
                                    int const nEtaHB,
                                    int const nEtaHE) {
      // can be relaxed if needed - minor updates are needed in that case!
      static_assert(NPULSES == NSAMPLES);

      // indices
      auto const gch = threadIdx.x + blockIdx.x * blockDim.x;
      auto const nchannelsf015 = nchannelsf01HE + nchannelsf5HB;
      if (gch >= nchannelsTotal)
        return;

      // if chi2 is set to -9999 do not run minimization
      if (outputChi2[gch] == -9999.f)
        return;

      // configure shared mem
      extern __shared__ char shrmem[];
      float* shrMatrixLFnnlsStorage =
          reinterpret_cast<float*>(shrmem) + calo::multifit::MapSymM<float, NPULSES>::total * threadIdx.x;
      float* shrAtAStorage = reinterpret_cast<float*>(shrmem) +
                             calo::multifit::MapSymM<float, NPULSES>::total * (threadIdx.x + blockDim.x);

      // conditions for pedestal widths
      auto const id = gch < nchannelsf01HE
                          ? idsf01HE[gch]
                          : (gch < nchannelsf015 ? idsf5HB[gch - nchannelsf01HE] : idsf3HB[gch - nchannelsf015]);
      //auto const id = gch >= nchannelsf01HE
      //    ? idsf5HB[gch - nchannelsf01HE]
      //    : idsf01HE[gch];
      auto const did = DetId{id};
      auto const hashedId =
          did.subdetId() == HcalBarrel
              ? hcal::reconstruction::did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
              : hcal::reconstruction::did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE) +
                    offsetForHashes;

      auto const* pedestalWidthsForChannel = useEffectivePedestals && (gch < nchannelsf01HE || gch >= nchannelsf015)
                                                 ? effectivePedestalWidths + hashedId * 4
                                                 : pedestalWidths + hashedId * 4;
      auto const averagePedestalWidth2 = 0.25 * (pedestalWidthsForChannel[0] * pedestalWidthsForChannel[0] +
                                                 pedestalWidthsForChannel[1] * pedestalWidthsForChannel[1] +
                                                 pedestalWidthsForChannel[2] * pedestalWidthsForChannel[2] +
                                                 pedestalWidthsForChannel[3] * pedestalWidthsForChannel[3]);
      auto const* gains = gainValues + hashedId * 4;
      // FIXME on cpu ts 0 capid was used - does it make any difference
      auto const gain = gains[0];
      auto const respCorrection = respCorrectionValues[hashedId];

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
      if (id != DETID_TO_DEBUG)
        return;
#endif
#endif

      /*
      // TODO: provide this properly
      int const soi = soiSamples[gch];
      */
      constexpr float deltaChi2Threashold = 1e-3;

      calo::multifit::ColumnVector<NPULSES, int> pulseOffsets;
#pragma unroll
      for (int i = 0; i < NPULSES; ++i)
        pulseOffsets(i) = i;
      //        pulseOffsets(i) = pulseOffsetValues[i] - pulseOffsetValues[0];

      // output amplitudes/weights
      calo::multifit::ColumnVector<NPULSES> resultAmplitudesVector = calo::multifit::ColumnVector<NPULSES>::Zero();

      // map views
      Eigen::Map<const calo::multifit::ColumnVector<NSAMPLES>> inputAmplitudesView{inputAmplitudes + gch * NSAMPLES};
      Eigen::Map<const calo::multifit::ColumnVector<NSAMPLES>> noiseTermsView{noiseTerms + gch * NSAMPLES};
      Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixMView{pulseMatricesM +
                                                                                              gch * NSAMPLES * NPULSES};
      Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixPView{pulseMatricesP +
                                                                                              gch * NSAMPLES * NPULSES};
      Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixView{pulseMatrices +
                                                                                             gch * NSAMPLES * NPULSES};

#ifdef HCAL_MAHI_GPUDEBUG
      for (int i = 0; i < NSAMPLES; i++)
        printf("inputValues(%d) = %f noiseTerms(%d) = %f\n", i, inputAmplitudesView(i), i, noiseTermsView(i));
      for (int i = 0; i < NSAMPLES; i++) {
        for (int j = 0; j < NPULSES; j++)
          printf("%f ", glbPulseMatrixView(i, j));
        printf("\n");
      }
      printf("\n");
      for (int i = 0; i < NSAMPLES; i++) {
        for (int j = 0; j < NPULSES; j++)
          printf("%f ", glbPulseMatrixMView(i, j));
        printf("\n");
      }
      printf("\n");
      for (int i = 0; i < NSAMPLES; i++) {
        for (int j = 0; j < NPULSES; j++)
          printf("%f ", glbPulseMatrixPView(i, j));
        printf("\n");
      }
#endif

      int npassive = 0;
      float chi2 = 0, previous_chi2 = 0.f, chi2_2itersback = 0.f;
      // TOOD: provide constants from configuration
      for (int iter = 1; iter < 50; iter++) {
        //float covarianceMatrixStorage[MapSymM<float, NSAMPLES>::total];
        // NOTE: only works when NSAMPLES == NPULSES
        // if does not hold -> slightly rearrange shared mem to still reuse
        // shared memory
        float* covarianceMatrixStorage = shrMatrixLFnnlsStorage;
        calo::multifit::MapSymM<float, NSAMPLES> covarianceMatrix{covarianceMatrixStorage};
#pragma unroll
        for (int counter = 0; counter < calo::multifit::MapSymM<float, NSAMPLES>::total; counter++)
          covarianceMatrixStorage[counter] = averagePedestalWidth2;
#pragma unroll
        for (int counter = 0; counter < calo::multifit::MapSymM<float, NSAMPLES>::stride; counter++)
          covarianceMatrix(counter, counter) += __ldg(&noiseTermsView.coeffRef(counter));

        // update covariance matrix
        update_covariance(
            resultAmplitudesVector, covarianceMatrix, glbPulseMatrixView, glbPulseMatrixMView, glbPulseMatrixPView);

#ifdef HCAL_MAHI_GPUDEBUG
        printf("covariance matrix\n");
        for (int i = 0; i < 8; i++) {
          for (int j = 0; j < 8; j++)
            printf("%f ", covarianceMatrix(i, j));
          printf("\n");
        }
#endif

        // compute Cholesky Decomposition L matrix
        //matrixDecomposition.compute(covarianceMatrix);
        //auto const& matrixL = matrixDecomposition.matrixL();
        float matrixLStorage[calo::multifit::MapSymM<float, NSAMPLES>::total];
        calo::multifit::MapSymM<float, NSAMPLES> matrixL{matrixLStorage};
        calo::multifit::compute_decomposition_unrolled(matrixL, covarianceMatrix);

        //
        // replace eigen
        //
        //auto const& A = matrixDecomposition
        //    .matrixL()
        //    .solve(pulseMatrixView);
        calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES> A;
        calo::multifit::solve_forward_subst_matrix(A, glbPulseMatrixView, matrixL);

        //
        // remove eigen
        //
        //auto const& b = matrixL
        //   .solve(inputAmplitudesView);
        //
        float reg_b[NSAMPLES];
        calo::multifit::solve_forward_subst_vector(reg_b, inputAmplitudesView, matrixL);

        // TODO: we do not really need to change these matrcies
        // will be fixed in the optimized version
        //ColMajorMatrix<NPULSES, NPULSES> AtA = A.transpose() * A;
        //ColumnVector<NPULSES> Atb = A.transpose() * b;
        //ColMajorMatrix<NPULSES, NPULSES> AtA;
        //float AtAStorage[MapSymM<float, NPULSES>::total];
        calo::multifit::MapSymM<float, NPULSES> AtA{shrAtAStorage};
        calo::multifit::ColumnVector<NPULSES> Atb;
#pragma unroll
        for (int icol = 0; icol < NPULSES; icol++) {
          float reg_ai[NSAMPLES];

// load column icol
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            reg_ai[counter] = A(counter, icol);

          // compute diagonal
          float sum = 0.f;
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            sum += reg_ai[counter] * reg_ai[counter];

          // store
          AtA(icol, icol) = sum;

// go thru the other columns
#pragma unroll
          for (int j = icol + 1; j < NPULSES; j++) {
            // load column j
            float reg_aj[NSAMPLES];
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              reg_aj[counter] = A(counter, j);

            // accum
            float sum = 0.f;
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              sum += reg_aj[counter] * reg_ai[counter];

            // store
            //AtA(icol, j) = sum;
            AtA(j, icol) = sum;
          }

          // Atb accum
          float sum_atb = 0;
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            sum_atb += reg_ai[counter] * reg_b[counter];

          // store atb
          Atb(icol) = sum_atb;
        }

#ifdef HCAL_MAHI_GPUDEBUG
        printf("AtA\n");
        for (int i = 0; i < 8; i++) {
          for (int j = 0; j < 8; j++)
            printf("%f ", AtA(i, j));
          printf("\n");
        }
        printf("Atb\n");
        for (int i = 0; i < 8; i++)
          printf("%f ", Atb(i));
        printf("\n");
        printf("result Amplitudes before nnls\n");
        for (int i = 0; i < 8; i++)
          printf("%f ", resultAmplitudesVector(i));
        printf("\n");
#endif

        // for fnnls
        calo::multifit::MapSymM<float, NPULSES> matrixLForFnnls{shrMatrixLFnnlsStorage};

        // run fast nnls
        // FIXME: provide values from config
        fnnls(AtA, Atb, resultAmplitudesVector, npassive, pulseOffsets, matrixLForFnnls, 1e-11, 500);

#ifdef HCAL_MAHI_GPUDEBUG
        printf("result Amplitudes\n");
        for (int i = 0; i < 8; i++)
          printf("resultAmplitudes(%d) = %f\n", i, resultAmplitudesVector(i));
#endif

        // replace pulseMatrixView * result - inputs
        // NOTE:
        float accum[NSAMPLES];
        Eigen::Map<calo::multifit::ColumnVector<NSAMPLES>> mapAccum{accum};
        {
          float results[NPULSES];

// preload results and permute according to the pulse offsets
#pragma unroll
          for (int counter = 0; counter < NPULSES; counter++) {
            results[counter] = resultAmplitudesVector[counter];
          }

// load accum
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            accum[counter] = -inputAmplitudesView(counter);

          // iterate
          for (int icol = 0; icol < NPULSES; icol++) {
            float pm_col[NSAMPLES];

// preload a column of pulse matrix
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              pm_col[counter] = __ldg(&glbPulseMatrixView.coeffRef(counter, icol));

// accum
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              accum[counter] += results[icol] * pm_col[counter];
          }
        }

        // compute chi2 and check that there is no rotation
        //chi2 = matrixDecomposition
        //    .matrixL()
        //    . solve(mapAccum)
        //            .solve(pulseMatrixView * resultAmplitudesVector - inputAmplitudesView)
        //    .squaredNorm();
        {
          float reg_b_tmp[NSAMPLES];
          float reg_L[NSAMPLES];
          float accumSum = 0;

// preload a column and load column 0 of cholesky
#pragma unroll
          for (int i = 0; i < NSAMPLES; i++) {
            reg_b_tmp[i] = mapAccum(i);
            reg_L[i] = matrixL(i, 0);
          }

          // compute x0 and store it
          auto x_prev = reg_b_tmp[0] / reg_L[0];
          accumSum += x_prev * x_prev;

// iterate
#pragma unroll
          for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
            for (int counter = iL; counter < NSAMPLES; counter++)
              reg_b_tmp[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
            for (int counter = iL; counter < NSAMPLES; counter++)
              reg_L[counter] = matrixL(counter, iL);

            // compute the next x for M(iL, icol)
            x_prev = reg_b_tmp[iL] / reg_L[iL];

            // store the result value
            accumSum += x_prev * x_prev;
          }

          chi2 = accumSum;
        }

        auto const deltaChi2 = std::abs(chi2 - previous_chi2);
        if (chi2 == chi2_2itersback && chi2 < previous_chi2)
          break;

        // update
        chi2_2itersback = previous_chi2;
        previous_chi2 = chi2;

        // exit condition
        if (deltaChi2 < deltaChi2Threashold)
          break;
      }

#ifdef HCAL_MAHI_GPUDEBUG
      for (int i = 0; i < NPULSES; i++)
        printf("pulseOffsets(%d) = %d outputAmplitudes(%d) = %f\n", i, pulseOffsets(i), i, resultAmplitudesVector(i));
      printf("chi2 = %f\n", chi2);
#endif

      outputChi2[gch] = chi2;
      auto const idx_for_energy = std::abs(pulseOffsetValues[0]);
      outputEnergy[gch] = (gain * resultAmplitudesVector(idx_for_energy)) * respCorrection;
      /*
      #pragma unroll
      for (int i=0; i<NPULSES; i++)
          if (pulseOffsets[i] == soi)
              // NOTE: gain is a number < 10^-3/4, multiply first to avoid stab issues
              outputEnergy[gch] = (gain*resultAmplitudesVector(i))*respCorrection;
      */
    }

    void entryPoint(InputDataGPU const& inputGPU,
                    OutputDataGPU& outputGPU,
                    ConditionsProducts const& conditions,
                    ScratchDataGPU& scratch,
                    ConfigParameters const& configParameters,
                    cudaStream_t cudaStream) {
      auto const totalChannels = inputGPU.f01HEDigis.size + inputGPU.f5HBDigis.size + inputGPU.f3HBDigis.size;

      // FIXME: may be move this assignment to emphasize this more clearly
      // FIXME: number of channels for output might change given that
      //   some channesl might be filtered out
      outputGPU.recHits.size = totalChannels;

      // TODO: this can be lifted by implementing a separate kernel
      // similar to the default one, but properly handling the diff in #samples
      // or modifying existing one
      auto const f01nsamples = compute_nsamples<Flavor01>(inputGPU.f01HEDigis.stride);
      auto const f5nsamples = compute_nsamples<Flavor5>(inputGPU.f5HBDigis.stride);
      auto const f3nsamples = compute_nsamples<Flavor3>(inputGPU.f3HBDigis.stride);
      assert(f01nsamples == f5nsamples && f01nsamples == f3nsamples);

      dim3 threadsPerBlock{f01nsamples, configParameters.kprep1dChannelsPerBlock};
      int blocks = static_cast<uint32_t>(threadsPerBlock.y) > totalChannels
                       ? 1
                       : (totalChannels + threadsPerBlock.y - 1) / threadsPerBlock.y;
      int nbytesShared =
          ((2 * f01nsamples + 2) * sizeof(float) + sizeof(uint64_t)) * configParameters.kprep1dChannelsPerBlock;
      kernel_prep1d_sameNumberOfSamples<<<blocks, threadsPerBlock, nbytesShared, cudaStream>>>(
          scratch.amplitudes.get(),
          scratch.noiseTerms.get(),
          outputGPU.recHits.energy.get(),
          outputGPU.recHits.chi2.get(),
          inputGPU.f01HEDigis.data.get(),
          inputGPU.f5HBDigis.data.get(),
          inputGPU.f3HBDigis.data.get(),
          inputGPU.f01HEDigis.ids.get(),
          inputGPU.f5HBDigis.ids.get(),
          inputGPU.f3HBDigis.ids.get(),
          inputGPU.f01HEDigis.stride,
          inputGPU.f5HBDigis.stride,
          inputGPU.f3HBDigis.stride,
          inputGPU.f01HEDigis.size,
          inputGPU.f5HBDigis.size,
          inputGPU.f5HBDigis.npresamples.get(),
          scratch.soiSamples.get(),
          outputGPU.recHits.energyM0.get(),
          outputGPU.recHits.timeM0.get(),
          outputGPU.recHits.did.get(),
          totalChannels,
          conditions.recoParams.param1,
          conditions.recoParams.param2,
          conditions.qieCoders.offsets,
          conditions.qieCoders.slopes,
          conditions.qieTypes.values,
          conditions.pedestalWidths.values,
          conditions.effectivePedestalWidths.values,
          conditions.pedestals.values,
          conditions.convertedEffectivePedestals ? conditions.convertedEffectivePedestals->values
                                                 : conditions.pedestals.values,
          configParameters.useEffectivePedestals,
          conditions.sipmParameters.type,
          conditions.sipmParameters.fcByPE,
          conditions.sipmCharacteristics.parLin1,
          conditions.sipmCharacteristics.parLin2,
          conditions.sipmCharacteristics.parLin3,
          conditions.gains.values,
          conditions.respCorrs.values,
          conditions.topology->maxDepthHB(),
          conditions.topology->maxDepthHE(),
          conditions.recConstants->getNPhi(1) > hcal::reconstruction::IPHI_MAX ? conditions.recConstants->getNPhi(1)
                                                                               : hcal::reconstruction::IPHI_MAX,
          conditions.topology->firstHBRing(),
          conditions.topology->lastHBRing(),
          conditions.topology->firstHERing(),
          conditions.topology->lastHERing(),
          conditions.recConstants->getEtaRange(0).second - conditions.recConstants->getEtaRange(0).first + 1,
          conditions.topology->firstHERing() > conditions.topology->lastHERing()
              ? 0
              : (conditions.topology->lastHERing() - conditions.topology->firstHERing() + 1),
          configParameters.sipmQTSShift,
          configParameters.sipmQNTStoSum,
          configParameters.firstSampleShift,
          conditions.offsetForHashes,
          configParameters.ts4Thresh);
      cudaCheck(cudaGetLastError());

      // 1024 is the max threads per block for gtx1080
      // FIXME: take this from cuda service or something like that
      uint32_t const channelsPerBlock = 1024 / (f01nsamples * conditions.pulseOffsetsHost.size());
      dim3 threadsPerBlock2{f01nsamples, static_cast<uint32_t>(conditions.pulseOffsetsHost.size()), channelsPerBlock};
      int blocks2 =
          threadsPerBlock2.z > totalChannels ? 1 : (totalChannels + threadsPerBlock2.z - 1) / threadsPerBlock2.z;

#ifdef HCAL_MAHI_CPUDEBUG
      std::cout << "threads: " << threadsPerBlock2.x << " " << threadsPerBlock2.y << "  " << threadsPerBlock2.z
                << std::endl;
      std::cout << "blocks: " << blocks2 << std::endl;
#endif

      kernel_prep_pulseMatrices_sameNumberOfSamples<<<blocks2, threadsPerBlock2, 0, cudaStream>>>(
          scratch.pulseMatrices.get(),
          scratch.pulseMatricesM.get(),
          scratch.pulseMatricesP.get(),
          conditions.pulseOffsets.values,
          scratch.amplitudes.get(),
          inputGPU.f01HEDigis.ids.get(),
          inputGPU.f5HBDigis.ids.get(),
          inputGPU.f3HBDigis.ids.get(),
          inputGPU.f01HEDigis.size,
          inputGPU.f5HBDigis.size,
          totalChannels,
          scratch.soiSamples.get(),
          conditions.recoParams.ids,
          conditions.recoParams.acc25nsVec,
          conditions.recoParams.diff25nsItvlVec,
          conditions.recoParams.accVarLenIdxMinusOneVec,
          conditions.recoParams.diffVarItvlIdxMinusOneVec,
          conditions.recoParams.accVarLenIdxZEROVec,
          conditions.recoParams.diffVarItvlIdxZEROVec,
          configParameters.meanTime,
          configParameters.timeSigmaSiPM,
          configParameters.timeSigmaHPD,
          conditions.topology->maxDepthHB(),
          conditions.topology->maxDepthHE(),
          conditions.recConstants->getNPhi(1) > hcal::reconstruction::IPHI_MAX ? conditions.recConstants->getNPhi(1)
                                                                               : hcal::reconstruction::IPHI_MAX,
          conditions.topology->firstHBRing(),
          conditions.topology->lastHBRing(),
          conditions.topology->firstHERing(),
          conditions.topology->lastHERing(),
          conditions.recConstants->getEtaRange(0).second - conditions.recConstants->getEtaRange(0).first + 1,
          conditions.topology->firstHERing() > conditions.topology->lastHERing()
              ? 0
              : (conditions.topology->lastHERing() - conditions.topology->firstHERing() + 1),
          conditions.offsetForHashes,
          configParameters.applyTimeSlew,
          configParameters.tzeroTimeSlew,
          configParameters.slopeTimeSlew,
          configParameters.tmaxTimeSlew);
      cudaCheck(cudaGetLastError());

      if (f01nsamples == 8 && conditions.pulseOffsetsHost.size() == 8u) {
        // FIXME: provide constants from configuration
        uint32_t threadsPerBlock = configParameters.kernelMinimizeThreads[0];
        uint32_t blocks = threadsPerBlock > totalChannels ? 1 : (totalChannels + threadsPerBlock - 1) / threadsPerBlock;
        auto const nbytesShared = 2 * threadsPerBlock * calo::multifit::MapSymM<float, 8>::total * sizeof(float);
        kernel_minimize<8, 8><<<blocks, threadsPerBlock, nbytesShared, cudaStream>>>(
            outputGPU.recHits.energy.get(),
            outputGPU.recHits.chi2.get(),
            scratch.amplitudes.get(),
            scratch.pulseMatrices.get(),
            scratch.pulseMatricesM.get(),
            scratch.pulseMatricesP.get(),
            conditions.pulseOffsets.values,
            scratch.noiseTerms.get(),
            scratch.soiSamples.get(),
            conditions.pedestalWidths.values,
            conditions.effectivePedestalWidths.values,
            configParameters.useEffectivePedestals,
            inputGPU.f01HEDigis.ids.get(),
            inputGPU.f5HBDigis.ids.get(),
            inputGPU.f3HBDigis.ids.get(),
            conditions.gains.values,
            conditions.respCorrs.values,
            inputGPU.f01HEDigis.size,
            inputGPU.f5HBDigis.size,
            totalChannels,
            conditions.offsetForHashes,
            conditions.topology->maxDepthHB(),
            conditions.topology->maxDepthHE(),
            conditions.recConstants->getNPhi(1) > hcal::reconstruction::IPHI_MAX ? conditions.recConstants->getNPhi(1)
                                                                                 : hcal::reconstruction::IPHI_MAX,
            conditions.topology->firstHBRing(),
            conditions.topology->lastHBRing(),
            conditions.topology->firstHERing(),
            conditions.topology->lastHERing(),
            conditions.recConstants->getEtaRange(0).second - conditions.recConstants->getEtaRange(0).first + 1,
            conditions.topology->firstHERing() > conditions.topology->lastHERing()
                ? 0
                : (conditions.topology->lastHERing() - conditions.topology->firstHERing() + 1));
      } else {
        throw cms::Exception("Invalid MahiGPU configuration")
            << "Currently support only 8 pulses and 8 time samples and provided: " << f01nsamples << " samples and "
            << conditions.pulseOffsetsHost.size() << " pulses" << std::endl;
      }
    }

  }  // namespace mahi
}  // namespace hcal
