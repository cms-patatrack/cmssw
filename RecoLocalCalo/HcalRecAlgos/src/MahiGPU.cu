#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/MahiGPU.h"

#include <cuda/api_wrappers.h>

namespace hcal { namespace mahi {

__constant__
float const qie8shape[129] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 
    7, 8, 9, 10, 11, 12, 13, 14, 
    16, 18, 20, 22, 24, 26, 28, 31, 
    34, 37, 40, 44, 48, 52, 57, 62, 
    57, 62, 67, 72, 77, 82, 87, 92, 
    97, 102, 107, 112, 117, 122, 127, 132, 
    142, 152, 162, 172, 182, 192, 202, 217, 
    232, 247, 262, 282, 302, 322, 347, 372, 
    347, 372, 397, 422, 447, 472, 497, 522, 
    547, 572, 597, 622, 647, 672, 697, 722, 
    772, 822, 872, 922, 972, 1022, 1072, 1147, 
    1222, 1297, 1372, 1472, 1572, 1672, 1797, 1922, 
    1797, 1922, 2047, 2172, 2297, 2422, 2547, 2672, 
    2797, 2922, 3047, 3172, 3297, 3422, 3547, 3672, 
    3922, 4172, 4422, 4672, 4922, 5172, 5422, 5797, 
    6172, 6547, 6922, 7422, 7922, 8422, 9047, 9672, 
    10297
};

__constant__
float const qie11shape[257] = {
    -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 
    7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 
    15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5, 29.5, 
    31.5, 33.5, 35.5, 37.5, 39.5, 41.5, 43.5, 45.5, 
    47.5, 49.5, 51.5, 53.5, 55.5, 59.5, 63.5, 67.5, 
    71.5, 75.5, 79.5, 83.5, 87.5, 91.5, 95.5, 99.5, 
    103.5, 107.5, 111.5, 115.5, 119.5, 123.5, 127.5, 131.5, 
    135.5, 139.5, 147.5, 155.5, 163.5, 171.5, 179.5, 187.5, 
    171.5, 179.5, 187.5, 195.5, 203.5, 211.5, 219.5, 227.5, 
    235.5, 243.5, 251.5, 259.5, 267.5, 275.5, 283.5, 291.5, 
    299.5, 315.5, 331.5, 347.5, 363.5, 379.5, 395.5, 411.5, 
    427.5, 443.5, 459.5, 475.5, 491.5, 507.5, 523.5, 539.5, 
    555.5, 571.5, 587.5, 603.5, 619.5, 651.5, 683.5, 715.5, 
    747.5, 779.5, 811.5, 843.5, 875.5, 907.5, 939.5, 971.5, 
    1003.5, 1035.5, 1067.5, 1099.5, 1131.5, 1163.5, 1195.5, 1227.5, 
    1259.5, 1291.5, 1355.5, 1419.5, 1483.5, 1547.5, 1611.5, 1675.5, 
    1547.5, 1611.5, 1675.5, 1739.5, 1803.5, 1867.5, 1931.5, 1995.5, 
    2059.5, 2123.5, 2187.5, 2251.5, 2315.5, 2379.5, 2443.5, 2507.5, 
    2571.5, 2699.5, 2827.5, 2955.5, 3083.5, 3211.5, 3339.5, 3467.5, 
    3595.5, 3723.5, 3851.5, 3979.5, 4107.5, 4235.5, 4363.5, 4491.5, 
    4619.5, 4747.5, 4875.5, 5003.5, 5131.5, 5387.5, 5643.5, 5899.5, 
    6155.5, 6411.5, 6667.5, 6923.5, 7179.5, 7435.5, 7691.5, 7947.5, 
    8203.5, 8459.5, 8715.5, 8971.5, 9227.5, 9483.5, 9739.5, 9995.5, 
    10251.5, 10507.5, 11019.5, 11531.5, 12043.5, 12555.5, 13067.5, 13579.5, 
    12555.5, 13067.5, 13579.5, 14091.5, 14603.5, 15115.5, 15627.5, 16139.5, 
    16651.5, 17163.5, 17675.5, 18187.5, 18699.5, 19211.5, 19723.5, 20235.5, 
    20747.5, 21771.5, 22795.5, 23819.5, 24843.5, 25867.5, 26891.5, 27915.5, 
    28939.5, 29963.5, 30987.5, 32011.5, 33035.5, 34059.5, 35083.5, 36107.5, 
    37131.5, 38155.5, 39179.5, 40203.5, 41227.5, 43275.5, 45323.5, 47371.5, 
    49419.5, 51467.5, 53515.5, 55563.5, 57611.5, 59659.5, 61707.5, 63755.5, 
    65803.5, 67851.5, 69899.5, 71947.5, 73995.5, 76043.5, 78091.5, 80139.5, 
    82187.5, 84235.5, 88331.5, 92427.5, 96523.5, 100620, 104716, 108812, 
    112908
};

constexpr int32_t IPHI_MAX = 72;

__forceinline__ __device__
uint32_t did2linearIndexHB(
        uint32_t const didraw, int const maxDepthHB, 
        int const firstHBRing, int const lastHBRing,
        int const nEtaHB) {
    HcalDetId did{didraw};
    uint32_t const value = (did.depth() - 1) + maxDepthHB * (did.iphi() - 1);
    return did.ieta() > 0
        ? value + maxDepthHB * IPHI_MAX * (did.ieta() - firstHBRing)
        : value + maxDepthHB * IPHI_MAX * (did.ieta() + lastHBRing + nEtaHB);
}

__forceinline__ __device__
uint32_t did2linearIndexHE(
        uint32_t const didraw, int const maxDepthHE, int const maxPhiHE,
        int const firstHERing, int const lastHERing,
        int const nEtaHE) {
    HcalDetId did{didraw};
    uint32_t const value = (did.depth() - 1) + maxDepthHE * (did.iphi() - 1);
    return did.ieta() > 0
        ? value + maxDepthHE * maxPhiHE * (did.ieta() - firstHERing)
        : value + maxDepthHE * maxPhiHE * (did.ieta() + lastHERing + nEtaHE);
}

__forceinline__ __device__
uint32_t get_qiecoder_index(uint32_t const capid, uint32_t const range) {
    return capid * 4 + range;
}

__forceinline__ __device__
float compute_reco_correction_factor(
        float const par1,
        float const par2, 
        float const par3,
        float const x) {
    return par3*x*x + par2*x + par1;
}

__forceinline__ __device__
float compute_coder_charge(
        int const qieType,
        uint8_t const adc,
        uint8_t const capid,
        float const* qieOffsets,
        float const* qieSlopes) {
    auto const range = qieType==0
        ? (adc >> 5) & 0x3
        : (adc >> 6) & 0x3;
    auto const* qieShapeToUse = qieType==0 ? qie8shape : qie11shape;
    auto const nbins = qieType==0 ? 32 : 64;
    auto const center = adc % nbins == nbins-1
        ? 0.5 * (3 * qieShapeToUse[adc] - qieShapeToUse[adc-1])
        : 0.5 * (qieShapeToUse[adc] + qieShapeToUse[adc+1]);
    auto const index = get_qiecoder_index(capid, range);
    return (center - qieOffsets[index]) / qieSlopes[index];
}

__forceinline__ __device__
float compute_diff_charge_gain(
        int const qieType, 
        uint8_t adc, 
        uint8_t const capid,
        float const* qieOffsets,
        float const* qieSlopes,
        bool const isqie11) {
    constexpr uint32_t mantissaMaskQIE8 = 0x1fu;
    constexpr uint32_t mantissaMaskQIE11 = 0x3f;
    auto const mantissaMask = isqie11 ? mantissaMaskQIE11 : mantissaMaskQIE8;
    auto const q = compute_coder_charge(qieType, adc, capid, qieOffsets, qieSlopes);
    auto const mantissa = adc & mantissaMask;

    if (mantissa==0u || mantissa==mantissaMask-1u)
        return compute_coder_charge(
            qieType, adc+1u, capid, qieOffsets, qieSlopes) - q;
    else if (mantissa==1u || mantissa==mantissaMask)
        return q - compute_coder_charge(
            qieType, adc-1u, capid, qieOffsets, qieSlopes);
    else {
        auto const qup = compute_coder_charge(
            qieType, adc+1u, capid, qieOffsets, qieSlopes);
        auto const qdown = compute_coder_charge(
            qieType, adc-1u, capid, qieOffsets, qieSlopes);
        auto const upgain = qup - q;
        auto const downgain = q - qdown;
        auto const averagegain = (qup - qdown) / 2.f;
        if (std::abs(upgain - downgain) < 0.01f*averagegain)
            return averagegain;
        else {
            auto const q2up = compute_coder_charge(
                qieType, adc+2u, capid, qieOffsets, qieSlopes);
            auto const q2down = compute_coder_charge(
                qieType, adc-2u, capid, qieOffsets, qieSlopes);
            auto const upgain2 = q2up - qup;
            auto const downgain2 = qdown - q2down;
            if (std::abs(upgain2 - upgain) < std::abs(downgain2 - downgain))
                return upgain;
            else 
                return downgain;
        }
    }
}

// TODO: add/validate restrict (will increase #registers in use by the kernel)
__global__
void kernel_prep1d_sameNumberOfSamples(
        uint16_t const* dataf01HE,
        uint16_t const* dataf5HB,
        uint32_t const* idsf01HE,
        uint32_t const* idsf5HB,
        uint32_t const stridef01HE,
        uint32_t const stridef5HB,
        uint32_t const nchannelsf01HE,
        float *method0Energy,
        float *method0Time,
        uint32_t *outputdid,
        uint32_t const nchannels,
        uint32_t const* recoParam1Values,
        uint32_t const* recoParam2Values,
        float const* qieCoderOffsets,
        float const* qieCoderSlopes,
        int const* qieTypes,
        float const* pedestals,
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
        uint32_t const offsetForHashes) {
    // indices + runtime constants
    auto const sample = threadIdx.x;
    int32_t const nsamplesExpected = blockDim.x;
    auto const lch = threadIdx.y;
    auto const gch = lch + blockDim.y*blockIdx.x;
    auto const nchannels_per_block = blockDim.x;
    auto const linearThPerBlock = threadIdx.x + threadIdx.y*blockDim.x;

    constexpr uint32_t mantissaMaskQIE8 = 0x1fu;
    constexpr uint32_t mantissaMaskQIE11 = 0x3f;

    // remove 
    if (gch >= nchannels) return;

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_SINGLECHANNEL
    if (gch > 0) return;
#endif
#endif

    // configure shared mem
    extern __shared__ char smem[];
    float *shrEnergyM0PerTS = reinterpret_cast<float*>(smem);
    float *shrChargeMinusPedestal = shrEnergyM0PerTS +
        nsamplesExpected*nchannels_per_block;
    float *shrMethod0EnergyAccum = shrChargeMinusPedestal + 
        nsamplesExpected*nchannels_per_block;
    unsigned long long int *shrMethod0EnergySamplePair = 
        reinterpret_cast<unsigned long long int*>(
            shrMethod0EnergyAccum + nchannels_per_block);
    if (sample == 0) {
        shrMethod0EnergyAccum[lch] = 0;
        shrMethod0EnergySamplePair[lch] = 
            __float_as_uint(std::numeric_limits<float>::min());
    }

    // get event input quantities
    auto const stride = gch >= nchannelsf01HE
        ? stridef5HB
        : stridef01HE;
    auto const nsamples = gch >= nchannelsf01HE
        ? compute_nsamples<Flavor5>(stride)
        : compute_nsamples<Flavor01>(stride);
    
#ifdef HCAL_MAHI_GPUDEBUG
    assert(nsamples == nsamplesExpected);
#endif

    auto const id = gch >= nchannelsf01HE
        ? idsf5HB[gch - nchannelsf01HE]
        : idsf01HE[gch];
    auto const did = HcalDetId{id};
    auto const adc = gch >= nchannelsf01HE
        ? adc_for_sample<Flavor5>(
            dataf5HB + stride*(gch - nchannelsf01HE), sample)
        : adc_for_sample<Flavor01>(dataf01HE + stride*gch, sample);
    auto const capid = gch >= nchannelsf01HE
        ? capid_for_sample<Flavor5>(
            dataf5HB + stride*(gch - nchannelsf01HE), sample)
        : capid_for_sample<Flavor01>(dataf01HE + stride*gch, sample);

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
    if (id != 1160268851) return;
#endif
#endif

    // compute hash for this did
    auto const hashedId = did.subdetId() == HcalBarrel
        ? did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
        : did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE)
            + offsetForHashes;

    // conditions based on the hash
    // FIXME: remove hardcoded values
    auto const qieType = qieTypes[hashedId] > 0 ? 1 : 0; // 2 types at this point
    auto const* qieOffsets = qieCoderOffsets + 
        hashedId*HcalQIECodersGPU::numValuesPerChannel;
    auto const* qieSlopes = qieCoderSlopes +
        hashedId*HcalQIECodersGPU::numValuesPerChannel;
    auto const* pedestalsForChannel = pedestals +
        hashedId*4;
    auto const* gains = gainValues + 
        hashedId*4;
    auto const gain = gains[capid];
    auto const respCorrection = respCorrectionValues[hashedId];
    auto const pedestal = pedestalsForChannel[capid];
    auto const sipmType = sipmTypeValues[hashedId];
    auto const fcByPE = fcByPEValues[hashedId];
    auto const recoParam1 = recoParam1Values[hashedId];
    auto const recoParam2 = recoParam2Values[hashedId];

#ifdef HCAL_MAHI_GPUDEBUG
    printf("qieType = %d qieOffset0 = %f qieOffset1 = %f qieSlope0 = %f qieSlope1 = %f\n", qieType, qieOffsets[0], qieOffsets[1], qieSlopes[0], qieSlopes[1]);
#endif

    // compute charge
    auto const charge = compute_coder_charge(
        qieType, adc, capid, qieOffsets, qieSlopes);

    // NOTE: this synching is really only needed for flavor 01 channels
    shrChargeMinusPedestal[linearThPerBlock] = charge - pedestal;
    __syncthreads();

    // TODO: this should be properly treated
    int32_t soi = 4;
    
    // 
    // compute various quantities (raw charge and tdc stuff)
    // NOTE: this branch will be divergent only for a single warp that 
    // sits on the boundary when flavor 01 channels end and flavor 5 start
    //
    float rawCharge;
    float tdcTime;
    auto const dfc = compute_diff_charge_gain(
        qieType,
        adc,
        capid,
        qieOffsets,
        qieSlopes,
        gch < nchannelsf01HE);
    if (gch >= nchannelsf01HE) {
        // flavor 5
        rawCharge = charge;
        tdcTime = HcalSpecialTimes::UNKNOWN_T_NOTDC;
    } else {
        // flavor 0 or 1
        // conditions needed for sipms
        auto const parLin1 = parLin1Values[sipmType-1];
        auto const parLin2 = parLin2Values[sipmType-1];
        auto const parLin3 = parLin3Values[sipmType-1];

        int const first = std::max(soi + sipmQTSShift, 0);
        int const last = std::min(soi + sipmQNTStoSum, nsamplesExpected);
        float sipmq = 0.0f;
        for (auto ts=first; ts<last; ts++)
            sipmq += shrChargeMinusPedestal[threadIdx.y*nsamplesExpected + ts];
        auto const effectivePixelsFired = sipmq / fcByPE;
        auto const factor = compute_reco_correction_factor(
            parLin1, parLin2, parLin3, effectivePixelsFired);
        auto const rawCharge = (charge - pedestal)*factor + pedestal;
        tdcTime = HcalSpecialTimes::getTDCTime(
            tdc_for_sample<Flavor01>(dataf01HE + stride*gch, sample));

#ifdef HCAL_MAHI_GPUDEBUG
        printf("first = %d last = %d factor = %f\n", first, last, factor);
#endif
    }

    // compute method 0 quantities
    // TODO: need to apply containment
    // TODO: need to apply time slew
    // TODO: for < run 3, apply HBM legacy energy correction
    auto const nsamplesToAdd = 
        recoParam1 < 10 ? recoParam2 : (recoParam1 >> 14) & 0xF;
    auto const startSampleTmp = soi + firstSampleShift;
    auto const startSample = startSampleTmp<0 ? 0 : startSampleTmp;
    auto const endSample = startSample + nsamplesToAdd < nsamples
        ? startSample + nsamplesToAdd
        : nsamples;
    // NOTE: gain is a small number < 10^-3, multiply it last
    auto const energym0_per_ts = gain*((rawCharge - pedestal)* respCorrection);
    // store to shared mem
    shrEnergyM0PerTS[lch*nsamplesExpected + sample] = energym0_per_ts;

#ifdef HCAL_MAHI_GPUDEBUG
    printf("id = %u sample = %d gch = %d hashedId = %u adc = %u capid = %u\n"
        "   charge = %f rawCharge = %f dfc = %f pedestal = %f\n"
        "   gain = %f respCorrection = %f energym0_per_ts = %f\n",
        id, sample, gch, hashedId, adc, capid, charge, rawCharge, dfc, pedestal,
        gain, respCorrection, energym0_per_ts);
    printf("startSample = %d endSample = %d param1 = %u param2 = %u\n",
        startSample, endSample, recoParam1, recoParam2);
#endif

    // FIXME: do we need to provide a global memory atomics
    // alternative for the Kepler arch?
    if (sample>=startSample && sample<endSample) {
        atomicAdd(&shrMethod0EnergyAccum[lch], energym0_per_ts);
        // pack sample, energy as 64 bit value
        unsigned long long int old = shrMethod0EnergySamplePair[lch], assumed;
        unsigned long long int val = 
            (static_cast<unsigned long long int>(sample) << 32) + 
            __float_as_uint(energym0_per_ts);
        do {
            assumed = old;
            // decode energy, sample values
            int const current_sample = (assumed >> 32) & 0xffffffff;
            float const current_energy = __uint_as_float(assumed & 0xffffffff);
            if (energym0_per_ts > current_energy)
                old = atomicCAS(&shrMethod0EnergySamplePair[lch], assumed, val);
            else 
                break;
        } while (assumed != old);
    }
    __syncthreads();

    if (sample == 0) {
        auto const method0_energy = shrMethod0EnergyAccum[lch];
        auto const val = shrMethod0EnergySamplePair[lch];
        int const max_sample = (val >> 32) & 0xffffffff;
        float const max_energy = __uint_as_float(val & 0xffffffff);
        float const max_energy_1 = max_sample < nsamples-1
            ? shrEnergyM0PerTS[lch*nsamplesExpected + max_sample + 1]
            : 0.f;
        float const position = nsamplesToAdd < nsamples
            ? max_sample - soi
            : max_sample;
        auto const sum = max_energy + max_energy_1;
        float const time = max_energy>0.f && max_energy_1>0.f
            ? 25.f * (position + max_energy_1 / sum)
            : 25.f * position;

        // store method0 quantities to global mem
        outputdid[gch] = id;
        method0Energy[gch] = method0_energy;
        method0Time[gch] = time;

#ifdef HCAL_MAHI_GPUDEBUG
        printf("method0_energy = %f max_sample = %d max_energy = %f time = %f\n",
            method0_energy, max_sample, max_energy, time);
#endif
    }
}

void entryPoint(
        InputDataGPU const& inputGPU,
        OutputDataGPU& outputGPU,
        ConditionsProducts const& conditions,
        ConfigParameters const& configParameters,
        cuda::stream_t<>& cudaStream) {
    auto const totalChannels = inputGPU.f01HEDigis.ndigis + inputGPU.f5HBDigis.ndigis;

    // FIXME: may be move this assignment to emphasize this more clearly
    // FIXME: number of channels for output might change given that 
    //   some channesl might be filtered out
    outputGPU.recHits.size = totalChannels;

    // TODO: this can be lifted by implementing a separate kernel
    // similar to the default one, but properly handling the diff in #samples
    auto const f01nsamples = compute_nsamples<Flavor01>(inputGPU.f01HEDigis.stride);
    auto const f5nsamples = compute_nsamples<Flavor5>(inputGPU.f5HBDigis.stride);
    assert(f01nsamples == f5nsamples);

    dim3 threadsPerBlock{f01nsamples, configParameters.kprep1dChannelsPerBlock};
    int blocks = static_cast<uint32_t>(threadsPerBlock.y) > totalChannels
        ? 1
        : (totalChannels + threadsPerBlock.y - 1) / threadsPerBlock.y;
    int nbytesShared = (2*f01nsamples + 2)*sizeof(float)*
        configParameters.kprep1dChannelsPerBlock;
    kernel_prep1d_sameNumberOfSamples<<<blocks, threadsPerBlock, nbytesShared, cudaStream.id()>>>(
        inputGPU.f01HEDigis.data,
        inputGPU.f5HBDigis.data,
        inputGPU.f01HEDigis.ids,
        inputGPU.f5HBDigis.ids,
        inputGPU.f01HEDigis.stride,
        inputGPU.f5HBDigis.stride,
        inputGPU.f01HEDigis.ndigis,
        outputGPU.recHits.energy,
        outputGPU.recHits.time,
        outputGPU.recHits.did,
        totalChannels,
        conditions.recoParams.param1,
        conditions.recoParams.param2,
        conditions.qieCoders.offsets,
        conditions.qieCoders.slopes,
        conditions.qieTypes.values,
        conditions.pedestals.values,
        conditions.sipmParameters.type,
        conditions.sipmParameters.fcByPE,
        conditions.sipmCharacteristics.parLin1,
        conditions.sipmCharacteristics.parLin2,
        conditions.sipmCharacteristics.parLin3,
        conditions.gains.values,
        conditions.respCorrs.values,
        conditions.topology->maxDepthHB(),
        conditions.topology->maxDepthHE(),
        conditions.recConstants->getNPhi(1) > IPHI_MAX
            ? conditions.recConstants->getNPhi(1)
            : IPHI_MAX,
        conditions.topology->firstHBRing(),
        conditions.topology->lastHBRing(),
        conditions.topology->firstHERing(),
        conditions.topology->lastHERing(),
        conditions.recConstants->getEtaRange(0).second - 
            conditions.recConstants->getEtaRange(0).first + 1,
        conditions.topology->firstHERing() > conditions.topology->lastHERing() 
            ? 0
            : (conditions.topology->lastHERing() - 
                conditions.topology->firstHERing() + 1),
        configParameters.sipmQTSShift,
        configParameters.sipmQNTStoSum,
        configParameters.firstSampleShift,
        conditions.offsetForHashes
    );
    cudaCheck( cudaGetLastError() );
}

}}
