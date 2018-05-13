//
// Author: Felice Pantaleo, CERN
//
#ifndef GPU_CACELL_H_
#define GPU_CACELL_H_

#include "GPUHitsAndDoublets.h"
#include "GPUSimpleVector.h"
#include <cuda.h>
struct Quadruplet {
  int2 layerPairsAndCellId[3];
};

class GPUCACell {
public:
  __host__ __device__ GPUCACell() {}

  __host__ __device__ void init(const GPULayerDoublets *doublets,
                                const GPULayerHits *hitsOnLayer,
                                int layerPairId, int doubletId, int innerHitId,
                                int outerHitId, float regionX, float regionY) {

    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;

    theDoublets = doublets;

    theDoubletId = doubletId;
    theLayerPairId = layerPairId;

    auto innerLayerId = doublets->innerLayerId;
    auto outerLayerId = doublets->outerLayerId;

    theInnerX = hitsOnLayer[innerLayerId].x[innerHitId];
    theOuterX = hitsOnLayer[outerLayerId].x[outerHitId];

    theInnerY = hitsOnLayer[innerLayerId].y[innerHitId];
    theOuterY = hitsOnLayer[outerLayerId].y[outerHitId];

    theInnerZ = hitsOnLayer[innerLayerId].z[innerHitId];
    theOuterZ = hitsOnLayer[outerLayerId].z[outerHitId];
    theInnerR = hypot(theInnerX - regionX, theInnerY - regionY);
    theOuterR = hypot(theOuterX - regionX, theOuterY - regionY);
    theOuterNeighbors.reset();
  }

  __host__ __device__ float get_inner_x() const { return theInnerX; }
  __host__ __device__ float get_outer_x() const { return theOuterX; }
  __host__ __device__ float get_inner_y() const { return theInnerY; }
  __host__ __device__ float get_outer_y() const { return theOuterY; }
  __host__ __device__ float get_inner_z() const { return theInnerZ; }
  __host__ __device__ float get_outer_z() const { return theOuterZ; }
  __host__ __device__ float get_inner_r() const { return theInnerR; }
  __host__ __device__ float get_outer_r() const { return theOuterR; }
  __host__ __device__ unsigned int get_inner_hit_id() const {
    return theInnerHitId;
  }
  __host__ __device__ unsigned int get_outer_hit_id() const {
    return theOuterHitId;
  }

  __host__ __device__ void print_cell() const {

    printf("printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: "
           "%d, innerradius %f, outerRadius %f \n",
           theDoubletId, theLayerPairId, theInnerHitId, theOuterHitId,
           theInnerR, theOuterR);
  }

  //        __host__    __device__
  //        void print_neighbors(int minNumberOfNeighbors = 0) const
  //        {
  //            if (theOuterNeighbors.m_size >= minNumberOfNeighbors)
  //            {
  //
  //                printf("\n\tIt has %d outerneighbors: \n",
  //                theOuterNeighbors.m_size); for (int i = 0; i <
  //                theOuterNeighbors.m_size; ++i)
  //                {
  //                    printf("\n\t\t%d outerneighbor: \n\t\t", i);
  //                    theOuterNeighbors.m_data[i]->print_cell();
  //
  //                }
  //            }
  //        }

  __host__ __device__ bool check_alignment_and_tag(
      const GPUCACell *cells, unsigned int innerCellId, const float ptmin,
      const float region_origin_x, const float region_origin_y,
      const float region_origin_radius, const float thetaCut,
      const float phiCut, const float hardPtCut) {
    auto ro = get_outer_r();
    auto zo = get_outer_z();
    const auto &otherCell = cells[innerCellId];

    auto r1 = otherCell.get_inner_r();
    auto z1 = otherCell.get_inner_z();
    bool aligned = areAlignedRZ(r1, z1, ro, zo, ptmin, thetaCut);
    // if(aligned) printf("\n they're aligned!\n");
    return (aligned &&
            haveSimilarCurvature(cells, innerCellId, ptmin, region_origin_x,
                                 region_origin_y, region_origin_radius, phiCut,
                                 hardPtCut));
  }
  __host__ __device__ bool areAlignedRZ(float r1, float z1, float ro, float zo,
                                        const float ptmin,
                                        const float thetaCut) const {
    float radius_diff = std::abs(r1 - ro);
    float distance_13_squared =
        radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

    float pMin =
        ptmin * std::sqrt(distance_13_squared); // this needs to be divided by
                                                // radius_diff later

    float tan_12_13_half_mul_distance_13_squared =
        fabs(z1 * (get_inner_r() - ro) + get_inner_z() * (ro - r1) +
             zo * (r1 - get_inner_r()));
    // printf("\n areAlignedRZresult %f, thetaCut %f", tan_12_13_half_mul_distance_13_squared * pMin/(distance_13_squared * radius_diff), thetaCut);
    return tan_12_13_half_mul_distance_13_squared * pMin <=
           thetaCut * distance_13_squared * radius_diff;
  }

  __host__ __device__ bool
  haveSimilarCurvature(const GPUCACell *cells, unsigned int innerCellId,
                       const float ptmin, const float region_origin_x,
                       const float region_origin_y,
                       const float region_origin_radius, const float phiCut,
                       const float hardPtCut) const {

    const auto &otherCell = cells[innerCellId];

    auto x1 = otherCell.get_inner_x();
    auto y1 = otherCell.get_inner_y();

    auto x2 = get_inner_x();
    auto y2 = get_inner_y();

    auto x3 = get_outer_x();
    auto y3 = get_outer_y();

    float distance_13_squared = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
    float tan_12_13_half_mul_distance_13_squared =
        fabs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2));
    // high pt : just straight
    if (tan_12_13_half_mul_distance_13_squared * ptmin <=
        1.0e-4f * distance_13_squared) {

      float distance_3_beamspot_squared =
          (x3 - region_origin_x) * (x3 - region_origin_x) +
          (y3 - region_origin_y) * (y3 - region_origin_y);

      float dot_bs3_13 = ((x1 - x3) * (region_origin_x - x3) +
                          (y1 - y3) * (region_origin_y - y3));
      float proj_bs3_on_13_squared =
          dot_bs3_13 * dot_bs3_13 / distance_13_squared;

      float distance_13_beamspot_squared =
          distance_3_beamspot_squared - proj_bs3_on_13_squared;

      return distance_13_beamspot_squared <
             (region_origin_radius + phiCut) * (region_origin_radius + phiCut);
    }

    // 87 cm/GeV = 1/(3.8T * 0.3)

    // take less than radius given by the hardPtCut and reject everything below
    float minRadius = hardPtCut * 87.f; // FIXME move out and use real MagField

    auto det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);

    auto offset = x2 * x2 + y2 * y2;

    auto bc = (x1 * x1 + y1 * y1 - offset) * 0.5f;

    auto cd = (offset - x3 * x3 - y3 * y3) * 0.5f;

    auto idet = 1.f / det;

    auto x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
    auto y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

    auto radius = std::sqrt((x2 - x_center) * (x2 - x_center) +
                            (y2 - y_center) * (y2 - y_center));

    if (radius < minRadius)
      return false; // hard cut on pt

    auto centers_distance_squared =
        (x_center - region_origin_x) * (x_center - region_origin_x) +
        (y_center - region_origin_y) * (y_center - region_origin_y);
    auto region_origin_radius_plus_tolerance = region_origin_radius + phiCut;
    auto minimumOfIntersectionRange =
        (radius - region_origin_radius_plus_tolerance) *
        (radius - region_origin_radius_plus_tolerance);

    if (centers_distance_squared >= minimumOfIntersectionRange) {
      auto maximumOfIntersectionRange =
          (radius + region_origin_radius_plus_tolerance) *
          (radius + region_origin_radius_plus_tolerance);
      return centers_distance_squared <= maximumOfIntersectionRange;
    }

    return false;
  }

  // trying to free the track building process from hardcoded layers, leaving
  // the visit of the graph based on the neighborhood connections between cells.

  template <int maxNumberOfQuadruplets>
  __device__ inline void find_ntuplets(
      const GPUCACell *cells,
      GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *foundNtuplets,
      GPUSimpleVector<3, unsigned int> &tmpNtuplet,
      const unsigned int minHitsPerNtuplet) const {

    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold
    Quadruplet tmpQuadruplet;

    if (tmpNtuplet.size() >= minHitsPerNtuplet - 1) {
      //                if(tmpNtuplet.size()==3)
      //                {
      //                    printf("\ntmpNtuplet contains %d:\n",
      //                    tmpNtuplet.size()); for(auto i = 0; i<
      //                    tmpNtuplet.size(); ++i)
      //                    {
      //                       printf("\t\t%d innerhit outerhit (xyz) (%f %f
      //                       %f), (%f %f %f)\n", tmpNtuplet.m_data[i],
      //                       cells[tmpNtuplet.m_data[i]].get_inner_x(),
      //                                cells[tmpNtuplet.m_data[i]].get_inner_y(),
      //                                cells[tmpNtuplet.m_data[i]].get_inner_z(),
      //                                cells[tmpNtuplet.m_data[i]].get_outer_x(),
      //                                cells[tmpNtuplet.m_data[i]].get_outer_y(),
      //                                cells[tmpNtuplet.m_data[i]].get_outer_z());
      //
      //                    }
      //                }

      for (int i = 0; i < minHitsPerNtuplet - 1; ++i) {
        tmpQuadruplet.layerPairsAndCellId[i].x =
            cells[tmpNtuplet.m_data[i]].theLayerPairId;
        tmpQuadruplet.layerPairsAndCellId[i].y = tmpNtuplet.m_data[i];
      }
      foundNtuplets->push_back_ts(tmpQuadruplet);

    }

    else {

      for (int j = 0; j < theOuterNeighbors.size(); ++j) {

        auto otherCell = theOuterNeighbors.m_data[j];
        tmpNtuplet.push_back(otherCell);
        cells[otherCell].find_ntuplets(cells, foundNtuplets, tmpNtuplet,
                                       minHitsPerNtuplet);

        tmpNtuplet.pop_back();
      }
    }
  }

  template <int maxNumberOfQuadruplets>
  __host__ inline void find_ntuplets_host(
      const GPUCACell *cells,
      GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *foundNtuplets,
      GPUSimpleVector<3, unsigned int> &tmpNtuplet,
      const unsigned int minHitsPerNtuplet) const {

    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold
    Quadruplet tmpQuadruplet;
    if (tmpNtuplet.size() >= minHitsPerNtuplet - 1) {
      //                 if(tmpNtuplet.size()==3)
      //                 {
      //                     printf("\ntmpNtuplet contains %d:\n",
      //                     tmpNtuplet.size()); for(auto i = 0; i<
      //                     tmpNtuplet.size(); ++i)
      //                     {
      //                        printf("\t\t%d innerhit outerhit (xyz) (%f %f
      //                        %f), (%f %f %f)\n", tmpNtuplet.m_data[i],
      //                        cells[tmpNtuplet.m_data[i]].get_inner_x(),
      //                                 cells[tmpNtuplet.m_data[i]].get_inner_y(),
      //                                 cells[tmpNtuplet.m_data[i]].get_inner_z(),
      //                                 cells[tmpNtuplet.m_data[i]].get_outer_x(),
      //                                 cells[tmpNtuplet.m_data[i]].get_outer_y(),
      //                                 cells[tmpNtuplet.m_data[i]].get_outer_z());
      //
      //                     }
      //                 }

      for (int i = 0; i < minHitsPerNtuplet - 1; ++i) {
        tmpQuadruplet.layerPairsAndCellId[i].x =
            cells[tmpNtuplet.m_data[i]].theLayerPairId;
        tmpQuadruplet.layerPairsAndCellId[i].y = tmpNtuplet.m_data[i];
      }
      foundNtuplets->push_back(tmpQuadruplet);

    }

    else {

      for (int j = 0; j < theOuterNeighbors.size(); ++j) {

        auto otherCell = theOuterNeighbors.m_data[j];
        tmpNtuplet.push_back(otherCell);
        cells[otherCell].find_ntuplets_host(cells, foundNtuplets, tmpNtuplet,
                                            minHitsPerNtuplet);

        tmpNtuplet.pop_back();
      }
    }
  }
  GPUSimpleVector<40, unsigned int> theOuterNeighbors;

  int theDoubletId;
  int theLayerPairId;

private:
  unsigned int theInnerHitId;
  unsigned int theOuterHitId;
  const GPULayerDoublets *theDoublets;
  float theInnerX;
  float theOuterX;
  float theInnerY;
  float theOuterY;
  float theInnerZ;
  float theOuterZ;
  float theInnerR;
  float theOuterR;
};

#endif /*CACELL_H_ */
