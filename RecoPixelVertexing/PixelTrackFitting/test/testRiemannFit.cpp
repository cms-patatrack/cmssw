#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "test_common.h"

using namespace Eigen;

void fillHitsAndHitsCov(Rfit::Matrix3xNd & hits, Rfit::Matrix3Nd & hits_cov) {
  hits << 1.98645, 4.72598, 7.65632, 11.3151,
          2.18002, 4.88864, 7.75845, 11.3134,
          2.46338, 6.99838,  11.808,  17.793;
  hits_cov(0,0) = 7.14652e-06;
  hits_cov(1,1) = 2.15789e-06;
  hits_cov(2,2) = 1.63328e-06;
  hits_cov(3,3) = 6.27919e-06;
  hits_cov(4,4) = 6.10348e-06;
  hits_cov(5,5) = 2.08211e-06;
  hits_cov(6,6) = 1.61672e-06;
  hits_cov(7,7) = 6.28081e-06;
  hits_cov(8,8) = 5.184e-05;
  hits_cov(9,9) = 1.444e-05;
  hits_cov(10,10) = 6.25e-06;
  hits_cov(11,11) = 3.136e-05;
  hits_cov(0,4) = hits_cov(4,0) = -5.60077e-06;
  hits_cov(1,5) = hits_cov(5,1) = -1.11936e-06;
  hits_cov(2,6) = hits_cov(6,2) = -6.24945e-07;
  hits_cov(3,7) = hits_cov(7,3) = -5.28e-06;
}

void testFit() {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd hits(3,4);
  Rfit::Matrix3Nd hits_cov = MatrixXd::Zero(12,12);

  fillHitsAndHitsCov(hits, hits_cov);
  std::cout << "Generated hits:\n" << hits << std::endl;
  

  // FAST_FIT_CPU
  Vector4d fast_fit_results = Rfit::Fast_fit(hits);

  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;


  // CIRCLE_FIT CPU
  u_int n = hits.cols();
  Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());

  Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, n),
      hits_cov.block(0, 0, 2 * n, 2 * n),
      fast_fit_results, rad, B, false);
  std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << std::endl;

  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_cov, circle_fit_results, fast_fit_results, true);
  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << std::endl;

}

int main (int argc, char * argv[]) {
  testFit();
  return 0;
}

