#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace Eigen;

__host__ __device__ void eigenValues(Matrix3d * m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType * ret) {
  printf("Matrix(0,0): %f\n", (*m)(0,0));
  printf("Matrix(1,1): %f\n", (*m)(1,1));
  printf("Matrix(2,2): %f\n", (*m)(2,2));
  SelfAdjointEigenSolver<Matrix3d> es;
  es.computeDirect(*m);
  (*ret) = es.eigenvalues();
  return;
}

__global__ void kernel(Matrix3d * m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType * ret) {
  eigenValues(m, ret);
}

bool isEqualFuzzy(double a, double b) {
  constexpr double epsilon = 1e-6;
  return std::abs(a-b) < std::min(std::abs(a), std::abs(b))*epsilon;
}

int main (int argc, char * argv[]) {
  
  Matrix3d m = Matrix3d::Random();
  Matrix3d mt = m.transpose();
  m += mt;
  Matrix3d * m_gpu;
  Matrix3d * mgpudebug = new Matrix3d();
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret = new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret1 = new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret_gpu;
  eigenValues(&m, ret);
  std::cout << "Generated Matrix M 3x3:\n" << m << std::endl;
  std::cout << "The eigenvalues of M are:" << std::endl << (*ret) << std::endl;
  std::cout << "*************************\n\n" << std::endl;

  cudaMalloc((void **)&m_gpu, sizeof(Matrix3d));
  cudaMalloc((void **)&ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType));
  cudaMemcpy(m_gpu, &m, sizeof(Matrix3d), cudaMemcpyHostToDevice);

  kernel<<<1,1>>>(m_gpu, ret_gpu);

  cudaDeviceSynchronize();

  cudaMemcpy(mgpudebug, m_gpu, sizeof(Matrix3d), cudaMemcpyDeviceToHost);
  cudaMemcpy(ret1, ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType), cudaMemcpyDeviceToHost);
  std::cout << "GPU Generated Matrix M 3x3:\n" << (*mgpudebug) << std::endl;
  std::cout << "GPU The eigenvalues of M are:" << std::endl << (*ret1) << std::endl;
  std::cout << "*************************\n\n" << std::endl;

  std::cout << "Ratio: " << (*ret)(0,0)/(*ret1)(0,0) << std::endl;
  std::cout << "Ratio: " << (*ret)(1,0)/(*ret1)(1,0) << std::endl;
  std::cout << "Ratio: " << (*ret)(2,0)/(*ret1)(2,0) << std::endl;
  assert(isEqualFuzzy((*ret)(0,0), (*ret1)(0,0)));
  assert(isEqualFuzzy((*ret)(1,0), (*ret1)(1,0)));
  assert(isEqualFuzzy((*ret)(2,0), (*ret1)(2,0)));
  return 0;
}
