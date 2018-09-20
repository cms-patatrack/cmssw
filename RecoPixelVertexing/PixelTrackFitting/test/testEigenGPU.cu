#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "test_common.h"

using namespace Eigen;

__global__
void kernelFullFit(Rfit::Matrix3xNd * hits,
				   Rfit::Matrix3Nd * hits_cov,
				   double B,
				   bool errors,
				   Rfit::circle_fit * circle_fit_resultsGPU,
				   Rfit::line_fit * line_fit_resultsGPU) {
	Vector4d fast_fit = Rfit::Fast_fit(*hits);
	
	u_int n = hits->cols();
	Rfit::VectorNd rad = (hits->block(0, 0, 2, n).colwise().norm());
	
	Rfit::Matrix2xNd hits2D_local = (hits->block(0,0,2,n)).eval();
	Rfit::Matrix2Nd hits_cov2D_local = (hits_cov->block(0, 0, 2 * n, 2 * n)).eval();
	Rfit::printIt(&hits2D_local, "kernelFullFit - hits2D_local: ");
	Rfit::printIt(&hits_cov2D_local, "kernelFullFit - hits_cov2D_local: ");
	/*
	 printf("kernelFullFit - hits address: %p\n", hits);
	 printf("kernelFullFit - hits_cov address: %p\n", hits_cov);
	 printf("kernelFullFit - hits_cov2D address: %p\n", &hits2D_local);
	 printf("kernelFullFit - hits_cov2D_local address: %p\n", &hits_cov2D_local);
	 */
	/* At some point I gave up and locally construct block on the stack, so that
	 the next invocation to Rfit::Circle_fit works properly. Failing to do so
	 implied basically an empty collection of hits and covariances. That could
	 have been partially fixed if values of the passed in matrices would have
	 been printed on screen since that, maybe, triggered internally the real
	 creations of the blocks. To be understood and compared against the myriad
	 of compilation warnings we have.
	 */
	(*circle_fit_resultsGPU) =
	Rfit::Circle_fit(hits->block(0,0,2,n), hits_cov->block(0, 0, 2 * n, 2 * n),
					 fast_fit, rad, B, errors);
	/*
	 (*circle_fit_resultsGPU) =
	 Rfit::Circle_fit(hits2D_local, hits_cov2D_local,
	 fast_fit, rad, B, errors, scattering);
	 */
	(*line_fit_resultsGPU) = Rfit::Line_fit(*hits, *hits_cov, *circle_fit_resultsGPU, fast_fit, errors);
	
	return;
}

__global__
void kernelFullBrokenLineFastFitAndData(BrokenLine::Matrix3xNd * hits,
										BrokenLine::Matrix3Nd * hits_cov,
										BrokenLine::PreparedBrokenLineData * data,
										Vector4d * fast_fit,
										double B,
										BrokenLine::helix_fit * helix_fit_resultsGPU,
										BrokenLine::karimaki_circle_fit * circleGPU,
										BrokenLine::line_fit * lineGPU,
										Matrix3d * JacobGPU,
										BrokenLine::MatrixNplusONEd * C_UGPU) {
	
	BrokenLine::helix_fit& helix = (*helix_fit_resultsGPU);
	
	helix.fast_fit=BrokenLine::BL_Fast_fit(*hits);
	
	BrokenLine::PrepareBrokenLineData(*hits,*hits_cov,helix.fast_fit,B,*data);
}

__global__
void kernelFullBrokenLineLine(BrokenLine::Matrix3xNd * hits,
							  BrokenLine::Matrix3Nd * hits_cov,
							  BrokenLine::PreparedBrokenLineData * data,
							  Vector4d * fast_fit,
							  double B,
							  BrokenLine::helix_fit * helix_fit_resultsGPU,
							  BrokenLine::karimaki_circle_fit * circleGPU,
							  BrokenLine::line_fit * lineGPU,
							  Matrix3d * JacobGPU,
							  BrokenLine::MatrixNplusONEd * C_UGPU) {
	
	BrokenLine::helix_fit& helix = (*helix_fit_resultsGPU);
	BrokenLine::line_fit& line = (*lineGPU);
	
	BrokenLine::BL_Line_fit(*hits,*hits_cov,helix.fast_fit,B,*data,line);
}

__global__
void kernelFullBrokenLineCircle(BrokenLine::Matrix3xNd * hits,
								BrokenLine::Matrix3Nd * hits_cov,
								BrokenLine::PreparedBrokenLineData * data,
								Vector4d * fast_fit,
								double B,
								BrokenLine::helix_fit * helix_fit_resultsGPU,
								BrokenLine::karimaki_circle_fit * circleGPU,
								BrokenLine::line_fit * lineGPU,
								Matrix3d * JacobGPU,
								BrokenLine::MatrixNplusONEd * C_UGPU) {
	
	BrokenLine::helix_fit& helix = (*helix_fit_resultsGPU);
	BrokenLine::karimaki_circle_fit& circle = (*circleGPU);
	
	BrokenLine::BL_Circle_fit(*hits,*hits_cov,helix.fast_fit,B,*data,circle,*JacobGPU,*C_UGPU);
}

__global__
void kernelFullBrokenLineHelix(BrokenLine::Matrix3xNd * hits,
							   BrokenLine::Matrix3Nd * hits_cov,
							   BrokenLine::PreparedBrokenLineData * data,
							   Vector4d * fast_fit,
							   double B,
							   BrokenLine::helix_fit * helix_fit_resultsGPU,
							   BrokenLine::karimaki_circle_fit * circleGPU,
							   BrokenLine::line_fit * lineGPU,
							   Matrix3d * JacobGPU,
							   BrokenLine::MatrixNplusONEd * C_UGPU) {
	
	BrokenLine::helix_fit& helix = (*helix_fit_resultsGPU);
	BrokenLine::karimaki_circle_fit& circle = (*circleGPU);
	BrokenLine::line_fit& line = (*lineGPU);
	
	// the circle fit gives k, but here we want p_t, so let's change the parameter and the covariance matrix
	Matrix3d& Jacob=(*JacobGPU);
	Jacob << 1,0,0,
	0,1,0,
	0,0,-abs(circle.par(2))*B/(BrokenLine::sqr(circle.par(2))*circle.par(2));
	circle.par(2)=B/abs(circle.par(2));
	circle.cov=Jacob*circle.cov*Jacob.transpose();
	
	helix.par << circle.par, line.par;
	helix.cov=MatrixXd::Zero(5, 5);
	helix.cov.block(0,0,3,3)=circle.cov;
	helix.cov.block(3,3,2,2)=line.cov;
	helix.q=circle.q;
	helix.chi2_circle=circle.chi2;
	helix.chi2_line=line.chi2;
	
	//(*helix_fit_resultsGPU) = BrokenLine::Helix_fit(*hits, *hits_cov, B);
}

__global__
void kernelFastFit(Rfit::Matrix3xNd * hits, Vector4d * results) {
	(*results) = Rfit::Fast_fit(*hits);
}

__global__
void kernelCircleFit(Rfit::Matrix3xNd * hits,
					 Rfit::Matrix3Nd * hits_cov, Vector4d * fast_fit_input, double B,
					 Rfit::circle_fit * circle_fit_resultsGPU) {
	u_int n = hits->cols();
	Rfit::VectorNd rad = (hits->block(0, 0, 2, n).colwise().norm());
	
#if TEST_DEBUG
	printf("fast_fit_input(0): %f\n", (*fast_fit_input)(0));
	printf("fast_fit_input(1): %f\n", (*fast_fit_input)(1));
	printf("fast_fit_input(2): %f\n", (*fast_fit_input)(2));
	printf("fast_fit_input(3): %f\n", (*fast_fit_input)(3));
	printf("rad(0,0): %f\n", rad(0,0));
	printf("rad(1,1): %f\n", rad(1,1));
	printf("rad(2,2): %f\n", rad(2,2));
	printf("hits_cov(0,0): %f\n", (*hits_cov)(0,0));
	printf("hits_cov(1,1): %f\n", (*hits_cov)(1,1));
	printf("hits_cov(2,2): %f\n", (*hits_cov)(2,2));
	printf("hits_cov(11,11): %f\n", (*hits_cov)(11,11));
	printf("B: %f\n", B);
#endif
	(*circle_fit_resultsGPU) =
	Rfit::Circle_fit(hits->block(0,0,2,n), hits_cov->block(0, 0, 2 * n, 2 * n),
					 *fast_fit_input, rad, B, false);
}

__global__
void kernelLineFit(Rfit::Matrix3xNd * hits,
				   Rfit::Matrix3Nd * hits_cov,
				   Rfit::circle_fit * circle_fit,
				   Vector4d * fast_fit,
				   Rfit::line_fit * line_fit)
{
	(*line_fit) = Rfit::Line_fit(*hits, *hits_cov, *circle_fit, *fast_fit, true);
}

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

void fillHitsAndHitsCovBrokenLine(BrokenLine::Matrix3xNd & hits, BrokenLine::Matrix3Nd & hits_cov) {
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
	Rfit::Matrix3xNd * hitsGPU = new Rfit::Matrix3xNd(3,4);
	Rfit::Matrix3Nd * hits_covGPU = nullptr;
	Vector4d * fast_fit_resultsGPU = new Vector4d();
	Vector4d * fast_fit_resultsGPUret = new Vector4d();
	Rfit::circle_fit * circle_fit_resultsGPU = new Rfit::circle_fit();
	Rfit::circle_fit * circle_fit_resultsGPUret = new Rfit::circle_fit();
	
	fillHitsAndHitsCov(hits, hits_cov);
	
	// FAST_FIT_CPU
	Vector4d fast_fit_results = Rfit::Fast_fit(hits);
#if TEST_DEBUG
	std::cout << "Generated hits:\n" << hits << std::endl;
#endif
	std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;
	
	// FAST_FIT GPU
	cudaMalloc((void**)&hitsGPU, sizeof(Rfit::Matrix3xNd(3,4)));
	cudaMalloc((void**)&fast_fit_resultsGPU, sizeof(Vector4d));
	cudaMemcpy(hitsGPU, &hits, sizeof(Rfit::Matrix3xNd(3,4)), cudaMemcpyHostToDevice);
	
	kernelFastFit<<<1, 1>>>(hitsGPU, fast_fit_resultsGPU);
	cudaDeviceSynchronize();
	
	cudaMemcpy(fast_fit_resultsGPUret, fast_fit_resultsGPU, sizeof(Vector4d), cudaMemcpyDeviceToHost);
	std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]): GPU\n" << *fast_fit_resultsGPUret << std::endl;
	assert(isEqualFuzzy(fast_fit_results, (*fast_fit_resultsGPUret)));
	
	// CIRCLE_FIT CPU
	u_int n = hits.cols();
	Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());
	
	Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, n),
														   hits_cov.block(0, 0, 2 * n, 2 * n),
														   fast_fit_results, rad, B, false);
	std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << std::endl;
	
	// CIRCLE_FIT GPU
	cudaMalloc((void **)&hits_covGPU, sizeof(Rfit::Matrix3Nd(12,12)));
	cudaMalloc((void **)&circle_fit_resultsGPU, sizeof(Rfit::circle_fit));
	cudaMemcpy(hits_covGPU, &hits_cov, sizeof(Rfit::Matrix3Nd(12,12)), cudaMemcpyHostToDevice);
	
	kernelCircleFit<<<1,1>>>(hitsGPU, hits_covGPU,
							 fast_fit_resultsGPU, B, circle_fit_resultsGPU);
	cudaDeviceSynchronize();
	
	cudaMemcpy(circle_fit_resultsGPUret, circle_fit_resultsGPU,
			   sizeof(Rfit::circle_fit), cudaMemcpyDeviceToHost);
	std::cout << "Fitted values (CircleFit) GPU:\n" << circle_fit_resultsGPUret->par << std::endl;
	assert(isEqualFuzzy(circle_fit_results.par, circle_fit_resultsGPUret->par));
	
	// LINE_FIT CPU
	Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_cov, circle_fit_results, fast_fit_results, true);
	std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << std::endl;
	
	// LINE_FIT GPU
	Rfit::line_fit * line_fit_resultsGPU = nullptr;
	Rfit::line_fit * line_fit_resultsGPUret = new Rfit::line_fit();
	
	cudaMalloc((void **)&line_fit_resultsGPU, sizeof(Rfit::line_fit));
	
	kernelLineFit<<<1,1>>>(hitsGPU, hits_covGPU, circle_fit_resultsGPU, fast_fit_resultsGPU, line_fit_resultsGPU);
	cudaDeviceSynchronize();
	
	cudaMemcpy(line_fit_resultsGPUret, line_fit_resultsGPU, sizeof(Rfit::line_fit), cudaMemcpyDeviceToHost);
	std::cout << "Fitted values (LineFit) GPU:\n" << line_fit_resultsGPUret->par << std::endl;
	assert(isEqualFuzzy(line_fit_results.par, line_fit_resultsGPUret->par));
}

void testFitOneGo(bool errors, double epsilon=1e-6) {
	constexpr double B = 0.0113921;
	Rfit::Matrix3xNd hits(3,4);
	Rfit::Matrix3Nd hits_cov = MatrixXd::Zero(12,12);
	
	fillHitsAndHitsCov(hits, hits_cov);
	
	// FAST_FIT_CPU
	Vector4d fast_fit_results = Rfit::Fast_fit(hits);
	// CIRCLE_FIT CPU
	u_int n = hits.cols();
	Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());
	
	Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, n),
														   hits_cov.block(0, 0, 2 * n, 2 * n),
														   fast_fit_results, rad, B, errors);
	// LINE_FIT CPU
	Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_cov, circle_fit_results,
													 fast_fit_results, errors);
	
	// FIT GPU
	std::cout << "GPU FIT" << std::endl;
	Rfit::Matrix3xNd * hitsGPU = nullptr; // new Rfit::Matrix3xNd(3,4);
	Rfit::Matrix3Nd * hits_covGPU = nullptr;
	Rfit::line_fit * line_fit_resultsGPU = nullptr;
	Rfit::line_fit * line_fit_resultsGPUret = new Rfit::line_fit();
	Rfit::circle_fit * circle_fit_resultsGPU = nullptr; // new Rfit::circle_fit();
	Rfit::circle_fit * circle_fit_resultsGPUret = new Rfit::circle_fit();
	
	cudaCheck(cudaMalloc((void **)&hitsGPU, sizeof(Rfit::Matrix3xNd(3,4))));
	cudaCheck(cudaMalloc((void **)&hits_covGPU, sizeof(Rfit::Matrix3Nd(12,12))));
	cudaCheck(cudaMalloc((void **)&line_fit_resultsGPU, sizeof(Rfit::line_fit)));
	cudaCheck(cudaMalloc((void **)&circle_fit_resultsGPU, sizeof(Rfit::circle_fit)));
	cudaCheck(cudaMemcpy(hitsGPU, &hits, sizeof(Rfit::Matrix3xNd(3,4)), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(hits_covGPU, &hits_cov, sizeof(Rfit::Matrix3Nd(12,12)), cudaMemcpyHostToDevice));
	
	kernelFullFit<<<1, 1>>>(hitsGPU, hits_covGPU, B, errors,
							circle_fit_resultsGPU, line_fit_resultsGPU);
	cudaCheck(cudaDeviceSynchronize());
	
	cudaCheck(cudaMemcpy(circle_fit_resultsGPUret, circle_fit_resultsGPU, sizeof(Rfit::circle_fit), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(line_fit_resultsGPUret, line_fit_resultsGPU, sizeof(Rfit::line_fit), cudaMemcpyDeviceToHost));
	
	std::cout << "Fitted values (CircleFit) CPU:\n" << circle_fit_results.par << std::endl;
	std::cout << "Fitted values (LineFit): CPU\n" << line_fit_results.par << std::endl;
	std::cout << "Fitted values (CircleFit) GPU:\n" << circle_fit_resultsGPUret->par << std::endl;
	std::cout << "Fitted values (LineFit): GPU\n" << line_fit_resultsGPUret->par << std::endl;
	assert(isEqualFuzzy(circle_fit_results.par, circle_fit_resultsGPUret->par, epsilon));
	assert(isEqualFuzzy(line_fit_results.par, line_fit_resultsGPUret->par, epsilon));
	
	cudaCheck(cudaFree(hitsGPU));
	cudaCheck(cudaFree(hits_covGPU));
	cudaCheck(cudaFree(line_fit_resultsGPU));
	cudaCheck(cudaFree(circle_fit_resultsGPU));
	delete line_fit_resultsGPUret;
	delete circle_fit_resultsGPUret;
	
	cudaDeviceReset();
}

void testBrokenLineOneGo(double epsilon=1e-6) {
	constexpr double B = 0.0113921;
	BrokenLine::Matrix3xNd hits(3,4);
	BrokenLine::Matrix3Nd hits_cov = MatrixXd::Zero(12,12);
	
	fillHitsAndHitsCovBrokenLine(hits, hits_cov);
	
	// HELIX_FIT CPU
	BrokenLine::helix_fit helix_fit_results = BrokenLine::Helix_fit(hits,hits_cov,B);
	
	std::cout << "Fitted values (HelixFit) CPU:\n" << helix_fit_results.par << std::endl;
	
	// FIT GPU
	std::cout << "GPU FIT" << std::endl;
	BrokenLine::Matrix3xNd * hitsGPU = nullptr;
	BrokenLine::Matrix3Nd * hits_covGPU = nullptr;
	BrokenLine::PreparedBrokenLineData * dataGPU = nullptr;
	Vector4d * fast_fitGPU = nullptr;
	BrokenLine::karimaki_circle_fit * circleGPU = nullptr;
	BrokenLine::line_fit * lineGPU = nullptr;
	Matrix3d * JacobGPU = nullptr;
	BrokenLine::MatrixNplusONEd * C_UGPU = nullptr;
	BrokenLine::helix_fit * helix_fit_resultsGPU = nullptr;
	BrokenLine::helix_fit * helix_fit_resultsGPUret = new BrokenLine::helix_fit();
	
	cudaCheck(cudaMalloc((void **)&hitsGPU, sizeof(BrokenLine::Matrix3xNd(3,4))));
	cudaCheck(cudaMalloc((void **)&hits_covGPU, sizeof(BrokenLine::Matrix3Nd(12,12))));
	cudaCheck(cudaMalloc((void **)&dataGPU, sizeof(BrokenLine::PreparedBrokenLineData)));
	cudaCheck(cudaMalloc((void **)&fast_fitGPU, sizeof(Vector4d)));
	cudaCheck(cudaMalloc((void **)&circleGPU, sizeof(BrokenLine::karimaki_circle_fit)));
	cudaCheck(cudaMalloc((void **)&lineGPU, sizeof(BrokenLine::line_fit)));
	cudaCheck(cudaMalloc((void **)&JacobGPU, sizeof(Matrix3d)));
	cudaCheck(cudaMalloc((void **)&C_UGPU, sizeof(BrokenLine::MatrixNplusONEd)));
	cudaCheck(cudaMalloc((void **)&helix_fit_resultsGPU, sizeof(BrokenLine::helix_fit)));
	
	cudaCheck(cudaMemcpy(hitsGPU, &hits, sizeof(BrokenLine::Matrix3xNd(3,4)), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(hits_covGPU, &hits_cov, sizeof(BrokenLine::Matrix3Nd(12,12)), cudaMemcpyHostToDevice));
	
	/*
	 IMPORTANT: if compiling with "-g -G" or using dinamically-sized matrices, you need to increase the stack size.
	 You can comment the following two lines when using static-sized-matrices.
	 1761 bytes is the minimum value that makes it work with dinamically-sized matrices.
	 */
	cudaDeviceSetLimit(cudaLimitStackSize, 1761);
	cudaCheck(cudaDeviceSynchronize());
	
	kernelFullBrokenLineFastFitAndData<<<1, 1>>>(hitsGPU, hits_covGPU, dataGPU, fast_fitGPU, B, helix_fit_resultsGPU, circleGPU, lineGPU, JacobGPU, C_UGPU);
	cudaCheck(cudaDeviceSynchronize());
	kernelFullBrokenLineLine<<<1, 1>>>(hitsGPU, hits_covGPU, dataGPU, fast_fitGPU, B, helix_fit_resultsGPU, circleGPU, lineGPU, JacobGPU, C_UGPU);
	cudaCheck(cudaDeviceSynchronize());
	kernelFullBrokenLineCircle<<<1, 1>>>(hitsGPU, hits_covGPU, dataGPU, fast_fitGPU, B, helix_fit_resultsGPU, circleGPU, lineGPU, JacobGPU, C_UGPU);
	cudaCheck(cudaDeviceSynchronize());
	kernelFullBrokenLineHelix<<<1, 1>>>(hitsGPU, hits_covGPU, dataGPU, fast_fitGPU, B, helix_fit_resultsGPU, circleGPU, lineGPU, JacobGPU, C_UGPU);
	cudaCheck(cudaDeviceSynchronize());
	
	cudaCheck(cudaMemcpy(helix_fit_resultsGPUret, helix_fit_resultsGPU, sizeof(BrokenLine::helix_fit), cudaMemcpyDeviceToHost));
	
	std::cout << "Fitted values (HelixFit) GPU:\n" << helix_fit_resultsGPUret->par << std::endl;
	assert(isEqualFuzzy(helix_fit_results.par, helix_fit_resultsGPUret->par, epsilon));
	
	cudaCheck(cudaFree(hitsGPU));
	cudaCheck(cudaFree(hits_covGPU));
	cudaCheck(cudaFree(helix_fit_resultsGPU));
	cudaCheck(cudaFree(dataGPU));
	cudaCheck(cudaFree(fast_fitGPU));
	cudaCheck(cudaFree(circleGPU));
	cudaCheck(cudaFree(lineGPU));
	cudaCheck(cudaFree(JacobGPU));
	cudaCheck(cudaFree(C_UGPU));
	delete helix_fit_resultsGPUret;
	
	cudaDeviceReset();
}

int main (int argc, char * argv[]) {
	//  testFit();
	/*std::cout << "TEST FIT, NO ERRORS" << std::endl;
	 testFitOneGo(false);
	 
	 std::cout << "TEST FIT, ERRORS AND SCATTER" << std::endl;
	 testFitOneGo(true, 1e-5);*/
	
	std::cout << "TEST BROKEN LINE" << std::endl;
	testBrokenLineOneGo(1e-5);
	
	return 0;
}

