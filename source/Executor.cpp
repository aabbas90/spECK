#include "cuda_runtime.h"
#include "Executor.h"
#include "Multiply.h"
#include "DataLoader.h"
#include <iomanip>
#include "Config.h"
#include "Compare.h"
#include <cusparse/include/cuSparseMultiply.h>
#include "Timings.h"
#include "spECKConfig.h"

template <typename ValueType>
int Executor<ValueType>::run()
{
	iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 5);
	iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	DataLoader<ValueType> data(runConfig.filePath);
	auto& matrices = data.matrices;
	std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";

	// std::cout<< "\n A \n";
	// print(matrices.cpuA);

	// std::cout<< "\n B \n";
	// print(matrices.cpuB);

	CSR<float> res;

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	bool measureAll = Config::getBool(Config::TrackIndividualTimes, false);
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	bool compareData = true;

	if(Config::getBool(Config::CompareResult) || true)
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);
		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}

	// Warmup iterations for multiplication
	for (int i = 0; i < iterationsWarmup; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		warmupTimings += timings;
		convert(res, dCsrHiRes);
		std::cout<< "\n res \n";
		// print(res);

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr)
		{
			if (!spECK::Compare(dCsrReference, dCsrHiRes, true))
				printf("Error: Matrix incorrect\n");
		}
	}

	// Multiplication
	for (int i = 0; i < iterationsExecution; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		benchTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr)
		{
			if (!spECK::Compare(dCsrReference, dCsrHiRes, true))
				printf("Error: Matrix incorrect\n");
		}
	}
	
	benchTimings /= iterationsExecution;


	std::cout << std::setw(20) << "var-SpGEMM -> NNZ: " << dCsrHiRes.nnz << std::endl;
	std::cout << std::setw(20) << "var-SpGEMM SpGEMM: " << benchTimings.complete << " ms" << std::endl;

	return 0;
}

 // template int Executor<float>::run();
template int Executor<float>::run();
