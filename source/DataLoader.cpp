#include "DataLoader.h"

#include <iostream>
#include "COO.h"
#include "cusparse/include/cuSparseMultiply.h"

template<typename T>
std::string typeExtension();
template<>
std::string typeExtension<float>()
{
	return std::string("");
}
template<>
std::string typeExtension<double>()
{
	return std::string("d_");
}

template class DataLoader<float>;
template class DataLoader<double>;

template <typename ValueType>
DataLoader<ValueType>::DataLoader(std::string path) : matrices()
{
	std::string csrPath_A = "../../A_big.hicsr"; 
	std::string csrPath_B = "../../B_big.hicsr";

	try
	{
		std::cout << "trying to load csr file \"" << csrPath_A << "\"\n";
		matrices.cpuA = loadCSR<ValueType>(csrPath_A.c_str());
		std::cout << "successfully loaded: \"" << csrPath_A << "\"\n";

		std::cout << "trying to load csr file \"" << csrPath_B << "\"\n";
		matrices.cpuB = loadCSR<ValueType>(csrPath_B.c_str());
		std::cout << "successfully loaded: \"" << csrPath_B << "\"\n";

	}
	catch (std::exception& ex)
	{
		std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
	}
	
	convert(matrices.gpuA, matrices.cpuA, 0);
	convert(matrices.gpuB, matrices.cpuB, 0);
}