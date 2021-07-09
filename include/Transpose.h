#pragma once

#include "dCSR_sp.h"

using namespace spECKWrapper;
namespace spECK {
	template <typename DataType>
	void Transpose(const dCSR<DataType>& matIn, dCSR<DataType>& matTransposeOut);
}