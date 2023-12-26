/*
 * debug_log_imp.cc
 *
 *  Created on: Dec 25, 2023
 *      Author: liyin
 */

#include "tensorflow/lite/micro/debug_log.h"

#include <stdio.h>
#include <stdarg.h>

extern "C" void DebugLog(const char* format, va_list args)
{
	vprintf(format, args);
}

extern "C" int DebugVsnprintf(char* buffer, size_t buf_size, const char* format, va_list vlist)
{
	return vsnprintf(buffer, buf_size, format, vlist);
}
