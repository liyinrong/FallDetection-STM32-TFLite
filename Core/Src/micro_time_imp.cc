/*
 * micro_time_imp.cc
 *
 *  Created on: Dec 29, 2023
 *      Author: liyin
 */

#include "tensorflow/lite/micro/micro_time.h"
#include "stm32l4xx_hal.h"

namespace tflite {

uint32_t ticks_per_second()
{
	return SystemCoreClock;
}

uint32_t GetCurrentTimeTicks()
{
	return DWT->CYCCNT;
}

}  // namespace tflite
