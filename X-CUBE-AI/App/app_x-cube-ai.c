
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Basic template to show how to use the TensorFlow lite micro API
  *
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"

/* USER CODE BEGIN includes */
/* USER CODE END includes */

#include <tflm_c.h>
/* Global handle - used to reference the instantiated model */
static uint32_t model_hdl = 0;

/* tflm_io_write() is the final callback implementation to implement the DebugLog() function
 * requested by the tflite::MicroErrorReporter object
 */
int tflm_io_write(const void *buff, uint16_t count)
{

    /*
     * add here where to send the log messages
     * for example
     *   HAL_StatusTypeDef status;
     *   status = HAL_UART_Transmit(&UartHandle, (uint8_t *)buff, count,
     *       HAL_MAX_DELAY);
     *
     * return (status == HAL_OK ? count : 0);
     */
     return((int)count);
}

static int ai_boostrap(const uint8_t *model, uint8_t *arena_addr,
    size_t arena_sz)
{
  TfLiteStatus res;
  int32_t size_io;

  /* USER CODE BEGIN 1 */
  printf("\r\nInstancing the network (TFLM)..\r\n");
  /* USER CODE END 1 */

  res = tflm_c_create(model, (uint8_t*)arena_addr, arena_sz, &model_hdl);

  if (res != kTfLiteOk) {
    return -1;
  }

  /* USER CODE BEGIN 2 */

  /* Report the main model info */

  printf(" Operator size      : %d\r\n", (int)tflm_c_operators_size(model_hdl));
  printf(" Tensor size        : %d\r\n", (int)tflm_c_tensors_size(model_hdl));

  /* Report the size of arena buffer which is really used during the set-up and run phases
   * - based on a debug service (see C++ interpreter i/f in tflm_c.cc file)
   * - after this step, no additional memory should be allocated from the arena buffer or
   *   through another system heap allocator.
   * Note: This info is useful to refine/adjust the requested size of the arena buffer.
   */
  printf(" Allocated size     : %d / %d\r\n", (int)tflm_c_arena_used_bytes(model_hdl),
      (int)arena_sz);

  /* Report the description of the IO tensors */

  size_io = tflm_c_inputs_size(model_hdl);
  printf(" Inputs size        : %d\r\n", (int)size_io);

  for (int i=0; i<size_io; i++) {
    struct tflm_c_tensor_info t_info;
    tflm_c_input(model_hdl, i, &t_info);
    printf("  %d: %s (%d bytes) (%d, %d, %d)", i, tflm_c_TfLiteTypeGetName(t_info.type),
        (int)t_info.bytes, (int)t_info.height, (int)t_info.width, (int)t_info.channels);
    if (t_info.scale)
      printf(" scale=%f, zp=%d\r\n", (float)t_info.scale, (int)t_info.zero_point);
    else
      printf("\r\n");
  }

  size_io = tflm_c_outputs_size(model_hdl);
  printf(" Outputs size       : %d\r\n", (int)size_io);

  for (int i=0; i<size_io; i++) {
    struct tflm_c_tensor_info t_info;
    tflm_c_output(model_hdl, i, &t_info);
    printf("  %d: %s (%d bytes) (%d, %d, %d)", i, tflm_c_TfLiteTypeGetName(t_info.type),
        (int)t_info.bytes, (int)t_info.height, (int)t_info.width, (int)t_info.channels);
    if (t_info.scale)
      printf(" scale=%f, zp=%d\r\n", (float)t_info.scale, (int)t_info.zero_point);
    else
      printf("\r\n");
  }

  if ((tflm_c_inputs_size(model_hdl) != 1) || (tflm_c_inputs_size(model_hdl) != 1)) {
    printf("WARNING - embedded TFL model is not compitable with the default template..\r\n");
  }

  /* USER CODE END 2 */

  return 0;
}

/* USER CODE BEGIN 3 */
extern uint8_t NewDataFetched;
extern float RecvBuffer[1][50][6];
extern uint8_t RecvBufferPTR;
extern uint8_t FallDetected;
extern void DWT_Start(void);
extern uint32_t DWT_Stop(void);

void pre_process(void* data)
{
	memcpy(data, (uint8_t*)(RecvBuffer+RecvBufferPTR), (50-RecvBufferPTR)*sizeof(float));
	memcpy(data+(50-RecvBufferPTR)*sizeof(float), (uint8_t*)RecvBuffer, RecvBufferPTR*sizeof(float));
}

void post_process(void* data)
{
	printf("output[0]=%d output[1]=%d\r\n", *(int8_t*)data, *((int8_t*)data+1));
}

void error_handler(void)
{
	__disable_irq();
	while (1)
	{
	}
}
/* USER CODE END 3 */

/* USER CODE BEGIN 4 */

/*
 * The following code is based on the generated/specific
 * network_tflite_data.h/.c files. These files embed a full
 * image of the TFLite file as a C-array
 * (g_tflm_network_model_data[]).
 *
 * Note: Thanks to X-CUBE-AI, a pre-calculated ARENA size is
 *       also provided (TFLM_NETWORK_TENSOR_AREA_SIZE).
 *       With the default TFLight micro environment, no
 *       service is available to report it. Recommended approach
 *       is to provide an initial size (roughly estimated by the user)
 *       and to adjust it during the integration phase. Real value
 *       can be only known after the call of the
 *       interpreter::AllocateTensors() function at runtime.
 */

#include "network_tflite_data.h"

#define BIN_ADDRESS &g_tflm_network_model_data[0]
#define ARENA_SIZE  TFLM_NETWORK_TENSOR_AREA_SIZE

/* Allocate the arena buffer to install the model
 *  - Size should be aligned on 16-bytes.
 *   Note: This is not really strict, but avoid to have a
 *         specific warning/debug message during the
 *         set-up phase. At the end, memory will be aligned by
 *          the TFLight micro runtime it-self.
 */

#define _CONCAT_ARG(a, b)     a ## b
#define _CONCAT(a, b)         _CONCAT_ARG(a, b)

#if defined(_MSC_VER)
  #define MEM_ALIGNED(x)
#elif defined(__ICCARM__) || defined (__IAR_SYSTEMS_ICC__)
  #define MEM_ALIGNED(x)         _CONCAT(MEM_ALIGNED_,x)
  #define MEM_ALIGNED_16         _Pragma("data_alignment = 16")
#elif defined(__CC_ARM)
  #define MEM_ALIGNED(x)         __attribute__((aligned (x)))
#elif defined(__GNUC__)
  #define MEM_ALIGNED(x)         __attribute__((aligned(x)))
#else
  #define MEM_ALIGNED(x)
#endif

MEM_ALIGNED(16)
static uint8_t tensor_arena[TFLM_NETWORK_TENSOR_AREA_SIZE];
/* USER CODE END 4 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
	int res = 0;

	printf("TFLite initializing.\r\n");

	res = ai_boostrap(BIN_ADDRESS, tensor_arena, ARENA_SIZE);

	if(res)
	{
		printf("TFLite initialization failed, code %d.\r\n", res);
		error_handler();
	}
	printf("TFLite initialized.\r\n");

    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
	volatile int res = 0;
	uint32_t InferenceTime;
	struct tflm_c_tensor_info info;
	uint8_t *in_data = NULL;
	uint8_t *out_data = NULL;

	if(NewDataFetched)
	{
	    tflm_c_input(model_hdl, 0, &info);
	    in_data = (uint8_t *)info.data;
	    tflm_c_output(model_hdl, 0, &info);
	    out_data = (uint8_t *)info.data;

		pre_process(in_data);
		printf("TFLite inference start.\r\n");
		DWT_Start();
		if (tflm_c_invoke(model_hdl) != kTfLiteOk) {
			res = -1;
		}
		InferenceTime = DWT_Stop();
		if(res)
		{
			printf("TFLite inference failed, code %d.\r\n", res);
			error_handler();
		}
		printf("TFLite inference complete, elapsed time: %luns.\r\n", InferenceTime);
		post_process(out_data);
		NewDataFetched = 0U;
	}

    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
