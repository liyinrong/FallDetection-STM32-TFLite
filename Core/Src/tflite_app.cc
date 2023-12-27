/*
 * tflite_app.cc
 *
 *  Created on: Dec 25, 2023
 *      Author: liyin
 */
#include <stdio.h>
#include <stdint.h>
#include "main.h"
#include "tflite_app.h"
#include "gen_micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TFLITE_SCHEMA_VERSION 	(3)
#define MODEL_DATA				&TinyFallNet_6axis_qat_FullInt_Rescaled_tflite[0]
#define TENSOR_ARENA_SIZE		32768

extern const unsigned char TinyFallNet_6axis_qat_tflite[];
extern const unsigned char TinyFallNet_6axis_qat_FullInt_Rescaled_tflite[];
extern const unsigned char TinyFallNet_6axis_8bitInput_qat_tflite[];
extern const unsigned char ResNet24_6axis_qat_tflite[];

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
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

void error_handler(void)
{
	__disable_irq();
	while (1)
	{
	}
}

void TFLite_Init(void)
{
	printf("TFLite initializing.\r\n");
	model = tflite::GetModel(MODEL_DATA);
	if(model->version() != TFLITE_SCHEMA_VERSION)
	{
	    printf("Invalid expected TFLite model version %d instead %d\r\n",
	        (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
	    error_handler();
	}
	static tflite::MicroMutableOpResolver<kNumberOperators> _resolver = get_resolver();
    static tflite::MicroInterpreter _interpreter(model, _resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr, nullptr, false);
	interpreter = &_interpreter;
    if(interpreter->AllocateTensors() != kTfLiteOk)
    {
    	printf("Failed to allocate tensors.\r\n");
    	error_handler();
    }
    input = interpreter->input(0);
    output = interpreter->output(0);
    printf("TFLite initialized.\r\n");
}

extern uint8_t NewDataFetched;
extern uint8_t FallDetected;

#ifdef FLOAT_DATA
extern float RecvBuffer[1][50][6];
#else
extern uint8_t RecvBuffer[1][50][6];
#endif
extern uint8_t RecvBufferPTR;

extern void DWT_Start(void);
extern uint32_t DWT_Stop(void);

void pre_process(void* data)
{
	#if not defined(FLOAT_MODEL_INPUT)
	int8_t* ptr = (int8_t*)data;
	TfLiteAffineQuantization *quant = (TfLiteAffineQuantization *)input->quantization.params;
	float scale = *(quant->scale->data);
	int zero_point = *(quant->zero_point->data);
	for(uint8_t i=RecvBufferPTR; i<50; i++)
	{
		for(uint8_t j=0; j<6; j++)
		{
			*(ptr++) = (int8_t)(RecvBuffer[0][i][j] / scale + zero_point);
		}
	}
	for(uint8_t i=0; i<RecvBufferPTR; i++)
	{
		for(uint8_t j=0; j<6; j++)
		{
			*(ptr++) = (int8_t)(RecvBuffer[0][i][j] / scale + zero_point);
		}
	}
	#else
	memcpy((uint8_t*)data, (uint8_t*)(&RecvBuffer[0][RecvBufferPTR][0]), (50-RecvBufferPTR)*(6*sizeof(RecvBuffer[0][0][0])));
	memcpy((uint8_t*)data+(50-RecvBufferPTR)*(6*sizeof(RecvBuffer[0][0][0])), (uint8_t*)RecvBuffer, RecvBufferPTR*(6*sizeof(RecvBuffer[0][0][0])));
	#endif
}

void post_process(void* data)
{
//	printf("output[0]=%d output[1]=%d\r\n", *(int8_t*)data, *((int8_t*)data+1));
}

void TFLite_Process(void)
{
	volatile int res = 0;
	uint32_t InferenceTime;
	uint8_t *in_data = NULL;
	uint8_t *out_data = NULL;

	if(NewDataFetched)
	{
	    in_data = (uint8_t *)(input->data.uint8);
	    out_data = (uint8_t *)(output->data.uint8);

		pre_process(in_data);
//		printf("TFLite inference start.\r\n");
		DWT_Start();
		if (interpreter->Invoke() != kTfLiteOk) {
			res = -1;
		}
		InferenceTime = DWT_Stop();
		if(res)
		{
			printf("Inference failed, code %d.\r\n", res);
			error_handler();
		}
//		printf("TFLite inference complete, elapsed time: %luus.\r\n", InferenceTime);
		post_process(out_data);
		#ifdef FLOAT_MODEL_OUTPUT
		printf("Inference completed, output=[%f, %f], elapsed time: %luus.\r\n", *(float*)out_data, *((float*)out_data+1), InferenceTime);
		#else
		printf("Inference completed, output=[%d, %d], elapsed time: %luus.\r\n", *(int8_t*)out_data, *((int8_t*)out_data+1), InferenceTime);
		#endif
		NewDataFetched = 0U;
	}
}

#ifdef __cplusplus
}
#endif
