/*
 * tflite_config.h
 *
 *  Created on: Dec 27, 2023
 *      Author: liyin
 */

#ifndef INC_TFLITE_CONFIG_H_
#define INC_TFLITE_CONFIG_H_

#define FULL_INT_MODEL
//#define FLOAT_INPUT
//#define FLOAT_OUTPUT
#define TFLITE_SCHEMA_VERSION (3)
#define MODEL_DATA			&TinyFallNet_6axis_qat_FInt_tflite[0]
#define TENSOR_ARENA_SIZE	32768

extern const unsigned char TinyFallNet_6axis_qat_tflite[];
extern const unsigned char TinyFallNet_6axis_qat_FInt_tflite[];
extern const unsigned char ResNet24_6axis_qat_tflite[];

#endif /* INC_TFLITE_CONFIG_H_ */
