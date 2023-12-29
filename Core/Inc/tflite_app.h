/*
 * tflite_app.h
 *
 *  Created on: Dec 26, 2023
 *      Author: liyin
 */

#ifndef INC_TFLITE_APP_H_
#define INC_TFLITE_APP_H_

#ifdef __cplusplus
extern "C" {
#endif

//#define FLOAT_DATA
//#define FLOAT_MODEL_INPUT
//#define FLOAT_MODEL_OUTPUT
//#define PROFILING

void TFLite_Init(void);
void TFLite_Process(void);

#ifdef __cplusplus
}
#endif

#endif /* INC_TFLITE_APP_H_ */
