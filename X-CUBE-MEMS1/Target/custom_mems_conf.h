/**
  ******************************************************************************
  * @file    custom_mems_conf.h
  * @author  MEMS Software Solutions Team
  * @brief   This file contains definitions of the MEMS components bus interfaces for custom boards
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef CUSTOM_MEMS_CONF_H
#define CUSTOM_MEMS_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32l4xx_hal.h"
#include "b_l475e_iot01a2_bus.h"
#include "b_l475e_iot01a2_errno.h"

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

#define USE_CUSTOM_MOTION_SENSOR_LSM6DSL_0        1U

#define USE_CUSTOM_MOTION_SENSOR_LIS3MDL_0        1U

#define CUSTOM_LSM6DSL_0_I2C_Init BSP_I2C2_Init
#define CUSTOM_LSM6DSL_0_I2C_DeInit BSP_I2C2_DeInit
#define CUSTOM_LSM6DSL_0_I2C_ReadReg BSP_I2C2_ReadReg
#define CUSTOM_LSM6DSL_0_I2C_WriteReg BSP_I2C2_WriteReg

#define CUSTOM_LIS3MDL_0_I2C_Init BSP_I2C2_Init
#define CUSTOM_LIS3MDL_0_I2C_DeInit BSP_I2C2_DeInit
#define CUSTOM_LIS3MDL_0_I2C_ReadReg BSP_I2C2_ReadReg
#define CUSTOM_LIS3MDL_0_I2C_WriteReg BSP_I2C2_WriteReg

#ifdef __cplusplus
}
#endif

#endif /* CUSTOM_MEMS_CONF_H*/
