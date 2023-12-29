/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "custom_motion_sensors.h"
#include "custom_motion_sensors_ex.h"
#include "tflite_app.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#ifdef __GNUC__
/* With GCC/RAISONANCE, small msg_info (option LD Linker->Libraries->Small msg_info
   set to 'Yes') calls __io_putchar() */
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#define GETCHAR_PROTOTYPE int __io_getchar(void)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#define GETCHAR_PROTOTYPE int fgetc(FILE *f)
#endif /* __GNUC__ */

#define BSP_RET_CHECK(x)	do { if (x != BSP_ERROR_NONE)	Error_Handler(); } while(0)

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
LSM6DSL_Object_t lsm6dsl_obj;
//CUSTOM_MOTION_SENSOR_AxesRaw_t AccData, GyrData, MagData;
uint8_t WorkMode = 0U;		//0: Off	1: From host	2: From sensor
uint8_t SwitchRequest = 0U;
uint8_t AccGyrRequest = 0U;
uint8_t MagRequest = 0U;
uint8_t HostRequest = 0U;
uint8_t NewDataFetched = 0U;
uint8_t FallDetected = 0U;

#ifdef FLOAT_DATA
float RecvBuffer[1][50][6];
#else
int8_t RecvBuffer[1][50][6];
#endif
uint8_t RecvBufferPTR = 0U;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void DWT_Init(void);
void DWT_Start(void);
uint32_t DWT_Stop(void);
void MX_MEMS_Init(void);
void Peripheral_Reconfig(void);
void DataFetchHandle(void);
void ModeSwitchHandle(void);
void LowPowerHandle(void);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  TFLite_Init();
  __HAL_RCC_WAKEUPSTOP_CLK_CONFIG(RCC_STOP_WAKEUPCLOCK_MSI);
  DWT_Init();
  MX_MEMS_Init();
  Peripheral_Reconfig();
  printf("リンクスタート！\r\n");

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  DataFetchHandle();
	  TFLite_Process();

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  ModeSwitchHandle();
	  LowPowerHandle();
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure LSE Drive Capability
  */
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSE|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable MSI Auto calibration
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/* USER CODE BEGIN 4 */
PUTCHAR_PROTOTYPE
{
  while (HAL_OK != HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY))
  {
    ;
  }
  return ch;
}

GETCHAR_PROTOTYPE
{
  uint8_t ch = 0;
  while (HAL_OK != HAL_UART_Receive(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY))
  {
    ;
  }
  return ch;
}

void DWT_Init(void)
{
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
	DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk; /* Disable counter */
}

void DWT_Start(void)
{
	DWT->CYCCNT = 0; /* Clear count of clock cycles */
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; /* Enable counter */
}

uint32_t DWT_Stop(void)
{
	volatile uint32_t cycles_count = 0U;
	uint32_t system_core_clock_mhz = 0U;

	DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk; /* Disable counter */
	cycles_count = DWT->CYCCNT; /* Read count of clock cycles */

	/* Calculate elapsed time in [us] */
	system_core_clock_mhz = SystemCoreClock / 1000000U;
	return cycles_count / system_core_clock_mhz;
}

void MX_MEMS_Init(void)
{
	printf("MEMS initializing.\r\n");
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Init(CUSTOM_LSM6DSL_0, MOTION_ACCELERO));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Init(CUSTOM_LSM6DSL_0, MOTION_GYRO));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Mode(CUSTOM_LSM6DSL_0, LSM6DSL_BYPASS_MODE));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Disable(CUSTOM_LSM6DSL_0, MOTION_ACCELERO));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Disable(CUSTOM_LSM6DSL_0, MOTION_GYRO));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetFullScale(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, 4));
	#ifdef FLOAT_DATA
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetFullScale(CUSTOM_LSM6DSL_0, MOTION_GYRO, 1000));
	#else
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetFullScale(CUSTOM_LSM6DSL_0, MOTION_GYRO, 500));
	#endif
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LSM6DSL_0, LSM6DSL_MASTER_CONFIG, 0x0));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_CTRL2, 0x0));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_ODR_Value(CUSTOM_LSM6DSL_0, 104.0f));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Decimation(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, LSM6DSL_FIFO_XL_NO_DEC));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Decimation(CUSTOM_LSM6DSL_0, MOTION_GYRO, LSM6DSL_FIFO_GY_NO_DEC));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Watermark_Level(CUSTOM_LSM6DSL_0, 90));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LSM6DSL_0, LSM6DSL_INT1_CTRL, 0x8));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Mode(CUSTOM_LSM6DSL_0, LSM6DSL_STREAM_MODE));



//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Init(CUSTOM_LIS3MDL_0, MOTION_MAGNETO));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetOutputDataRate(CUSTOM_LIS3MDL_0, MOTION_MAGNETO, 80.0f));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetFullScale(CUSTOM_LIS3MDL_0, MOTION_MAGNETO, 4));
//	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LIS3MDL_0, LIS3MDL_CTRL_REG3, 0x0));
	printf("MEMS initialized.\r\n");

}

void Peripheral_Reconfig(void)
{
	HAL_UART_AbortReceive(&huart1);
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Mode(CUSTOM_LSM6DSL_0, LSM6DSL_BYPASS_MODE));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Disable(CUSTOM_LSM6DSL_0, MOTION_ACCELERO));
	BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Disable(CUSTOM_LSM6DSL_0, MOTION_GYRO));

	if(WorkMode == 2U)
	{
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LSM6DSL_0, LSM6DSL_INT1_CTRL, 0x8));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetOutputDataRate(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, 104.0f));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetOutputDataRate(CUSTOM_LSM6DSL_0, MOTION_GYRO, 104.0f));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Enable(CUSTOM_LSM6DSL_0, MOTION_ACCELERO));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Enable(CUSTOM_LSM6DSL_0, MOTION_GYRO));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_FIFO_Set_Mode(CUSTOM_LSM6DSL_0, LSM6DSL_STREAM_MODE));
	}
	else if(WorkMode == 3U)
	{
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LSM6DSL_0, LSM6DSL_INT1_CTRL, 0x3));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetOutputDataRate(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, 12.5f));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_SetOutputDataRate(CUSTOM_LSM6DSL_0, MOTION_GYRO, 12.5f));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTX_H_XL, (uint8_t*)RecvBuffer));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTX_H_G, (uint8_t*)RecvBuffer));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Enable(CUSTOM_LSM6DSL_0, MOTION_ACCELERO));
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Enable(CUSTOM_LSM6DSL_0, MOTION_GYRO));
	}
	else
	{
		BSP_RET_CHECK(CUSTOM_MOTION_SENSOR_Write_Register(CUSTOM_LSM6DSL_0, LSM6DSL_INT1_CTRL, 0x0));
		if(WorkMode == 1U)
		{
			HAL_UART_Receive_IT(&huart1, (uint8_t *)RecvBuffer, 9);
		}
	}
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, (WorkMode==1U || WorkMode==3U) ? GPIO_PIN_SET : GPIO_PIN_RESET);
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, (WorkMode==2U || WorkMode==3U) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void DataFetchHandle(void)
{
	uint8_t temp_u8[2];
	uint16_t temp_u16;
	float* ptr_f32;
	if(WorkMode == 1U)
	{
		if(HostRequest)
		{
			printf("Echo\r\n");
			if(HAL_UART_Receive(&huart1, (uint8_t *)RecvBuffer, sizeof(RecvBuffer), 3000U) == HAL_OK)
			{
				printf("Data received.\r\n");
				NewDataFetched = 1U;
			}
			else
			{
				printf("Transmission failed.\r\n");
			}
			HostRequest = 0U;
			HAL_UART_Receive_IT(&huart1, (uint8_t *)RecvBuffer, 9);
		}
	}
	else if(WorkMode == 2U)
	{
		if(AccGyrRequest)
		{
			CUSTOM_MOTION_SENSOR_FIFO_Get_Pattern(CUSTOM_LSM6DSL_0, &temp_u16);
			while(temp_u16 != 0U)
			{
				CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_DATA_OUT_L, temp_u8);
				CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_DATA_OUT_H, temp_u8+1);
				CUSTOM_MOTION_SENSOR_FIFO_Get_Pattern(CUSTOM_LSM6DSL_0, &temp_u16);
			}
			for(uint8_t i=0; i<10; i++)
			{
				for(uint8_t j=3; j<6; j++)
				{
					CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_DATA_OUT_L, temp_u8);
					CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_DATA_OUT_H, temp_u8+1);
					#ifdef FLOAT_DATA
					RecvBuffer[0][RecvBufferPTR][j] = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_1000DPS * -0.001f);
					#else
					RecvBuffer[0][RecvBufferPTR][j] = -(int8_t)(temp_u8[1]);
					#endif
				}
				RecvBuffer[0][RecvBufferPTR][5] = -RecvBuffer[0][RecvBufferPTR][5];
				for(uint8_t j=0; j<3; j++)
				{
					CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_DATA_OUT_L, temp_u8);
					CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_FIFO_DATA_OUT_H, temp_u8+1);
					#ifdef FLOAT_DATA
					RecvBuffer[0][RecvBufferPTR][j] = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_ACC_SENSITIVITY_FS_4G * 0.001f);
					#else
					RecvBuffer[0][RecvBufferPTR][j] = (int8_t)(temp_u8[1]);
					#endif
				}
				RecvBuffer[0][RecvBufferPTR][2] = -RecvBuffer[0][RecvBufferPTR][2];
				RecvBufferPTR = (RecvBufferPTR + 1U) % 50U;
//				CUSTOM_MOTION_SENSOR_FIFO_Get_Axis(CUSTOM_LSM6DSL_0, MOTION_GYRO, &GyrData.x);
//				CUSTOM_MOTION_SENSOR_FIFO_Get_Axis(CUSTOM_LSM6DSL_0, MOTION_GYRO, &GyrData.y);
//				CUSTOM_MOTION_SENSOR_FIFO_Get_Axis(CUSTOM_LSM6DSL_0, MOTION_GYRO, &GyrData.z);
//				CUSTOM_MOTION_SENSOR_FIFO_Get_Axis(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, &AccData.x);
//				CUSTOM_MOTION_SENSOR_FIFO_Get_Axis(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, &AccData.y);
//				CUSTOM_MOTION_SENSOR_FIFO_Get_Axis(CUSTOM_LSM6DSL_0, MOTION_ACCELERO, &AccData.z);
			}
//			printf("AccGyr data fetched.\r\n");
			AccGyrRequest = 0U;
			NewDataFetched = 1U;
		}
	}
	else if(WorkMode == 3U)
	{
		if(AccGyrRequest)
		{
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_STATUS_REG, temp_u8);
			while((*temp_u8 & 0x3) != 0x3)
			{
				CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_STATUS_REG, temp_u8);
			}
			ptr_f32 = (float*)RecvBuffer;
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTX_L_XL, temp_u8);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTX_H_XL, temp_u8+1);
			*(ptr_f32) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_ACC_SENSITIVITY_FS_4G * 0.001f);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTY_L_XL, temp_u8);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTY_H_XL, temp_u8+1);
			*(ptr_f32+1) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_ACC_SENSITIVITY_FS_4G * 0.001f);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTZ_L_XL, temp_u8);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTZ_H_XL, temp_u8+1);
			*(ptr_f32+2) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_ACC_SENSITIVITY_FS_4G * -0.001f);

			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTX_L_G, temp_u8);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTX_H_G, temp_u8+1);
			#ifdef FLOAT_MODEL_INPUT
			*(ptr_f32+3) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_1000DPS * -0.001f);
			#else
			*(ptr_f32+3) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_500DPS * -0.001f);
			#endif
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTY_L_G, temp_u8);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTY_H_G, temp_u8+1);
			#ifdef FLOAT_MODEL_INPUT
			*(ptr_f32+4) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_1000DPS * -0.001f);
			#else
			*(ptr_f32+4) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_500DPS * -0.001f);
			#endif
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTZ_L_G, temp_u8);
			CUSTOM_MOTION_SENSOR_Read_Register(CUSTOM_LSM6DSL_0, LSM6DSL_OUTZ_H_G, temp_u8+1);
			#ifdef FLOAT_MODEL_INPUT
			*(ptr_f32+5) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_1000DPS * 0.001f);
			#else
			*(ptr_f32+5) = (int16_t)(((int16_t)temp_u8[1] << 8) | temp_u8[0]) * (LSM6DSL_GYRO_SENSITIVITY_FS_500DPS * 0.001f);
			#endif

//			printf("AccGyr data fetched.\r\n");
			printf("Ax=%f  Ay=%f  Az=%f  Gx=%f  Gy=%f  Gz=%f\r\n",
					*(ptr_f32), *(ptr_f32+1), *(ptr_f32+2),	*(ptr_f32+3), *(ptr_f32+4), *(ptr_f32+5));

			AccGyrRequest = 0U;
		}
	}
}

void ModeSwitchHandle(void)
{
	if(SwitchRequest)
	{
		WorkMode = (WorkMode + 1U) % 4U;
		#ifdef PROFILING
		WorkMode = (WorkMode==2U) ? 3U : WorkMode;
		#endif
		Peripheral_Reconfig();
		printf("Mode %u selected.\r\n", WorkMode);
		AccGyrRequest = 0U;
		MagRequest = 0U;
		HostRequest = 0U;
		RecvBufferPTR = 0U;
		SwitchRequest = 0U;
	}
}

void LowPowerHandle(void)
{
	HAL_SuspendTick();
	if(WorkMode == 1U)
	{
//		HAL_PWREx_EnableLowPowerRunMode();
		HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);
	}
	else
	{
		HAL_PWREx_EnterSTOP2Mode(PWR_SLEEPENTRY_WFI);
		SystemClock_Config();
	}
	HAL_ResumeTick();
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	switch(GPIO_Pin)
	{
	case GPIO_PIN_8:
		if(WorkMode == 2U)
		{
			MagRequest = 1U;
		}
		break;
	case GPIO_PIN_11:
		if(WorkMode == 2U || WorkMode == 3U)
		{
			AccGyrRequest = 1U;
		}
		break;
	case GPIO_PIN_13:
		SwitchRequest = 1U;
		break;

	default:
		break;
	}
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if(huart->Instance == USART1)
	{
		if(WorkMode==1U && strstr((char*)RecvBuffer, "Connect"))
		{
			HostRequest = 1U;
		}
	}
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
