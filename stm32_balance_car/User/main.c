/*********************************************************************************
 * 文件名  : 平衡车主程序文件
 * 描述    ：STM32 平衡车主程序
*********************************************************************************/
#include "stm32f10x.h"
#include "mpu6050.h"         // mpu6050 姿态传感器头文件
#include "i2c_mpu6050.h"
#include "motor.h"           // 电机驱动模块
#include "Balancecar.h"      // 平衡车主要头文件
#include "SysTick.h"
#include "Serial.h"          // 串口调试及蓝牙通讯            
#include "timer.h"
#include "UltraSonic.h"     // 超声波传感器模块
#include <string.h>
#include <stdio.h>
/**************************************************************
 * 函数名：main
 * 描述  ：主函数
 **************************************************************/
int main(void)
{	
  SystemInit();                   
	Timerx_Init(5000,7199);				   //定时器TIM1
	UltraSonic_Init();               //超声波初始化		    

	Serial_Init();						      // 串口调试              
	bluetooth_Init();						    //蓝牙模块初始化
	
	TIM2_PWM_Init();					      //PWM输出初始化
	MOTOR_Init();				            //电机初始化函数
	
  TIM3_Encoder_Init();            //编码器获取脉冲数 PA6 PA7 
  TIM4_Encoder_Init();            //编码器获取脉冲数 PB6 PB7	
	
	MPU6050_Init();						          //MPU6050 DMP陀螺仪初始化
	SysTick_Init();						         //SysTick函数初始化	
	BalancingCarParameterInit();			 //平衡车参数初始化
	SysTick->CTRL |=  SysTick_CTRL_ENABLE_Msk;	 //使能总时钟

	while (1)
	{
		  MPU6050_Pose();						    //得到MPU6050 的角度数据
		if(MyUsart3.flag)	              //接收到一次数据了
		{
			   MyUsart3.flag=0;           //清空标志位
			   Packetparsing();           // 帧协议解析
		}
				 Balancingcarcontrol();	    // 机器人控制程序	
	 }
 								    
}
