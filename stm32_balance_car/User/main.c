/*********************************************************************************
 * �ļ���  : ƽ�⳵�������ļ�
 * ����    ��STM32 ƽ�⳵������
*********************************************************************************/
#include "stm32f10x.h"
#include "mpu6050.h"         // mpu6050 ��̬������ͷ�ļ�
#include "i2c_mpu6050.h"
#include "motor.h"           // �������ģ��
#include "Balancecar.h"      // ƽ�⳵��Ҫͷ�ļ�
#include "SysTick.h"
#include "Serial.h"          // ���ڵ��Լ�����ͨѶ            
#include "timer.h"
#include "UltraSonic.h"     // ������������ģ��
#include <string.h>
#include <stdio.h>
/**************************************************************
 * ��������main
 * ����  ��������
 **************************************************************/
int main(void)
{	
  SystemInit();                   
	Timerx_Init(5000,7199);				   //��ʱ��TIM1
	UltraSonic_Init();               //��������ʼ��		    

	Serial_Init();						      // ���ڵ���              
	bluetooth_Init();						    //����ģ���ʼ��
	
	TIM2_PWM_Init();					      //PWM�����ʼ��
	MOTOR_Init();				            //�����ʼ������
	
  TIM3_Encoder_Init();            //��������ȡ������ PA6 PA7 
  TIM4_Encoder_Init();            //��������ȡ������ PB6 PB7	
	
	MPU6050_Init();						          //MPU6050 DMP�����ǳ�ʼ��
	SysTick_Init();						         //SysTick������ʼ��	
	BalancingCarParameterInit();			 //ƽ�⳵������ʼ��
	SysTick->CTRL |=  SysTick_CTRL_ENABLE_Msk;	 //ʹ����ʱ��

	while (1)
	{
		  MPU6050_Pose();						    //�õ�MPU6050 �ĽǶ�����
		if(MyUsart3.flag)	              //���յ�һ��������
		{
			   MyUsart3.flag=0;           //��ձ�־λ
			   Packetparsing();           // ֡Э�����
		}
				 Balancingcarcontrol();	    // �����˿��Ƴ���	
	 }
 								    
}
