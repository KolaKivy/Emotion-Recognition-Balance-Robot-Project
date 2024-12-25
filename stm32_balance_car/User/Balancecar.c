/*************************************************************
 平衡车控制主要功能
**************************************************************/
#include "Balancecar.h"
#include "I2C_MPU6050.h"
#include "MOTOR.h"
#include "Serial.h"
#include "MPU6050.h"
#include "Ultrasonic.h"
#include "stm32f10x_gpio.h"
#include "math.h" 
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*****************************************/
TypeUsart3 MyUsart3;  // 蓝牙模块接收

// 平衡小车运动状态枚举
enum{
  enCARSTOP = 0,
  enCARRUN,
  enCARBACK,
  enCARLEFT,
  enCARRIGHT,
  enCARTLEFT,
  enCARTRIGHT
}enCarState;


#define 	run_balancecar     '1'// 平衡车前进
#define 	back_balancecar    '2'// 平衡车后退
#define 	left_balancecar    '4'// 平衡车左转
#define 	right_balancecar   '3'// 平衡车右转
#define 	stop_balancecar    '0'// 平衡车停止


int get_newcarstate = enCARSTOP; //  1前2后3左4右0停止

/*****************多数据************************/
uint8_t CLB_u8MainEventCount;						  //主循环判断计数  在SysTick_Handler(void)中使用 每1ms加1
uint8_t CLB_u8SpeedControlCount;						  //速度控制循环计数  在SysTick_Handler(void)中使用 每5ms加1
uint8_t CLB_u8SpeedControlPeriod;
uint8_t CLB_u8DirectionControlPeriod;
uint8_t CLB_u8DirectionControlCount;					  //转向控制循环计数  在SysTick_Handler(void)中使用 每5ms加1 
uint8_t CLB_u8trig;
uint8_t ucBluetoothValue;                      //蓝牙接收数据


/******电机控制参数******/
float CLB_fSpeedControlOut;						   //速度控制PWM
float CLB_fSpeedControlOutOld;
float CLB_fSpeedControlOutNew;
float CLB_fAngleControlOut;
float CLB_fLeftMotorOut;
float CLB_fRightMotorOut;

float CLB_fCarAngle;						 //角度控制PWM

//-----角度环和速度环PID控制参数-----
float  CLB_fCarAngle_P = 91.3;  
float  CLB_fCarAngle_D = 0.21;	

float  CLB_fCarSpeed_P= - 5.1;
float  CLB_fCarSpeed_I= - 0.10;;


/******************************************************
    速度控制参数
******************************************************/

int16_t   CLB_s16LeftMotorPulse;					   //左电机脉冲数
int16_t	  CLB_s16RightMotorPulse;					   //右电机脉冲数

int32_t   CLB_s32LeftMotorPulseOld;
int32_t   CLB_s32RightMotorPulseOld;
int32_t   CLB_s32LeftMotorPulseSigma;				  //50ms左电机叠加值
int32_t   CLB_s32RightMotorPulseSigma;				 //50ms右电机叠加值

float CLB_fCarSpeed;							           //测速码盘得出的车速
float CLB_fCarSpeedOld;

float CLB_fCarPosition;						          //测速码盘通过计算得到的小车位移

/*-----悬停参数-----*/
int leftstop=0;
int rightstop=0;
int stopflag=0;

/******超声波********/
float fultrasonic  = 0;							         //超声波控制量
float Distance  = 0;									       //超声波距离

/******蓝牙控制参数******/																	
float CLB_fBluetoothSpeed;						     //蓝牙控制车速
float CLB_fBluetoothDirectionNew;			     //用于平缓输出车速使用
float CLB_fBluetoothDirectionSL;			     //左转标志位  
float CLB_fBluetoothDirectionSR;			     //右转标志位	  
int Ultrasonicflag = 0;
int y1,z1,y2,z2,flagbt;

float dac = 0,dgy = 0;

/************旋转*****************/
float CLB_fBluetoothDirectionL;				   //左旋转标志位  
float CLB_fBluetoothDirectionR;				   //右旋转标志位  
int driectionxco=800;

/**********延时子函数*******************************************/
void delay_nms(uint16_t time)
{    
   uint16_t i=0;  
   while(time--)
   {
      i=12000;  
      while(i--) ;    
   }
}
/**************************************************************/

/***************************************************************
** 函数名称: BalancingCarParameterInit
** 功能描述: 平衡车参数初始化
***************************************************************/
void BalancingCarParameterInit(void)
{
	CLB_s16LeftMotorPulse = CLB_s16RightMotorPulse = 0;					                  //左右脉冲值   初始化
	CLB_s32LeftMotorPulseSigma = CLB_s32RightMotorPulseSigma = 0;		              //叠加脉冲数	 初始化

	CLB_fCarSpeed = CLB_fCarSpeedOld = 0;								                          //平衡小车车速	初始化
	CLB_fCarPosition = 0;												                                  //平衡小车位移量	初始化
	CLB_fCarAngle    = 0;												                                  //平衡小车角度值	初始化

	CLB_fAngleControlOut = CLB_fSpeedControlOut = CLB_fBluetoothDirectionNew = 0;	//角度PWM、车速PWM、蓝牙控制PWM	 初始化
	CLB_fLeftMotorOut    = CLB_fRightMotorOut   = 0;								              //左右车轮PWM输出值 			       初始化
	CLB_fBluetoothSpeed  = 0;														                          //蓝牙控制车速值                 初始化
	CLB_fBluetoothDirectionL =CLB_fBluetoothDirectionR= 0;						            //蓝牙控制左右旋转标志位         初始化
	CLB_fBluetoothDirectionSL =CLB_fBluetoothDirectionSR= 0;						          //蓝牙控制左右转向标志位         初始化
	
  CLB_u8MainEventCount=0;															                          //用于5ms定时器子程序SysTick_Handler(void)中总中断计数位
	CLB_u8SpeedControlCount=0;														                        //用于5ms定时器子程序SysTick_Handler(void)中50ms速度平衡融入计数位
  CLB_u8SpeedControlPeriod=0;														                        //用于5ms定时器子程序SysTick_Handler(void)中50ms速度平衡融入计数位

	fultrasonic=0;											                                          //用于5ms定时器子程序SysTick_Handler(void)中超声波平衡融入计数位
}


/***************************************************************
** 函数名称: AngleControl
** 功能描述: 角度环控制函数

***************************************************************/
void AngleControl(void)	 
{
	if(flagbt==1)
	{
		CLB_fCarAngle_P=0;
		CLB_fCarAngle_P=y1*1.71875;
	}
		if(flagbt==2)
	{
		CLB_fCarAngle_D=0;
		CLB_fCarAngle_D=(z1-64)*0.15625;
	}
	dac=accel[2];
	dgy=gyro[2];
	if(Pitch==0||Pitch<-20||Pitch>20)			     //MPU6050状态指示灯 STM32核心板 PC13 绿色灯亮起为不正常
	{
	  GPIO_ResetBits(GPIOC, GPIO_Pin_13); 		 //MPU6050状态指示灯 STM32核心板 PC13 绿色灯亮起为不正常
	}
	else 
	{GPIO_SetBits(GPIOC, GPIO_Pin_13);}			 //MPU6050状态正常时LED灯熄灭
	
	CLB_fCarAngle = Roll - CAR_ZERO_ANGLE;													   //DMP ROLL滚动方向角度与预设小车倾斜角度值的差得出角度   
	CLB_fAngleControlOut =  CLB_fCarAngle * CLB_fCarAngle_P + gyro[0] * CLB_fCarAngle_D ;	  //角度PD控制							   
}

/***************************************************************
** 函数名称: SetMotorVoltageAndDirection
** 功能描述: 电机转速             
***************************************************************/
void SetMotorVoltageAndDirection(int16_t s16LeftVoltage,int16_t s16RightVoltage)
{
	  uint16_t u16LeftMotorValue;
	  uint16_t u16RightMotorValue;
	
    if(s16LeftVoltage<0)								 //当左电机PWM输出为负时 PB14设为正 PB15设为负
    {	
	    GPIO_SetBits(GPIOB, GPIO_Pin_14 );				    
      GPIO_ResetBits(GPIOB, GPIO_Pin_15 );
      s16LeftVoltage = (-s16LeftVoltage);
    }
    else 
    {	
      GPIO_SetBits(GPIOB, GPIO_Pin_15 );  //当左电机PWM输出为正时 PB14设为负 PB15设为正
      GPIO_ResetBits(GPIOB, GPIO_Pin_14 ); 
      s16LeftVoltage = s16LeftVoltage;
    }

    if(s16RightVoltage<0)
    {															       //当右电机PWM输出为负时 PB12设为正 PB13设为负
      GPIO_SetBits(GPIOB, GPIO_Pin_13 );				    
      GPIO_ResetBits(GPIOB, GPIO_Pin_12 );
      s16RightVoltage = (-s16RightVoltage);
    }
    else														    //当右电机PWM输出为正时 PB12设为负 PB13设为正
    {
	    GPIO_SetBits(GPIOB, GPIO_Pin_12 );				    
      GPIO_ResetBits(GPIOB, GPIO_Pin_13 );	
     
      s16RightVoltage = s16RightVoltage;
    }
		
	   u16RightMotorValue= (u16)s16RightVoltage;
	   u16LeftMotorValue = (u16)s16LeftVoltage;


	TIM_SetCompare3(TIM2,u16LeftMotorValue);			  //TIM2与 u16RightMotorValue对比，不相同则翻转波形，调节PWM占空比
	TIM_SetCompare4(TIM2,u16RightMotorValue);			  //TIM3与 u16LeftMotorValue对比，不相同则翻转波形，调节PWM占空比
		
	// 判断平衡车是否出现悬停或者跌倒
  if(Pitch>10||Pitch<-10&CLB_fBluetoothDirectionSR==0&CLB_fBluetoothDirectionSL==0)
	{		
		TIM_SetCompare3(TIM2,0);
		TIM_SetCompare4(TIM2,0);
		stopflag=1;		
	}
	else stopflag=0;
	
	if(CLB_fCarAngle > 50 || CLB_fCarAngle < (-50))
	{
		TIM_SetCompare3(TIM2,0);
		TIM_SetCompare4(TIM2,0);  
		stopflag=1;	
	}
	else stopflag=0;
}

// 电机输出函数
void MotorOutput(void)																					  
{	   
	  //右电机转向PWM控制融合平衡角度、速度输出
		CLB_fLeftMotorOut  = CLB_fAngleControlOut +CLB_fSpeedControlOutNew + CLB_fBluetoothDirectionNew;		//左电机转向PWM控制融合平衡角度、速度输出	
    CLB_fRightMotorOut = CLB_fAngleControlOut +CLB_fSpeedControlOutNew - CLB_fBluetoothDirectionNew;		//右电机转向PWM控制融合平衡角度、速度输出

	if((int16_t)CLB_fLeftMotorOut  > MOTOR_OUT_MAX)	CLB_fLeftMotorOut  = MOTOR_OUT_MAX;
	if((int16_t)CLB_fLeftMotorOut  < MOTOR_OUT_MIN)	CLB_fLeftMotorOut  = MOTOR_OUT_MIN;
	if((int16_t)CLB_fRightMotorOut > MOTOR_OUT_MAX)	CLB_fRightMotorOut = MOTOR_OUT_MAX;
	if((int16_t)CLB_fRightMotorOut < MOTOR_OUT_MIN)	CLB_fRightMotorOut = MOTOR_OUT_MIN;
	
    SetMotorVoltageAndDirection((int16_t)CLB_fLeftMotorOut,(int16_t)CLB_fRightMotorOut);
    
}
// 获取电机的脉冲
void GetMotorPulse(void)              //采集电机速度脉冲
{ 
	uint16_t u16TempLeft;
	uint16_t u16TempRight;
	
	u16TempLeft = TIM_GetCounter(TIM3);   //  TIM3定时器计算调用
 	u16TempRight= TIM_GetCounter(TIM4);	//	 TIM4定时器计算调用
	leftstop=u16TempLeft;
	rightstop=u16TempRight;
	TIM_SetCounter(TIM3,0);
	TIM_SetCounter(TIM4,0);  //清零
	CLB_s16LeftMotorPulse=u16TempLeft;
	CLB_s16RightMotorPulse=(-u16TempRight);
		
	CLB_s32LeftMotorPulseSigma  +=CLB_s16LeftMotorPulse;		 //脉冲值叠加 40ms叠加值
	CLB_s32RightMotorPulseSigma +=CLB_s16RightMotorPulse; 	 //脉冲值叠加 40ms叠加值
}

/***************************************************************
** 函数名称: SpeedControl
** 功能描述: 速度环控制函数
***************************************************************/
void SpeedControl(void)
{
	CLB_fCarSpeed = (CLB_s32LeftMotorPulseSigma  + CLB_s32RightMotorPulseSigma );  //左右电机脉冲数平均值作为小车当前车速
	CLB_s32LeftMotorPulseSigma = CLB_s32RightMotorPulseSigma = 0;	                //全局变量 注意及时清零		
	CLB_fCarSpeedOld *= 0.7;
	CLB_fCarSpeedOld +=CLB_fCarSpeed*0.3;
	
	CLB_fCarPosition += CLB_fCarSpeedOld; 		 //路程  即速度积分	  
	CLB_fCarPosition += CLB_fBluetoothSpeed;   //融合  蓝牙给定速度
	CLB_fCarPosition +=	fultrasonic;		       //融合  超声波给定速度
	if(stopflag==1)
	{
		CLB_fCarPosition=0;	
	}
	//积分上限设限//
	if((int32_t)CLB_fCarPosition > CAR_POSITION_MAX)    CLB_fCarPosition = CAR_POSITION_MAX;
	if((int32_t)CLB_fCarPosition < CAR_POSITION_MIN)    CLB_fCarPosition = CAR_POSITION_MIN;
	
	if(flagbt==3)
	{
		CLB_fCarSpeed_P=0;
		CLB_fCarSpeed_P=(y2-128)*0.46875;
	}
	if(flagbt==4)
	{
		CLB_fCarSpeed_I=0;
		CLB_fCarSpeed_I=(z2-192)*0.15625;
	}																							  
	CLB_fSpeedControlOutNew = (CLB_fCarSpeedOld -CAR_SPEED_SET ) * CLB_fCarSpeed_P + (CLB_fCarPosition - CAR_POSITION_SET ) * CLB_fCarSpeed_I; //速度PI算法 速度*P +位移*I=速度PWM输出
}
/**********************************************************
** 函数功能：
** 蓝牙模块接收命令中断函数
***********************************************************/
void USART3_IRQHandler(void)
{  
   if(USART_GetITStatus(USART3,USART_IT_RXNE)==SET)   //判断接收中断
   {
	   USART_ClearITPendingBit(USART3,USART_IT_RXNE);  //清除接收标志位
      MyUsart3.buff[MyUsart3.len++]=USART_ReceiveData(USART3);      //接收数据
   }
	 
   if(USART_GetITStatus(USART3,USART_IT_IDLE)==SET)   //判断空闲中断
   { 
      
		  MyUsart3.buff[MyUsart3.len]='\0';
      MyUsart3.flag=1;
      MyUsart3.len=0;
      USART_ReceiveData(USART3);
   }
}	 
//超声波读取距离值然后把该值加入到速度PI中计算
void UltraSonicProcessing(void)
{  
	if(Ultrasonicflag==0)
	{
    Distance =TIM_GetCounter(TIM1)*5*34/200.0;  //单位转换为cm
		// 如果距离值小于5cm，平衡车输出向后PWM值
	  if(Distance <  5.00)							
		{
	    	fultrasonic=500;
	    }
		// 如果距离值大于5cm且小于或等于8cm小车前进，追踪
		else if(Distance >= 5 & Distance <= 8)  
		{
			fultrasonic=-300;
		}
		// 距离值大于8cm，超声波调整失效
	  else 
		{
			fultrasonic=0;	
 	  }
  }
 }
// 平衡小车控制程序
void Balancingcarcontrol(void)
{
	switch (get_newcarstate)
	{
		case enCARSTOP: //停止
		{
			CLB_fBluetoothSpeed = 0;
			fultrasonic=0;
			CLB_fBluetoothDirectionNew=0;
			Ultrasonicflag=0;
		} break; 					   
		case enCARRUN: //向前速度 -800
		{
			CLB_fBluetoothDirectionNew= 0; 	
			CLB_fBluetoothSpeed =  -800 ;
			Ultrasonicflag=1;
		}break;	   
		case enCARLEFT://左转 
		{
			CLB_fBluetoothDirectionNew= 300; 
			Ultrasonicflag=1;

		}break;   
		case enCARRIGHT: //右转
		{
			CLB_fBluetoothDirectionNew= -300; 
			Ultrasonicflag=1;
		}break;	
		case enCARBACK: //后退速度 800
		{
			CLB_fBluetoothDirectionNew= 0; 		
			CLB_fBluetoothSpeed = (800);
			Ultrasonicflag=1;  
		}break;
		case enCARTLEFT: //左旋
		{
			CLB_fBluetoothDirectionNew = driectionxco; 
			Ultrasonicflag=1; 
		}break;
		case enCARTRIGHT: //右旋
		{
			CLB_fBluetoothDirectionNew = -driectionxco; 
			Ultrasonicflag=1;
		}break;		
		default: CLB_fBluetoothSpeed = 0; break; 					   //停止
	}
}
/***************************************************************************
串口协议数据解析
***************************************************************************/ 
void Packetparsing(void)
{
		if(strcmp((const char*)MyUsart3.buff,"ONA")==0){printf("go forward!\r\n"); get_newcarstate = enCARRUN;}      // 前进
		else if(strcmp((const char*)MyUsart3.buff,"ONB")==0){printf("go back!\r\n");get_newcarstate = enCARBACK;}    // 后退
		else if(strcmp((const char*)MyUsart3.buff,"ONC")==0){printf("go left!\r\n"); get_newcarstate = enCARLEFT;}   // 左转
		else if(strcmp((const char*)MyUsart3.buff,"OND")==0){printf("go right!\r\n");get_newcarstate = enCARRIGHT;}  // 右转
		else if(strcmp((const char*)MyUsart3.buff,"ONG")==0){printf("Left rotation!\r\n");get_newcarstate = enCARTLEFT;}   // 左旋
		else if(strcmp((const char*)MyUsart3.buff,"ONH")==0){printf("right rotation!\r\n");get_newcarstate = enCARTRIGHT;} // 右旋
		else if(strcmp((const char*)MyUsart3.buff,"ONF")==0){printf("Stop!\r\n");get_newcarstate = enCARSTOP;}   // 停止
		else if(strcmp((const char*)MyUsart3.buff,"ONE")==0){printf("Stop!\r\n"); get_newcarstate = enCARSTOP;}  // 停止	
}

 
 
 
 
 
 
 
 
 
