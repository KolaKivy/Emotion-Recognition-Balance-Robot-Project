/************************************************************************
 电机驱动定义如下说明：
  控制GPIO PB12/PB13/PB14/PB15
  电机MOTOR  TIM2_CH3/TIM2_CH4
  光电编码器 TIM3_CH1/TIM3_CH2
  光电编码器 TIM4_CH1/TIM4_CH2
************************************************************************/
#include "stm32f10x.h"
#include "motor.h"
#include "stm32f10x_exti.h"

#define ENCODER_TIM_PERIOD (uint16_t)0xFFFF
#define MAX_COUNT          (uint16_t)0x0FFF

int leftcount;
int rightcount;


void MOTOR_Init(void)
{		
	/*定义一个GPIO_InitTypeDef类型的结构体*/
	GPIO_InitTypeDef GPIO_InitStructure;

	/*开启GPIOB的外设时钟*/
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE); 

	/*选择要控制的GPIOB引脚*/															   
  	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15;	

	/*设置引脚模式为通用推挽输出*/
  	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;   

	/*设置引脚速率为50MHz */   
  	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 

	/*调用库函数，初始化GPIOB*/
  	GPIO_Init(GPIOB, &GPIO_InitStructure);		  

	/* 低电平	*/
	GPIO_SetBits(GPIOB, GPIO_Pin_13 | GPIO_Pin_14|GPIO_Pin_12 | GPIO_Pin_15);
}

/*
 * 函数名：TIM2_GPIO_Config
 * 描述  ：配置TIM3复用输出PWM时用到的I/O
 * 输入  ：无
 * 输出  ：无
 * 调用  ：内部调用
 */
static void TIM2_GPIO_Config(void) 
{
  GPIO_InitTypeDef GPIO_InitStructure;

	/* TIM2 clock enable */
	//PCLK1经过2倍频后作为TIM2的时钟源等于72MHz
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE); 

  /* GPIOA clock enable */
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE); 

  /*GPIOA Configuration: TIM2 channel 3 and 4 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin =  GPIO_Pin_2 | GPIO_Pin_3;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;		    // 复用推挽输出
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

  GPIO_Init(GPIOA, &GPIO_InitStructure);

}

/*
 * 函数名：TIM2_Mode_Config
 * 描述  ：配置TIM3输出的PWM信号的模式，如周期、极性、占空比
 * 输入  ：无
 * 输出  ：无
 * 调用  ：内部调用
 */
static void TIM2_Mode_Config(void)
{
	TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	TIM_OCInitTypeDef  TIM_OCInitStructure;

	/* PWM信号电平跳变值 */
	uint16_t CCR3_Val = 0;
	uint16_t CCR4_Val = 0;

  /* Time base configuration */		 
  TIM_TimeBaseStructure.TIM_Period = 999 ;       //ARR 当定时器从0计数到999，即为1000次，为一个定时周期
  TIM_TimeBaseStructure.TIM_Prescaler = 0;	    //设置预分频：不预分频，即为72MHz
  TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1 ;	//设置时钟分频系数：不分频
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //向上计数模式

  TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);

  /* PWM2 Mode configuration: Channel3 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;	    //配置为PWM模式1
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;	//比较输出使能
  TIM_OCInitStructure.TIM_Pulse = CCR3_Val;	   //设置跳变值，当计数器计数到这个值时，电平发生跳变
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;  //当定时器计数值小于CCR1_Val时为高电平

  TIM_OC3Init(TIM2, &TIM_OCInitStructure);	 //使能通道2
  TIM_OC3PreloadConfig(TIM2, TIM_OCPreload_Enable);

  /* PWM1 Mode configuration: Channel4 */
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = CCR4_Val;	  //设置通道2的电平跳变值，输出另外一个占空比的PWM
	TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;  //当定时器计数值小于CCR1_Val时为高电平

  TIM_OC4Init(TIM2, &TIM_OCInitStructure);	  //使能通道3
  TIM_OC4PreloadConfig(TIM2, TIM_OCPreload_Enable);

  TIM_ARRPreloadConfig(TIM2, ENABLE);			 // 使能TIM3重载寄存器ARR

  /* TIM2 enable counter */
  TIM_Cmd(TIM2, ENABLE);                   //使能定时器2	
}

/*
 * 函数名：TIM2_PWM_Init
 * 描述  ：TIM2 输出PWM信号初始化，只要调用这个函数
 *         TIM2的四个通道就会有PWM信号输出
 * 输入  ：无
 * 输出  ：无
 * 调用  ：外部调用
 */
void TIM2_PWM_CHANGE(uint16_t CCR3,uint16_t CCR4)
{
	
	TIM_OCInitTypeDef  TIM_OCInitStructure;
	uint16_t CCR3_Val ;
	uint16_t CCR4_Val ;
	CCR3_Val = CCR3;
	CCR4_Val = CCR4;
	TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;	    //配置为PWM模式1
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;	
  TIM_OCInitStructure.TIM_Pulse = CCR3_Val;	   //设置跳变值，当计数器计数到这个值时，电平发生跳变
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;  //当定时器计数值小于CCR1_Val时为高电平

  TIM_OC3Init(TIM2, &TIM_OCInitStructure);	 //使能通道2

  TIM_OC3PreloadConfig(TIM2, TIM_OCPreload_Enable);

  /* PWM1 Mode configuration: Channel4 */
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = CCR4_Val;	  //设置通道2的电平跳变值，输出另外一个占空比的PWM

  TIM_OC4Init(TIM2, &TIM_OCInitStructure);	  //使能通道3
  TIM_OC4PreloadConfig(TIM2, TIM_OCPreload_Enable);
}


void TIM2_PWM_Init(void)
{
	TIM2_GPIO_Config();
	TIM2_Mode_Config();	
}

void TIM3_IRQHandler(void)
{ 		    		  			    
	uint16_t tsr;
	tsr=TIM3->SR;	
	if(tsr&0X0001)//溢出中断
	{
																				
	}				   
	TIM3->SR&=~(1<<0);//清除中断标志位 	 
}


void TIM4_IRQHandler(void)
{ 		    		  			    
	uint16_t tsr;
	tsr=TIM4->SR;	
	if(tsr&0X0001)//溢出中断
	{
																				
	}				   
	TIM4->SR&=~(1<<0);//清除中断标志位 	 
}

void TIM3_Encoder_Init(void)
{
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_ICInitTypeDef TIM_ICInitStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
  
  /* Encoder unit connected to TIM3, 4X mode */    
  GPIO_InitTypeDef GPIO_InitStructure;
  //NVIC_InitTypeDef NVIC_InitStructure;
  
  /* TIM3 clock source enable */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);
  /* Enable GPIOA, clock */
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
  
  GPIO_StructInit(&GPIO_InitStructure);
  /* Configure PA.06,07 as encoder input */
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_7;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOA, &GPIO_InitStructure);
  
  /* Enable the TIM3 Update Interrupt */
  NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);    //优先级组别
  NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority =1;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
  
  /* Timer configuration in Encoder mode */
  TIM_DeInit(TIM3);
  TIM_TimeBaseStructInit(&TIM_TimeBaseStructure);
  
  TIM_TimeBaseStructure.TIM_Prescaler = 0x0;  // No prescaling 
  TIM_TimeBaseStructure.TIM_Period = 0xFFFF;  
  TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;  //设置时钟分频系数：不分频
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //向上计数模式 
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);
 
  //TIM_EncoderInterfaceConfig(TIM3, TIM_EncoderMode_TI12, TIM_ICPolarity_Falling, TIM_ICPolarity_Falling); //TIM_ICPolarity_Rising上升沿捕获
  TIM_EncoderInterfaceConfig(TIM3, TIM_EncoderMode_TI12, TIM_ICPolarity_BothEdge, TIM_ICPolarity_BothEdge); //TIM_ICPolarity_Rising上升沿捕获
  //TIM_EncoderInterfaceConfig(TIM3, TIM_EncoderMode_TI12, TIM_ICPolarity_Rising, TIM_ICPolarity_Rising); //TIM_ICPolarity_Rising上升沿捕获
  TIM_ICStructInit(&TIM_ICInitStructure);
  TIM_ICInitStructure.TIM_ICFilter = 6; //无滤波器
  TIM_ICInit(TIM3, &TIM_ICInitStructure);
  
 // Clear all pending interrupts
  TIM_ClearFlag(TIM3, TIM_FLAG_Update);
  TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);	  //使能中断
  TIM_Cmd(TIM3, ENABLE);  	   //使能定时器3
}

void TIM4_Encoder_Init(void)
{
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_ICInitTypeDef TIM_ICInitStructure;
  NVIC_InitTypeDef NVIC_InitStructure;
  
/* Encoder unit connected to TIM3, 4X mode */    
  GPIO_InitTypeDef GPIO_InitStructure;
  //NVIC_InitTypeDef NVIC_InitStructure;
  
  /* TIM3 clock source enable */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM4, ENABLE);
  /* Enable GPIOA, clock */
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
  
  GPIO_StructInit(&GPIO_InitStructure);
  /* Configure PA.06,07 as encoder input */
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_7;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOB, &GPIO_InitStructure);
  
  /* Enable the TIM3 Update Interrupt */
  NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);    //优先级组别
  NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
   
  /* Timer configuration in Encoder mode */
  TIM_DeInit(TIM4);
  TIM_TimeBaseStructInit(&TIM_TimeBaseStructure);
  
  TIM_TimeBaseStructure.TIM_Prescaler = 0x0;  // No prescaling 
  TIM_TimeBaseStructure.TIM_Period = 0xFFFF;  
  TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;  //设置时钟分频系数：不分频
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  //向上计数模式 
  TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);
 
  TIM_EncoderInterfaceConfig(TIM4, TIM_EncoderMode_TI12, TIM_ICPolarity_BothEdge, TIM_ICPolarity_BothEdge); //TIM_ICPolarity_Rising上升沿捕获
  TIM_ICStructInit(&TIM_ICInitStructure);
  TIM_ICInitStructure.TIM_ICFilter = 6; //无滤波器
  TIM_ICInit(TIM4, &TIM_ICInitStructure);
  
  // Clear all pending interrupts
  TIM_ClearFlag(TIM4, TIM_FLAG_Update);
  TIM_ITConfig(TIM4, TIM_IT_Update, ENABLE);	  //使能中断
 
  TIM_Cmd(TIM4, ENABLE);  	   //使能定时器3
}


void TIM3_External_Clock_CountingMode(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	  NVIC_InitTypeDef NVIC_InitStructure;
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7; //GPIO_Pin_7
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);    //优先级组别
    NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
		NVIC_Init(&NVIC_InitStructure);
	
    TIM_DeInit(TIM3); 
    TIM_TimeBaseStructure.TIM_Period = 0xFFFF; 
    TIM_TimeBaseStructure.TIM_Prescaler = 0x00; 
    TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1; /*定时器时钟(CK_INT)频率与数字滤波器(ETR,TIx)
                                                                            使用的采样频率之间的分频比为1*/
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up; 
    TIM_TimeBaseInit( TIM3, &TIM_TimeBaseStructure); // Time base configuration  
    
                             
	TIM_TIxExternalClockConfig(TIM3,TIM_TIxExternalCLK1Source_TI2,TIM_ICPolarity_Rising,0);//TIM_TIxExternalCLK1Source_TI2
	TIM_SetCounter(TIM3, 0);   // 清零计数器CNT
	TIM_Cmd(TIM3,ENABLE); 
}

void TIM4_External_Clock_CountingMode(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	  NVIC_InitTypeDef NVIC_InitStructure;
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM4, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7; 
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
	
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);    //优先级组别
    NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;
		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
		NVIC_Init(&NVIC_InitStructure);
    
    TIM_DeInit(TIM4); 
    TIM_TimeBaseStructure.TIM_Period = 0xFFFF; 
    TIM_TimeBaseStructure.TIM_Prescaler = 0x00; 
    TIM_TimeBaseStructure.TIM_ClockDivision = 0x0; /*定时器时钟(CK_INT)频率与数字滤波器(ETR,TIx)
                                                                            使用的采样频率之间的分频比为1*/
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up; 
    TIM_TimeBaseInit( TIM4, &TIM_TimeBaseStructure); // Time base configuration  

                             
    TIM_TIxExternalClockConfig(TIM4,TIM_TIxExternalCLK1Source_TI2,TIM_ICPolarity_Rising,0);
    TIM_SetCounter(TIM4, 0);   // 清零计数器CNT
    TIM_Cmd(TIM4,ENABLE); 
}



