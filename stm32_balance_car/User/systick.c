#include "SysTick.h"

static __IO uint32_t TimingDelay;

void SysTick_Init(void)		  //1ms定时时钟
{
	if (SysTick_Config(SystemCoreClock / 200))	// ST3.5.0库版本   1s/x= a ms  
 	{ 
		/* Capture error */ 
		while (1);
	}
		// 关闭滴答定时器  
	SysTick->CTRL &= ~ SysTick_CTRL_ENABLE_Msk;
}

void Delay_us(__IO uint32_t nTime)
{ 
	TimingDelay = nTime;	

	// 使能滴答定时器  
	SysTick->CTRL |=  SysTick_CTRL_ENABLE_Msk;

	while(TimingDelay != 0);
}
 
void TimingDelay_Decrement(void)
{
	if (TimingDelay != 0x00)
	{ 
	TimingDelay--;
	}
}
