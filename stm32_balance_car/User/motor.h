
#ifndef __MOTOR_H
#define __MOTOR_H

#include "stm32f10x.h"

void TIM2_PWM_Init(void);

void MOTOR_Init(void);          // 电机初始化函数
void TIM3_Encoder_Init(void);
void TIM4_Encoder_Init(void);
void TIM2_PWM_CHANGE(uint16_t CCR3,uint16_t CCR4);
void TIM4_External_Clock_CountingMode(void);
void TIM3_External_Clock_CountingMode(void);
void exitcount1_init(void);


extern int leftcount;      
extern int rightcount;     

#endif
