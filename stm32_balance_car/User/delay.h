#ifndef _DELAY__H_
#define _DELAY__H_

#include "stm32f10x.h"

void delay_us(uint32_t n);
void delay_ms(uint32_t n);
void get_ms(unsigned long *time);
#endif
