#include "stm32f10x.h"
#include "delay.h"

void delay_us(uint32_t n)
{
	uint8_t j;
	while(n--)
	for(j=0;j<10;j++);
}
void delay_ms(uint32_t n)
{
	while(n--)
	delay_us(1000);
}
void get_ms(unsigned long *time)
{

}
