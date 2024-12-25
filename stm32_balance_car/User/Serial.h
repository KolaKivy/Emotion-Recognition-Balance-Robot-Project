#ifndef __SERIAL_H
#define	__SERIAL_H

#include "stm32f10x.h"
#include <stdio.h>

#define UART_RX_LEN		128
extern uint8_t Uart_Rx[UART_RX_LEN];

void Serial_Init(void);
int  fputc(int ch, FILE *f);
void bluetooth_Init(void);
void NVIC_Configurationusart3(void);
void Serial_Printf(char *format, ...);
void UART3_Send_Char(char *s);

#endif
