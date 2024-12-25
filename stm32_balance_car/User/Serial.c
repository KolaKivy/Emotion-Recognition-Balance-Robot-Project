#include "Serial.h"
#include <stdarg.h>
#include "misc.h"

//串口接收DMA缓存
uint8_t Uart_Rx[UART_RX_LEN] = {0};

//串口调试初始化
void Serial_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStruture;
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1,ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);

	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA,&GPIO_InitStructure);
	
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	USART_InitStruture.USART_BaudRate = 115200;
	USART_InitStruture.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStruture.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	USART_InitStruture.USART_Parity = USART_Parity_No;
	USART_InitStruture.USART_StopBits = USART_StopBits_1;
	USART_InitStruture.USART_WordLength = USART_WordLength_8b;
	USART_Init(USART1,&USART_InitStruture);
	
	USART_Cmd(USART1,ENABLE);
}

// 蓝牙模块初始化，使用的是UART3
void bluetooth_Init(void)
{
  //定义串口初始化结构体
	GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;	
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3 , ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB	, ENABLE);
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);    
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
  	
	NVIC_Configurationusart3(); 
	USART_InitStructure.USART_BaudRate = 9600;//波特率9600
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;//8位数据
	USART_InitStructure.USART_StopBits = USART_StopBits_1;//1个停止位
	USART_InitStructure.USART_Parity = USART_Parity_No ;//无校验位
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;//禁用RTSCTS硬件流控制
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;	//使能发送接收
  

	USART_Init(USART3, &USART_InitStructure); 
	USART_ITConfig(USART3, USART_IT_RXNE, ENABLE);//使能接收中断
	USART_ITConfig(USART3,USART_IT_IDLE,ENABLE);   //使能串口外设空闲中断
	
	USART_Cmd(USART3, ENABLE);
	
}

void NVIC_Configurationusart3(void)
{
   NVIC_InitTypeDef NVIC_InitStructure;
/* Configure the NVIC Preemption Priority Bits */  
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_3);
	
	/* Enable the USARTy Interrupt */
	NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;	 
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 4;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);
}

void Serial_SendByte(uint8_t Byte)
{
	USART_SendData(USART1,Byte);
	while(USART_GetFlagStatus(USART1,USART_FLAG_TXE) == RESET);
	
}


int fputc(int ch,FILE *f)	
{
	Serial_SendByte(ch);
	return ch;
}



// 蓝牙模块发送一个字节
void USART3_Send_Byte(unsigned char byte)   
{
        USART_SendData(USART3, byte);        
        while( USART_GetFlagStatus(USART3,USART_FLAG_TC)!= SET);         
}

void UART3_Send_Char(char *s)
{
	char *p;
	p=s;
	while(*p != '\0')
	{
		USART3_Send_Byte(*p);
		p++;
	}	
}


void Serial_SendString(char *String)
{
	uint8_t i;
	for (i = 0;String[i] != '\0';i++)
	{
		Serial_SendByte(String[i]);
	}
}

void Serial_Printf(char *format, ...)
{
	char String[100];
	va_list arg;
	va_start(arg,format);
	vsprintf(String,format,arg);
	va_end(arg);
	Serial_SendString(String);
}
