/*********************************************************************************
**      MPU6050 姿态传感器I2C管脚定义说明
**           PB8 - I2C1_SCL  
**			     PB9 - I2C1_SDA  
**********************************************************************************/
#include "I2C.h"
 //  I2C 设备管脚定义，采用软件模拟驱动
#define I2CSCL_H         GPIOB->BSRR = GPIO_Pin_8
#define I2CSCL_L         GPIOB->BRR  = GPIO_Pin_8 

#define I2CSDA_H         GPIOB->BSRR = GPIO_Pin_9
#define I2CSDA_L         GPIOB->BRR  = GPIO_Pin_9 

#define I2CSCL_read      GPIOB->IDR  & GPIO_Pin_8
#define I2CSDA_read      GPIOB->IDR  & GPIO_Pin_9 

// I2C专用延时函数
static void MPU_I2C_delay(void)
{
    volatile int i = 7;
    while (i)
        i--;
}
 // I2C 起始信号
static bool I2C_Start(void)
{
    I2CSDA_H;
    I2CSCL_H;
    MPU_I2C_delay();
    if (!I2CSDA_read)
        return false;
    I2CSDA_L;
    MPU_I2C_delay();
    if (I2CSDA_read)
        return false;
    I2CSDA_L;
    MPU_I2C_delay();
    return true;
}
// I2C停止信号
static void I2C_Stop(void)
{
    I2CSCL_L;
    MPU_I2C_delay();
    I2CSDA_L;
    MPU_I2C_delay();
    I2CSCL_H;
    MPU_I2C_delay();
    I2CSDA_H;
    MPU_I2C_delay();
}
// I2C应答信号
static void I2C_Ack(void)
{
    I2CSCL_L;
    MPU_I2C_delay();
    I2CSDA_L;
    MPU_I2C_delay();
    I2CSCL_H;
    MPU_I2C_delay();
    I2CSCL_L;
    MPU_I2C_delay();
}
// I2C 非应答信号
static void I2C_NoAck(void)
{
    I2CSCL_L;
    MPU_I2C_delay();
    I2CSDA_H;
    MPU_I2C_delay();
    I2CSCL_H;
    MPU_I2C_delay();
    I2CSCL_L;
    MPU_I2C_delay();
}
// I2C等待应答信号
static bool I2C_WaitAck(void)
{
    I2CSCL_L;
    MPU_I2C_delay();
    I2CSDA_H;
    MPU_I2C_delay();
    I2CSCL_H;
    MPU_I2C_delay();
    if (I2CSDA_read) {
        I2CSCL_L;
        return false;
    }
    I2CSCL_L;
    return true;
}
// I2C 发送一个字节
static void I2C_SendByte(uint8_t byte)
{
    uint8_t i = 8;
    while (i--) {
        I2CSCL_L;
        MPU_I2C_delay();
        if (byte & 0x80)
            I2CSDA_H;
        else
            I2CSDA_L;
        byte <<= 1;
        MPU_I2C_delay();
        I2CSCL_H;
        MPU_I2C_delay();
    }
    I2CSCL_L;
}
// I2C 读取一个字节
static uint8_t I2C_ReceiveByte(void)
{
    uint8_t i = 8;
    uint8_t byte = 0;

    I2CSDA_H;
    while (i--) {
        byte <<= 1;
        I2CSCL_L;
        MPU_I2C_delay();
        I2CSCL_H;
        MPU_I2C_delay();
        if (I2CSDA_read) {
            byte |= 0x01;
        }
    }
    I2CSCL_L;
    return byte;
}
//  初始化函数
void i2cInit(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
	  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB , ENABLE);    
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 | GPIO_Pin_9;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
}
// I2C 写入缓冲区
bool i2cWriteBuffer(uint8_t addr, uint8_t reg, uint8_t len, uint8_t * data)
{
    int i;
    if (!I2C_Start())
        return false;
    I2C_SendByte(addr << 1 | I2C_Direction_Transmitter);
    if (!I2C_WaitAck()) {
        I2C_Stop();
        return false;
    }
    I2C_SendByte(reg);
    I2C_WaitAck();
    for (i = 0; i < len; i++) {
        I2C_SendByte(data[i]);
        if (!I2C_WaitAck()) {
            I2C_Stop();
            return false;
        }
    }
    I2C_Stop();
    return true;
}
// I2C写入字符
int8_t i2cwrite(uint8_t addr, uint8_t reg, uint8_t len, uint8_t * data)
{
	if(i2cWriteBuffer(addr,reg,len,data))
	{
		return TRUE;
	}
	else
	{
		return FALSE;
	}
}
// I2C  读函数
int8_t i2cread(uint8_t addr, uint8_t reg, uint8_t len, uint8_t *buf)
{
	if(i2cRead(addr,reg,len,buf))
	{
		return TRUE;
	}
	else
	{
		return FALSE;
	}
}
// I2C 写函数
bool i2cWrite(uint8_t addr, uint8_t reg, uint8_t data)
{
    if (!I2C_Start())
        return false;
    I2C_SendByte(addr << 1 | I2C_Direction_Transmitter);
    if (!I2C_WaitAck()) {
        I2C_Stop();
        return false;
    }
    I2C_SendByte(reg);
    I2C_WaitAck();
    I2C_SendByte(data);
    I2C_WaitAck();
    I2C_Stop();
    return true;
}

bool i2cRead(uint8_t addr, uint8_t reg, uint8_t len, uint8_t *buf)
{
    if (!I2C_Start())
        return false;
    I2C_SendByte(addr << 1 | I2C_Direction_Transmitter);
    if (!I2C_WaitAck()) {
        I2C_Stop();
        return false;
    }
    I2C_SendByte(reg);
    I2C_WaitAck();
    I2C_Start();
    I2C_SendByte(addr << 1 | I2C_Direction_Receiver);
    I2C_WaitAck();
    while (len) {
        *buf = I2C_ReceiveByte();
        if (len == 1)
            I2C_NoAck();
        else
            I2C_Ack();
        buf++;
        len--;
    }
    I2C_Stop();
    return true;
}
// I2C  错误信号
uint16_t i2cGetErrorCounter(void)
{
    return 0;
}
