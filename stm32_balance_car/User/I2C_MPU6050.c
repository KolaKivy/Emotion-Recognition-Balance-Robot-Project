/*********************************************************************************
**        MPU6050 姿态传感器I2C总线
**	   		   PB8 - I2C1_SCL  
**			     PB9 - I2C1_SDA  	     
**********************************************************************************/
#include "I2C_MPU6050.h"
#include "delay.h"

static void I2C_GPIO_Config(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	
	/* 使能与 I2C1 有关的时钟 */
	RCC_APB2PeriphClockCmd  (RCC_APB2Periph_GPIOB,ENABLE ); 
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1,ENABLE);  

  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO,ENABLE);
	GPIO_PinRemapConfig(GPIO_Remap_I2C1, ENABLE);   //使用重映射功能

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 | GPIO_Pin_9; 
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz; 
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_OD; 
	GPIO_Init(GPIOB, &GPIO_InitStructure); 
}

static void I2C_Mode_Config(void)
{
	 /* Initialize the I2C1 according to the I2C_InitStructure members */ 
	I2C_InitTypeDef I2C_InitStructure; 
	 
	  /* I2C 配置 */
	I2C_InitStructure.I2C_Mode = I2C_Mode_I2C ; 
	I2C_InitStructure.I2C_DutyCycle = I2C_DutyCycle_2; 
	I2C_InitStructure.I2C_OwnAddress1 = MPU6050_SlaveAddress; 
	I2C_InitStructure.I2C_Ack = I2C_Ack_Enable; 
	I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit; 
	I2C_InitStructure.I2C_ClockSpeed = 400000; 


	/* I2C1 初始化 */
	I2C_Init(I2C1, &I2C_InitStructure);	   
	
	/* 使能 I2C1 */
	I2C_Cmd  (I2C1,ENABLE); 
	/*允许应答模式*/
	I2C_AcknowledgeConfig(I2C1, ENABLE);   
}

void I2C_MPU6050_Init(void)
{	   
 	I2C_GPIO_Config();
	I2C_Mode_Config();
}  

	
void I2C_ByteWrite(uint8_t REG_Address,uint8_t REG_data)
{

I2C_GenerateSTART(I2C1,ENABLE);

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_MODE_SELECT));

I2C_Send7bitAddress(I2C1,MPU6050_SlaveAddress,I2C_Direction_Transmitter);

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));

I2C_SendData(I2C1,REG_Address);

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_BYTE_TRANSMITTED));

I2C_SendData(I2C1,REG_data);

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_BYTE_TRANSMITTED));

I2C_GenerateSTOP(I2C1,ENABLE);

}



uint8_t I2C_ByteRead(uint8_t REG_Address)
{
uint8_t REG_data;

while(I2C_GetFlagStatus(I2C1,I2C_FLAG_BUSY));

I2C_GenerateSTART(I2C1,ENABLE);//起始信号

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_MODE_SELECT));

I2C_Send7bitAddress(I2C1,MPU6050_SlaveAddress,I2C_Direction_Transmitter);//发送设备地址+写信号

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));//

I2C_Cmd(I2C1,ENABLE);

I2C_SendData(I2C1,REG_Address);//发送存储单元地址，从0开始

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_BYTE_TRANSMITTED));

I2C_GenerateSTART(I2C1,ENABLE);//起始信号

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_MODE_SELECT));

I2C_Send7bitAddress(I2C1,MPU6050_SlaveAddress,I2C_Direction_Receiver);//发送设备地址+读信号

while(!I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED));

I2C_AcknowledgeConfig(I2C1,DISABLE);

I2C_GenerateSTOP(I2C1,ENABLE);

while(!(I2C_CheckEvent(I2C1,I2C_EVENT_MASTER_BYTE_RECEIVED)));

REG_data=I2C_ReceiveData(I2C1);//读出寄存器数据

return REG_data;

}

void InitMPU6050(void)
{
	I2C_ByteWrite(MPU6050_PWR_MGMT_1,0x00);//解除休眠状态
	delay_ms(100);
	I2C_ByteWrite(MPU6050_SMPLRT_DIV,0x07);
	delay_ms(50);
	I2C_ByteWrite(MPU6050_CONFIG,0x06);
	delay_ms(50);
	I2C_ByteWrite(MPU6050_GYRO_CONFIG,0x00);
	delay_ms(50);
	I2C_ByteWrite(MPU6050_ACCEL_CONFIG,0x00);
	delay_ms(50); 
}

unsigned int GetData(unsigned char REG_Address)
{
	char H,L;
	H=I2C_ByteRead(REG_Address);
	L=I2C_ByteRead(REG_Address+1);
	return (H<<8)+L;   //数据合成
}








