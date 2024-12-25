#ifndef __I2C_H
#define __I2C_H
#include "stm32f10x.h"


#define true 1
#define false 0 
#define   bool  uint8_t

#define TRUE  0
#define FALSE -1

bool i2cWriteBuffer(uint8_t addr_, uint8_t reg_, uint8_t len_, uint8_t *data);
bool i2cWrite(uint8_t addr_, uint8_t reg_, uint8_t data);
bool i2cRead(uint8_t addr_, uint8_t reg_, uint8_t len, uint8_t* buf);
void i2cInit(void);
uint16_t i2cGetErrorCounter(void);

int8_t i2cwrite(uint8_t addr, uint8_t reg, uint8_t len, uint8_t * data);
int8_t i2cread(uint8_t addr, uint8_t reg, uint8_t len, uint8_t *buf);

#endif
