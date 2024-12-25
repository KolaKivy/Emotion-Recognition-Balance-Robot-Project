/**************************************************************
 MPU6050 驱动函数
***************************************************************/
#include "mpu6050.h"
#include "i2c.h"
#include "inv_mpu.h"
#include "inv_mpu_dmp_motion_driver.h"
#include "Serial.h"
#include "math.h"

static signed char gyro_orientation[9] = {-1, 0, 0,
                                           0,-1, 0,
                                           0, 0, 1};
float q0=1.0f,q1=0.0f,q2=0.0f,q3=0.0f;
float Pitch,Roll,Yaw;
unsigned long sensor_timestamp;
short gyro[3], accel[3], sensors;
unsigned char more;
long quat[4];

// 自己定义了一个小的延时																					 
void delay_xms(uint16_t time)
{    
   uint16_t i=0;  
   while(time--)
   {
      i=12000;  
      while(i--) ;    
   }
}
 
void MPU6050_Init(void)
{
	int result=0;
  i2cInit();							      //IIC初始化 用于挂靠在总线上的设备使用
	delay_xms(10);						   //延时10ms
	result=mpu_init();
	if(!result)
	{	 		 
	
		Serial_Printf("mpu initialization complete......\n ");		//mpu initialization complete	 	  

		if(!mpu_set_sensors(INV_XYZ_GYRO | INV_XYZ_ACCEL))		//mpu_set_sensor
			Serial_Printf("mpu_set_sensor complete ......\n");
		else
			Serial_Printf("mpu_set_sensor come across error ......\n");

		if(!mpu_configure_fifo(INV_XYZ_GYRO | INV_XYZ_ACCEL))	//mpu_configure_fifo
			Serial_Printf("mpu_configure_fifo complete ......\n");
		else
			Serial_Printf("mpu_configure_fifo come across error ......\n");

		if(!mpu_set_sample_rate(DEFAULT_MPU_HZ))	   	  		//mpu_set_sample_rate
		 Serial_Printf("mpu_set_sample_rate complete ......\n");
		else
		 	Serial_Printf("mpu_set_sample_rate error ......\n");

		if(!dmp_load_motion_driver_firmware())   	  			//dmp_load_motion_driver_firmvare
			Serial_Printf("dmp_load_motion_driver_firmware complete ......\n");
		else
			Serial_Printf("dmp_load_motion_driver_firmware come across error ......\n");

		if(!dmp_set_orientation(inv_orientation_matrix_to_scalar(gyro_orientation))) 	  //dmp_set_orientation
		 	Serial_Printf("dmp_set_orientation complete ......\n");
		else
		 	Serial_Printf("dmp_set_orientation come across error ......\n");

		if(!dmp_enable_feature(DMP_FEATURE_6X_LP_QUAT | DMP_FEATURE_TAP |
		    DMP_FEATURE_ANDROID_ORIENT | DMP_FEATURE_SEND_RAW_ACCEL | DMP_FEATURE_SEND_CAL_GYRO |
		    DMP_FEATURE_GYRO_CAL))		   	 					 //dmp_enable_feature
		 	Serial_Printf("dmp_enable_feature complete ......\n");
		else
		 	Serial_Printf("dmp_enable_feature come across error ......\n");

		if(!dmp_set_fifo_rate(DEFAULT_MPU_HZ))   	 			 //dmp_set_fifo_rate
		 	Serial_Printf("dmp_set_fifo_rate complete ......\n");
		else
		 	Serial_Printf("dmp_set_fifo_rate come across error ......\n");

		run_self_test();		//自检

		if(!mpu_set_dmp_state(1))
		 	Serial_Printf("mpu_set_dmp_state complete ......\n");
		else
		 	Serial_Printf("mpu_set_dmp_state come across error ......\n");
	}
	else												                 
	 {
	 // MPU6050 异常指示灯
	 GPIO_ResetBits(GPIOC, GPIO_Pin_13);				
	 while(1);
	 }
	 
}

// MPU6050 四元数转为姿态角
void MPU6050_Pose(void)
{
	dmp_read_fifo(gyro, accel, quat, &sensor_timestamp, &sensors,&more);	 

	if(sensors & INV_WXYZ_QUAT )
	{
		q0 = quat[0] / q30;	
		q1 = quat[1] / q30;
		q2 = quat[2] / q30;
		q3 = quat[3] / q30;

		Pitch = asin(-2 * q1 * q3 + 2 * q0* q2)* 57.3;	// pitch
		Roll  = atan2(2 * q2 * q3 + 2 * q0 * q1, -2 * q1 * q1 - 2 * q2* q2 + 1)*57.3;	// roll
		Yaw   = atan2(2*(q1*q2 + q0*q3),q0*q0+q1*q1-q2*q2-q3*q3) * 57.3;	//yaw	
	}
}
