import time
import numpy as np
from smbus2 import SMBus

# Constants for I2C and MPU9250 setup
I2C_BUS = 8
MPU9250_ADDR = 0x68  

# MPU9250 Register Addresses
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
PWR_MGMT_1 = 0x6B  # Power management registers

# Constants for calculations
gravity = 9.81  # Acceleration due to gravity in m/s²
noise_threshold = 0.01  # Threshold to filter out noise in acceleration

# Initial states
accel_bias = [0.0, 0.0, 0.0]  # Bias correction for accelerometer

def read_i2c_word(bus, addr, reg):
    """Read two bytes from I2C and combine into a signed word."""
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    return value - 65536 if value > 32768 else value

def calibrate_sensor(bus, samples=100):
    """Calibrate the accelerometer to calculate biases."""
    global accel_bias
    ax_total, ay_total, az_total = 0.0, 0.0, 0.0
    for _ in range(samples):
        ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0
        ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_YOUT_H) / 16384.0
        az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_ZOUT_H) / 16384.0
        ax_total += ax
        ay_total += ay
        az_total += az
        time.sleep(0.01)
    accel_bias = [ax_total / samples, ay_total / samples, az_total / samples - gravity]  # Subtract gravity for Z-axis

def read_accel(bus):
    """Read accelerometer data."""
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0 - accel_bias[0]
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_YOUT_H ) / 16384.0 - accel_bias[1]
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_ZOUT_H ) / 16384.0 - accel_bias[2]

    return ax, ay, az

def cancel_gravity(ax, ay, az, pitch, roll):
    """Cancel the effect of gravity from raw acceleration readings."""
    g_x = gravity * np.sin(np.radians(pitch))
    g_y = -gravity * np.sin(np.radians(roll)) * np.cos(np.radians(pitch))
    g_z = gravity * np.cos(np.radians(roll)) * np.cos(np.radians(pitch))

    ax_corrected = ax - g_x
    ay_corrected = ay - g_y
    az_corrected = az - g_z

    return ax_corrected, ay_corrected, az_corrected

def read_acceleration():
    with SMBus(I2C_BUS) as bus:
        bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up sensor
        calibrate_sensor(bus)  # Perform calibration

        while True:
            try:
                ax, ay, az = read_accel(bus)

                # Cancel gravitational effect to correct the raw acceleration
                ax_corrected, ay_corrected, az_corrected = cancel_gravity(ax, ay, az, pitch=0, roll=0)  

                # Apply threshold to filter out noise
                ax_corrected = ax_corrected if abs(ax_corrected) > noise_threshold else 0.0
                ay_corrected = ay_corrected if abs(ay_corrected) > noise_threshold else 0.0
                az_corrected = az_corrected if abs(az_corrected) > noise_threshold else 0.0

                # Print corrected acceleration
                print(f"Acceleration: X={ax_corrected:.2f}, Y={ay_corrected:.2f}, Z={az_corrected:.2f} m/s²")

                time.sleep(0.1)

            except OSError as e:
                if e.errno == 6:
                    print("I2C device not found. Please check the connection.")
                    break
                else:
                    raise

# Run the acceleration reading
read_acceleration()
