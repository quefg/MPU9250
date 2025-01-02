import time
import numpy as np
from smbus2 import SMBus
from vpython import box, vector, rate, scene

# I2C setup
I2C_BUS = 1  # Usually 1 for Raspberry Pi and Ubuntu
MPU9250_ADDR = 0x68  # MPU9250 I2C address

# MPU9250 Register Map
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B

# Constants
dt = 0.02  # Sampling period (adjust based on your sensor data rate)
gravity = 9.81  # Earth's gravity (m/s^2)

# Initialize position, velocity, and acceleration
position = np.array([0.0, 0.0, 0.0])
velocity = np.array([0.0, 0.0, 0.0])

# Initialize 3D representation
scene.background = vector(0.2, 0.2, 0.2)
scene.center = vector(0, 0, 0)
mpu_box = box(size=vector(1, 1, 1), color=vector(0, 1, 0))

# Initial orientation (calibration step)
initial_accel = None

# I2C functions
def read_i2c_word(bus, addr, reg):
    """Read two bytes from I2C and combine them into a signed word."""
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    if value > 32768:  # Handle negative values
        value -= 65536
    return value

def read_accel_gyro(bus):
    """Read accelerometer and gyroscope data from MPU9250."""
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0  # Scale for ±2g
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0
    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / 131.0  # Scale for ±250°/s
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / 131.0
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / 131.0
    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def setup_mpu(bus):
    """Initialize the MPU9250."""
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up MPU9250
    bus.write_byte_data(MPU9250_ADDR, ACCEL_CONFIG, 0x00)  # ±2g
    bus.write_byte_data(MPU9250_ADDR, GYRO_CONFIG, 0x00)  # ±250°/s

# Calibration
def calibrate_sensor(bus):
    """Calibrate sensor to set the initial orientation."""
    global initial_accel
    print("Calibrating sensor... Hold the MPU upright.")
    time.sleep(3)
    samples = []
    for _ in range(50):
        accel, _ = read_accel_gyro(bus)
        samples.append(accel)
    initial_accel = np.mean(samples, axis=0)
    print(f"Initial acceleration calibrated: {initial_accel}")

def remove_gravity(accel):
    """Remove gravity component from the acceleration data."""
    return accel - initial_accel

# Main program
with SMBus(I2C_BUS) as bus:
    setup_mpu(bus)
    calibrate_sensor(bus)

    while True:
        rate(1 / dt)  # Control refresh rate

        # Read data from sensor
        accel, gyro = read_accel_gyro(bus)

        # Remove gravity
        linear_accel = remove_gravity(accel)

        # Calculate velocity (integral of acceleration)
        velocity += linear_accel * dt

        # Calculate position (integral of velocity)
        position += velocity * dt

        # Update 3D visualization
        mpu_box.pos = vector(position[0], position[1], position[2])
        mpu_box.axis = vector(gyro[0], gyro[1], gyro[2])  # Orient based on gyroscope data
        