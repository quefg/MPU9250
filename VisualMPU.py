import time
import numpy as np
from smbus2 import SMBus
from vpython import vector, box, rate, scene, cylinder

# I2C setup
I2C_BUS = 8  # Update with your I2C bus number
MPU9250_ADDR = 0x68

# MPU9250 Register Map
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B

# Constants
dt = 0.02  # Sampling period in seconds (50 Hz)
alpha = 0.98  # Complementary filter coefficient

# I2C Functions
def read_i2c_word(bus, addr, reg):
    """Read two bytes from I2C and combine into a signed word."""
    try:
        high = bus.read_byte_data(addr, reg)
        low = bus.read_byte_data(addr, reg + 1)
        value = (high << 8) | low
        return value - 65536 if value > 32768 else value
    except Exception as e:
        print(f"I2C Read Error: {e}")
        return 0

def read_accel_gyro(bus):
    """Read accelerometer and gyroscope data."""
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0  # Scale for ±2g
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0
    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / 131.0  # Scale for ±250°/s
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / 131.0
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / 131.0
    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def setup_mpu(bus):
    """Initialize MPU9250."""
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up MPU9250

# Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "MPU9250 3D Visualization"
scene.range = 2  # Adjust the view range

# Draw reference XYZ axes
x_axis = cylinder(pos=vector(0, 0, 0), axis=vector(2, 0, 0), radius=0.02, color=vector(1, 0, 0))
y_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 2, 0), radius=0.02, color=vector(0, 1, 0))
z_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 2), radius=0.02, color=vector(0, 0, 1))

mpu_box = box(
    size=vector(1.004, 0.606, 0.118),  # Dimensions in inches
    color=vector(0, 1, 0)
)

# Initial orientation (calibration step)
initial_orientation = np.array([0.0, 0.0, 0.0])  # Reference orientation (calibrated)
gyro_angle = np.array([0.0, 0.0, 0.0])  # Tracks rotation angles
orientation = np.array([0.0, 0.0, 0.0])  # Tracks filtered orientation

def calibrate_sensor(bus):
    """Calibrate the sensor to get the initial orientation."""
    global initial_orientation
    print("Calibrating sensor... Place it upright on a flat surface.")
    time.sleep(3)
    samples = []
    for _ in range(50):
        accel, _ = read_accel_gyro(bus)
        samples.append(accel)
    avg_accel = np.mean(samples, axis=0)
    initial_orientation[0] = np.arctan2(avg_accel[1], avg_accel[2]) * 180 / np.pi
    initial_orientation[1] = np.arctan2(-avg_accel[0], np.sqrt(avg_accel[1]**2 + avg_accel[2]**2)) * 180 / np.pi
    print(f"Initial orientation: {initial_orientation}")

# Main Program
def run_visualization():
    global gyro_angle, orientation

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)
        calibrate_sensor(bus)

        while True:
            rate(50)  # Update rate 50 Hz
            accel, gyro = read_accel_gyro(bus)

            # Normalize accelerometer data for tilt estimation
            accel_norm = np.linalg.norm(accel)
            if accel_norm != 0:
                accel = accel / accel_norm

            # Calculate tilt angles from accelerometer
            accel_pitch = np.arctan2(accel[1], accel[2]) * 180 / np.pi
            accel_roll = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)) * 180 / np.pi

            # Integrate gyroscope data for angles
            gyro_angle[0] += gyro[0] * dt
            gyro_angle[1] += gyro[1] * dt
            gyro_angle[2] += gyro[2] * dt

            # Complementary filter to combine accelerometer and gyroscope data
            orientation[0] = alpha * (orientation[0] + gyro[0] * dt) + (1 - alpha) * (accel_roll - initial_orientation[0])
            orientation[1] = alpha * (orientation[1] + gyro[1] * dt) + (1 - alpha) * (accel_pitch - initial_orientation[1])
            orientation[2] = gyro_angle[2]  # Use only gyroscope for yaw (no accel info for yaw)

            # Update 3D visualization
            mpu_box.axis = vector(
                np.sin(np.radians(orientation[0])),
                np.sin(np.radians(orientation[1])),
                np.cos(np.radians(orientation[2]))
            )

# Run the visualization
run_visualization()
