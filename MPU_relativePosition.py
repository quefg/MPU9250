import time
import numpy as np
from smbus2 import SMBus
from vpython import vector, box, rate, scene, label

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
gravity = 9.81  # Gravity in m/s^2
threshold = 0.1  # Stationary threshold in m/s²
distance_update_interval = 5  # Calculate distance every 5 seconds

# Initial states
velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (vx, vy, vz)
position = np.array([0.0, 0.0, 0.0])  # Initial position (x, y, z)
yaw, pitch, roll = 0.0, 0.0, 0.0  # Initial orientation

# Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "3D Motion Tracking with MPU9250"
scene.range = 2

mpu_box = box(size=vector(0.5, 0.1, 0.1), color=vector(0, 1, 0))
distance_label = label(pos=vector(0, -1, 0), text="")
orientation_label = label(pos=vector(0, -1.5, 0), text="")

# I2C Functions
def read_i2c_word(bus, addr, reg):
    try:
        high = bus.read_byte_data(addr, reg)
        low = bus.read_byte_data(addr, reg + 1)
        value = (high << 8) | low
        return value - 65536 if value > 32768 else value
    except Exception as e:
        print(f"I2C Read Error: {e}")
        return 0

def read_accel(bus):
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0  # Scale for ±2g
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0
    return np.array([ax, ay, az])

def read_gyro(bus):
    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / 131.0  # Scale for ±250°/s
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / 131.0
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / 131.0
    return np.array([gx, gy, gz])

def setup_mpu(bus):
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up MPU9250

# Main Program
def run_tracking():
    global velocity, position, yaw, pitch, roll

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)

        last_distance_update = time.time()

        while True:
            rate(50)  # Update rate 50 Hz

            accel = read_accel(bus)
            gyro = read_gyro(bus)

            # Complementary filter for orientation
            pitch_accel = np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2)) * 180 / np.pi
            roll_accel = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)) * 180 / np.pi

            pitch = alpha * (pitch + gyro[0] * dt) + (1 - alpha) * pitch_accel
            roll = alpha * (roll + gyro[1] * dt) + (1 - alpha) * roll_accel
            yaw += gyro[2] * dt

            # Gravity compensation
            accel_corrected = accel * gravity
            accel_corrected[2] -= gravity  # Remove gravity component from Z-axis

            # Detect stationary and reset velocity
            if np.linalg.norm(accel_corrected) < threshold:
                velocity = np.array([0.0, 0.0, 0.0])

            # Update velocity and position
            velocity += accel_corrected * dt
            position += velocity * dt

            # Update visualization
            mpu_box.pos = vector(position[0], position[1], position[2])
            orientation_label.text = f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°"
            distance_label.text = f"Position (cm): X={position[0]*100:.2f}, Y={position[1]*100:.2f}, Z={position[2]*100:.2f}"

            # Calculate distance traveled every 5 seconds
            if time.time() - last_distance_update >= distance_update_interval:
                distance = np.linalg.norm(position)
                print(f"Distance traveled in last {distance_update_interval} seconds: {distance:.2f} cm")
                last_distance_update = time.time()

# Run tracking
run_tracking()
