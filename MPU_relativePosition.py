import time
import numpy as np
from smbus2 import SMBus
from vpython import vector, box, rate, scene, label, cylinder

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
gyro_scale = 131.0  # Scale for ±250°/s
threshold = 0.1  # Threshold for ignoring small accelerations

# Initial states
velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (m/s)
position = np.array([0.0, 0.0, 0.0])  # Position relative to initial (m)
initial_position = np.array([0.0, 0.0, 0.0])  # Initial position
previous_accel = np.array([0.0, 0.0, 0.0])  # Previous acceleration for filtering
orientation = np.array([0.0, 0.0, 0.0])  # Pitch, Roll, Yaw in degrees

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

def read_accel_gyro(bus):
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0
    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / gyro_scale
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / gyro_scale
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / gyro_scale
    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def setup_mpu(bus):
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)

def apply_low_pass_filter(accel, previous_accel, alpha=0.98):
    return alpha * previous_accel + (1 - alpha) * accel

def reset_velocity_if_stationary(accel, velocity, threshold):
    if np.linalg.norm(accel) < threshold:
        return np.array([0.0, 0.0, 0.0])
    return velocity

def update_orientation(gyro, orientation, dt):
    return orientation + gyro * dt  # Integrate angular velocity

def get_rotation_matrix(pitch, roll, yaw):
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

# Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "MPU9250 3D Visualization with Relative Position"
scene.range = 2

x_axis = cylinder(pos=vector(0, 0, 0), axis=vector(2, 0, 0), radius=0.02, color=vector(1, 0, 0))
y_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 2, 0), radius=0.02, color=vector(0, 1, 0))
z_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 2), radius=0.02, color=vector(0, 0, 1))

mpu_box = box(size=vector(1.004, 0.606, 0.118), color=vector(0, 1, 0))

angle_label = label(pos=vector(0, -2, 0), text="Angles: ")
position_label = label(pos=vector(0, -2.5, 0), text="Relative Position: ")

# Main Program
def run_visualization():
    global velocity, position, initial_position, previous_accel, orientation

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)

        while True:
            rate(50)  # 50 Hz update rate
            accel, gyro = read_accel_gyro(bus)

            # Apply low-pass filter
            accel = apply_low_pass_filter(accel, previous_accel)
            previous_accel = accel.copy()

            # Convert acceleration to m/s² and remove gravity
            accel_corrected = accel * gravity
            accel_corrected[2] -= gravity

            # Reset velocity if stationary
            velocity = reset_velocity_if_stationary(accel_corrected, velocity, threshold)

            # Update velocity and position
            velocity += accel_corrected * dt
            position += velocity * dt

            # Update orientation
            orientation = update_orientation(gyro, orientation, dt)

            # Compute rotation matrix and update 3D box
            R = get_rotation_matrix(orientation[0], orientation[1], orientation[2])
            mpu_box.axis = vector(R[0, 2], R[1, 2], R[2, 2])
            mpu_box.up = vector(R[0, 1], R[1, 1], R[2, 1])

            # Update labels
            angle_label.text = f"Angles (Pitch, Roll, Yaw): {np.round(orientation, 2)}°"
            position_label.text = f"Relative Position (X, Y, Z): {np.round(position, 2)} m"

# Run the visualization
run_visualization()
