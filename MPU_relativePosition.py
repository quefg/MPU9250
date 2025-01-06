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
gravity = 9.81  # Gravity in m/s²
velocity_threshold = 0.05  # Threshold to detect stationary motion (cm/s)
noise_threshold = 0.02  # Ignore small accelerometer values (g)
prev_accel = np.array([0.0, 0.0, 0.0])  # For low-pass filtering

# Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "MPU9250 3D Visualization"
scene.range = 2

# 3D axes
x_axis = cylinder(pos=vector(0, 0, 0), axis=vector(2, 0, 0), radius=0.02, color=vector(1, 0, 0))
y_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 2, 0), radius=0.02, color=vector(0, 1, 0))
z_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 2), radius=0.02, color=vector(0, 0, 1))
mpu_box = box(size=vector(1.004, 0.606, 0.118), color=vector(0, 1, 0))

# Labels
angle_label = label(pos=vector(0, -2, 0), text="Angles: ")
distance_label = label(pos=vector(0, -3.5, 0), text="Distance: X=0.00 cm, Y=0.00 cm, Z=0.00 cm")
accel_label_x = label(pos=vector(-1.5, -4, 0), text="Accel X: 0.00 m/s²", color=vector(1, 0, 0))
accel_label_y = label(pos=vector(0, -4, 0), text="Accel Y: 0.00 m/s²", color=vector(0, 1, 0))
accel_label_z = label(pos=vector(1.5, -4, 0), text="Accel Z: 0.00 m/s²", color=vector(0, 0, 1))

# Initialize variables
velocity = np.array([0.0, 0.0, 0.0])  # Velocity in m/s
distance = np.array([0.0, 0.0, 0.0])  # Distance in cm
orientation = np.array([0.0, 0.0, 0.0])  # Orientation (pitch, roll, yaw)
prev_accel = np.array([0.0, 0.0, 0.0])

# Functions for filtering
def low_pass_filter(accel, prev_accel, alpha=0.5):
    return alpha * accel + (1 - alpha) * prev_accel

def high_pass_filter(accel):
    accel[2] -= 1.0  # Remove gravity
    return accel

def reset_velocity_if_stationary(gyro):
    global velocity
    if np.linalg.norm(gyro) < velocity_threshold:
        velocity[:] = 0.0

def get_rotation_matrix(pitch, roll, yaw):
    pitch, roll, yaw = np.radians([pitch, roll, yaw])
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

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
    try:
        ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0
        ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
        az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0

        gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / 131.0
        gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / 131.0
        gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / 131.0

        return np.array([ax, ay, az]), np.array([gx, gy, gz])
    except Exception as e:
        print(f"I2C Read Error: {e}")
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

# Main loop
def run_visualization():
    global velocity, distance, prev_accel, orientation

    with SMBus(I2C_BUS) as bus:
        bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up sensor

        while True:
            rate(50)
            accel, gyro = read_accel_gyro(bus)

            # Filter acceleration
            accel = low_pass_filter(accel, prev_accel)
            prev_accel = accel
            accel_no_gravity = high_pass_filter(accel)
            accel_no_gravity[np.abs(accel_no_gravity) < noise_threshold] = 0

            # Reset velocity if stationary
            reset_velocity_if_stationary(gyro)

            # Update velocity and distance
            velocity += accel_no_gravity * gravity * dt
            distance += velocity * dt * 100  # Convert to cm

            # Calculate tilt angles
            pitch = np.arctan2(accel[1], accel[2]) * 180 / np.pi
            roll = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)) * 180 / np.pi
            orientation[0] = alpha * (orientation[0] + gyro[0] * dt) + (1 - alpha) * (roll)
            orientation[1] = alpha * (orientation[1] + gyro[1] * dt) + (1 - alpha) * (pitch)
            orientation[2] += gyro[2] * dt  # Yaw

            # Update 3D box rotation
            R = get_rotation_matrix(orientation[1], orientation[0], orientation[2])
            mpu_box.axis = vector(R[0, 2], R[1, 2], R[2, 2])
            mpu_box.up = vector(R[0, 1], R[1, 1], R[2, 1])

            # Update labels
            angle_label.text = f"Angles (Pitch, Roll, Yaw): {np.round(orientation, 2)}"
            distance_label.text = f"Distance: X={distance[0]:.2f} cm, Y={distance[1]:.2f} cm, Z={distance[2]:.2f} cm"
            accel_label_x.text = f"Accel X: {accel[0] * gravity:.2f} m/s²"
            accel_label_y.text = f"Accel Y: {accel[1] * gravity:.2f} m/s²"
            accel_label_z.text = f"Accel Z: {accel[2] * gravity:.2f} m/s²"

# Run the visualization
run_visualization()
