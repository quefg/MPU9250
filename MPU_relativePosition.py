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

# Labels for angles
angle_label = label(pos=vector(0, -2, 0), text="Angles: ")

# Labels for acceleration
accel_label_x = label(pos=vector(0, -2.5, 0), text="Accel X: 0.00 m/s²", color=vector(1, 0, 0))
accel_label_y = label(pos=vector(0, -2.8, 0), text="Accel Y: 0.00 m/s²", color=vector(0, 1, 0))
accel_label_z = label(pos=vector(0, -3.1, 0), text="Accel Z: 0.00 m/s²", color=vector(0, 0, 1))

# Labels for distance in cm
distance_label = label(pos=vector(0, -3.5, 0), text="Distance: X=0.00 cm, Y=0.00 cm, Z=0.00 cm", color=vector(1, 1, 1))

# Initialize state variables
velocity = np.array([0.0, 0.0, 0.0])  # Velocity in m/s
distance = np.array([0.0, 0.0, 0.0])  # Distance in cm
prev_accel = np.array([0.0, 0.0, 0.0])  # For low-pass filtering

def low_pass_filter(accel, prev_accel, alpha=0.5):
    """Apply a low-pass filter to smooth acceleration data."""
    return alpha * accel + (1 - alpha) * prev_accel

def high_pass_filter(accel):
    """Remove gravity from acceleration data."""
    accel[2] -= 1.0  # Assuming gravity on Z-axis
    return accel

def reset_velocity_if_stationary(gyro):
    """Reset velocity if the sensor is stationary."""
    global velocity
    if np.linalg.norm(gyro) < velocity_threshold:  # Check if angular velocity is below threshold
        velocity[:] = 0.0

# Main Program
def run_visualization():
    global velocity, distance, prev_accel

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)

        while True:
            rate(50)  # Update rate 50 Hz
            accel, gyro = read_accel_gyro(bus)

            # Apply low-pass filter to acceleration
            accel = low_pass_filter(accel, prev_accel)
            prev_accel = accel

            # Remove gravity (high-pass filter)
            accel_no_gravity = high_pass_filter(accel)

            # Ignore small acceleration values below noise threshold
            accel_no_gravity[np.abs(accel_no_gravity) < noise_threshold] = 0

            # Reset velocity if stationary
            reset_velocity_if_stationary(gyro)

            # Update velocity (v = u + at)
            velocity += accel_no_gravity * gravity * dt

            # Update distance (s = s + vt, converted to cm)
            distance += velocity * dt * 100  # Convert m to cm

            # Update labels
            accel_label_x.text = f"Accel X: {accel[0] * gravity:.2f} m/s²"
            accel_label_y.text = f"Accel Y: {accel[1] * gravity:.2f} m/s²"
            accel_label_z.text = f"Accel Z: {accel[2] * gravity:.2f} m/s²"
            distance_label.text = f"Distance: X={distance[0]:.2f} cm, Y={distance[1]:.2f} cm, Z={distance[2]:.2f} cm"

# Run the visualization
run_visualization()
