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
threshold = 0.1  # Threshold for stationary detection (m/s²)

# Initial states
velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (vx, vy, vz) in m/s
position = np.array([0.0, 0.0, 0.0])  # Initial position (x, y, z) in meters
previous_position = np.array([0.0, 0.0, 0.0])  # Initial previous position for relative distance
time_since_last_update = 0  # Time tracker for relative distance calculation
orientation = np.array([0.0, 0.0, 0.0])  # Pitch, Roll, Yaw in degrees
previous_accel = np.array([0.0, 0.0, 0.0])  # For low-pass filter

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
    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / gyro_scale  # Scale for ±250°/s
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / gyro_scale
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / gyro_scale
    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def setup_mpu(bus):
    """Initialize MPU9250."""
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up MPU9250

def calibrate_accelerometer(bus):
    """Calibrate the accelerometer to remove bias."""
    print("Calibrating accelerometer... Please keep the sensor stationary.")
    time.sleep(3)
    samples = []

    for _ in range(100):  # Collect 100 samples
        accel, _ = read_accel_gyro(bus)
        samples.append(accel)
        time.sleep(0.01)  # Short delay between readings

    # Calculate the mean bias for each axis
    bias = np.mean(samples, axis=0)
    print(f"Calibration complete. Bias: {bias}")
    return bias

def apply_low_pass_filter(accel, previous_accel, alpha=0.98):
    """Apply a simple low-pass filter to smooth accelerometer data."""
    return alpha * previous_accel + (1 - alpha) * accel

def reset_velocity_if_stationary(accel, velocity, threshold):
    """Reset velocity if the device is stationary."""
    if np.linalg.norm(accel) < threshold:
        return np.array([0.0, 0.0, 0.0])
    return velocity

def update_orientation(gyro, orientation, dt):
    """Update orientation (pitch, roll, yaw) based on gyroscope readings."""
    return orientation + gyro * dt  # Integrate angular velocity

def get_rotation_matrix(pitch, roll, yaw):
    """Calculate the 3D rotation matrix."""
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

# Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "MPU9250 3D Visualization with Distance Display"
scene.range = 2  # Adjust the view range

# Draw reference XYZ axes
x_axis = cylinder(pos=vector(0, 0, 0), axis=vector(2, 0, 0), radius=0.02, color=vector(1, 0, 0))
y_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 2, 0), radius=0.02, color=vector(0, 1, 0))
z_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 2), radius=0.02, color=vector(0, 0, 1))

mpu_box = box(
    size=vector(1.004, 0.606, 0.118),  # Dimensions in inches
    color=vector(0, 1, 0)
)

# Labels for acceleration and distance
acceleration_label = label(pos=vector(0, -2.5, 0), text="Acceleration: ")
distance_label = label(pos=vector(0, -3, 0), text="Distance: ")

# Main Program
def run_visualization():
    global velocity, position, previous_position, time_since_last_update, previous_accel, orientation

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)

        # Calibrate accelerometer
        calibration_bias = calibrate_accelerometer(bus)

        while True:
            rate(50)  # Update rate 50 Hz
            time_since_last_update += dt  # Increment time tracker

            accel, gyro = read_accel_gyro(bus)

            # Apply calibration to remove bias
            accel -= calibration_bias

            # Apply low-pass filter to smooth noise
            accel = apply_low_pass_filter(accel, previous_accel)
            previous_accel = accel.copy()

            # Convert accelerometer readings to m/s²
            accel_corrected = accel * gravity
            accel_corrected[2] -= gravity  # Remove gravity component from Z-axis

            # Update velocity and position
            velocity = reset_velocity_if_stationary(accel_corrected, velocity, threshold)
            velocity += accel_corrected * dt
            position += velocity * dt + 0.5 * accel_corrected * dt**2

            # Update orientation using gyroscope data
            orientation = update_orientation(gyro, orientation, dt)

            # Compute rotation matrix and apply to the 3D box
            R = get_rotation_matrix(orientation[0], orientation[1], orientation[2])
            mpu_box.axis = vector(R[0, 2], R[1, 2], R[2, 2])
            mpu_box.up = vector(R[0, 1], R[1, 1], R[2, 1])

            # Calculate distance traveled every 5 seconds
            if time_since_last_update >= 5.0:  # Update distance every 5 seconds
                relative_distance = np.linalg.norm(position - previous_position)
                previous_position = position.copy()
                time_since_last_update = 0
                distance_label.text = f"Distance Traveled: {np.round(relative_distance * 100, 2)} cm"

            # Update labels
            acceleration_label.text = f"Acceleration (x, y, z): {np.round(accel_corrected, 2)} m/s²"

# Run the visualization
run_visualization()
