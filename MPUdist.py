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

# Labels for angles, acceleration, and distance
angle_label = label(pos=vector(0, -2, 0), text="Angles: ")
acceleration_label = label(pos=vector(0, -2.5, 0), text="Acceleration: ")
distance_label = label(pos=vector(0, -3, 0), text="Distance: ")

# Initial orientation and displacement
initial_orientation = np.array([0.0, 0.0, 0.0])  # Reference orientation (calibrated)
orientation = np.array([0.0, 0.0, 0.0])  # Tracks filtered orientation
velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (vx, vy, vz) in m/s
position = np.array([0.0, 0.0, 0.0])  # Initial position (x, y, z) in meters

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

def update_position(accel, dt):
    """Update position and velocity based on acceleration."""
    global velocity, position

    # Convert accelerometer readings to m/s² and subtract gravity in the Z-axis
    accel_corrected = accel * gravity  # Convert g to m/s²
    accel_corrected[2] -= gravity  # Remove gravity from Z-axis

    # Update velocity (v = u + at)
    velocity += accel_corrected * dt

    # Update position (s = ut + 0.5 * a * t²)
    position += velocity * dt + 0.5 * accel_corrected * dt**2

    return position

# Main Program
def run_visualization():
    global orientation, position

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)
        calibrate_sensor(bus)

        while True:
            rate(50)  # Update rate 50 Hz
            accel, gyro = read_accel_gyro(bus)

            # Calculate tilt angles from accelerometer
            accel_pitch = np.arctan2(accel[1], accel[2]) * 180 / np.pi
            accel_roll = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2)) * 180 / np.pi

            # Complementary filter to combine accelerometer and gyroscope data
            orientation[0] = alpha * (orientation[0] + gyro[0] * dt) + (1 - alpha) * (accel_roll - initial_orientation[0])
            orientation[1] = alpha * (orientation[1] + gyro[1] * dt) + (1 - alpha) * (accel_pitch - initial_orientation[1])
            orientation[2] += gyro[2] * dt  # Gyroscope-only yaw

            # Update rotation matrix
            R = get_rotation_matrix(orientation[1], orientation[0], orientation[2])

            # Apply rotation matrix to box
            mpu_box.axis = vector(R[0, 2], R[1, 2], R[2, 2])
            mpu_box.up = vector(R[0, 1], R[1, 1], R[2, 1])

            # Update position based on accelerometer
            position = update_position(accel, dt)

            # Calculate total distance moved
            distance = np.linalg.norm(position)  # Euclidean distance from origin

            # Update labels
            angle_label.text = f"Angles (Pitch, Roll, Yaw): {np.round(orientation, 2)}"
            acceleration_label.text = f"Acceleration (x, y, z): {np.round(accel * gravity, 2)} m/s²"
            distance_label.text = f"Distance moved: {np.round(distance * 100, 2)} cm"

# Run the visualization
run_visualization()
