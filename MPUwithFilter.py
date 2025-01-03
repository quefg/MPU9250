import time
import numpy as np
from smbus2 import SMBus
from vpython import box, vector, rate, scene, arrow, label

# I2C setup
I2C_BUS = 8
MPU9250_ADDR = 0x68  # MPU9250 I2C address

# MPU9250 Register Map
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B

# Constants
dt = 20  # Sampling period in milliseconds
gravity = 9.81  # Earth's gravity (m/s^2)

# 3D Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "MPU9250 Visualization"
scene.width = 800
scene.height = 600

# MPU 3D Box
mpu_box = box(
    size=vector(1.004, 0.606, 0.118),  # Real board dimensions in inches
    color=vector(0, 1, 0)
)

# Axes
x_axis = arrow(pos=vector(0, 0, 0), axis=vector(2, 0, 0), color=vector(1, 0, 0))  # Red for X
y_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 2, 0), color=vector(0, 1, 0))  # Green for Y
z_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 2), color=vector(0, 0, 1))  # Blue for Z

# Labels
info_label = label(pos=vector(0, -2, 0), text="Initializing...")

# Initial orientation (calibration step)
initial_accel = None
gyro_angle = np.array([0.0, 0.0, 0.0])  # Tracks rotation angles

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
    bus.write_byte_data(MPU9250_ADDR, ACCEL_CONFIG, 0x00)  # ±2g
    bus.write_byte_data(MPU9250_ADDR, GYRO_CONFIG, 0x00)  # ±250°/s

def calibrate_sensor(bus):
    """Calibrate sensor to get initial orientation."""
    global initial_accel
    print("Calibrating sensor... Hold the MPU steady.")
    time.sleep(3)
    samples = []
    for _ in range(50):
        accel, _ = read_accel_gyro(bus)
        samples.append(accel)
    initial_accel = np.mean(samples, axis=0)
    print(f"Initial acceleration: {initial_accel}")

def remove_gravity(accel):
    """Remove gravity component."""
    return accel - initial_accel

# Main Program
with SMBus(I2C_BUS) as bus:
    setup_mpu(bus)
    calibrate_sensor(bus)

    velocity = np.array([0.0, 0.0, 0.0])
    position = np.array([0.0, 0.0, 0.0])

    while True:
        rate(50)  # Update rate in Hz
        accel, gyro = read_accel_gyro(bus)
        linear_accel = remove_gravity(accel)

        # Integrate acceleration for velocity
        velocity += linear_accel * (dt / 1000.0)  # Convert dt to seconds

        # Integrate velocity for position
        position += velocity * (dt / 1000.0)  # Convert dt to seconds

        # Update rotation angles using gyroscope data
        gyro_angle += gyro * (dt / 1000.0)  # Convert dt to seconds

        # Update 3D visualization
        mpu_box.pos = vector(position[0], position[1], position[2])  # Position update
        mpu_box.axis = vector(
            np.cos(np.radians(gyro_angle[0])),
            np.cos(np.radians(gyro_angle[1])),
            np.cos(np.radians(gyro_angle[2]))
        )  # Orientation update

        # Update axes
        x_axis.axis = vector(2 * np.cos(np.radians(gyro_angle[0])), 0, 0)
        y_axis.axis = vector(0, 2 * np.cos(np.radians(gyro_angle[1])), 0)
        z_axis.axis = vector(0, 0, 2 * np.cos(np.radians(gyro_angle[2])))

        # Display sensor data
        info_label.text = (
            f"Position: {position}\n"
            f"Velocity: {velocity}\n"
            f"Angles: Roll: {gyro_angle[0]:.2f}, Pitch: {gyro_angle[1]:.2f}, Yaw: {gyro_angle[2]:.2f}"
        )
