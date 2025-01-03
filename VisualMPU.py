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
dt = 0.02  # Sampling period in seconds
gravity = 9.81  # Earth's gravity (m/s^2)

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
mpu_box = box(
    size=vector(1.004, 0.606, 0.118),  # Dimensions in inches
    color=vector(0, 1, 0)
)
info_label = label(pos=vector(0, -2, 0), text="")

# Initial orientation (calibration step)
initial_accel = None
gyro_angle = np.array([0.0, 0.0, 0.0])  # Tracks rotation angles

def calibrate_sensor(bus):
    """Calibrate the sensor to get the initial orientation."""
    global initial_accel
    print("Calibrating sensor... Hold the MPU steady.")
    time.sleep(3)
    samples = []
    for _ in range(50):
        accel, _ = read_accel_gyro(bus)
        samples.append(accel)
    initial_accel = np.mean(samples, axis=0)
    print(f"Initial acceleration calibrated: {initial_accel}")

def remove_gravity(accel):
    """Remove gravity component."""
    return accel - initial_accel

# Main Program (Jupyter-Compatible)
def run_visualization():
    global gyro_angle  # Declare gyro_angle as global
    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)
        calibrate_sensor(bus)

        velocity = np.array([0.0, 0.0, 0.0])
        position = np.array([0.0, 0.0, 0.0])

        while True:
            rate(50)  # Update rate 50 Hz
            accel, gyro = read_accel_gyro(bus)

            # Remove gravity
            linear_accel = remove_gravity(accel)

            # Integrate acceleration for velocity
            velocity += linear_accel * dt

            # Integrate velocity for position
            position += velocity * dt

            # Update rotation angles using gyroscope data
            gyro_angle += gyro * dt  # Update global gyro_angle

            # Update visualization
            mpu_box.pos = vector(position[0], position[1], position[2])
            mpu_box.axis = vector(
                np.cos(np.radians(gyro_angle[0])),
                np.cos(np.radians(gyro_angle[1])),
                np.cos(np.radians(gyro_angle[2]))
            )
            info_label.text = f"Pos: {position}, Vel: {velocity}, Angles: {gyro_angle}"

# Call the function to start visualization
run_visualization()
