import time
import numpy as np
from smbus2 import SMBus
from vpython import vector, box, rate, scene, label

# I2C setup
I2C_BUS = 8  # Update with your I2C bus number
MPU9250_ADDR = 0x68

# MPU9250 Register Map
ACCEL_XOUT_H = 0x3B
PWR_MGMT_1 = 0x6B

# Constants
dt = 0.02  # Sampling period in seconds (50 Hz)
alpha = 0.98  # Complementary filter coefficient
gravity = 9.81  # Gravity in m/s^2
threshold = 0.1  # Stationary threshold in m/s²

# Initial states
velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (vx, vy, vz)
position = np.array([0.0, 0.0, 0.0])  # Initial position (x, y, z)
previous_accel = np.array([0.0, 0.0, 0.0])  # Previous filtered acceleration
initial_orientation = np.array([0.0, 0.0, 0.0])  # Reference orientation

# Visualization Setup
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "Pilates Reformer Tracking"
scene.range = 2

mpu_box = box(size=vector(0.5, 0.1, 0.1), color=vector(0, 1, 0))
distance_label = label(pos=vector(0, -1, 0), text="Distance: ")

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

def setup_mpu(bus):
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up MPU9250

def calibrate_sensor(bus):
    global initial_orientation
    print("Calibrating sensor...")
    time.sleep(3)
    samples = []
    for _ in range(50):
        accel = read_accel(bus)
        samples.append(accel)
        time.sleep(0.01)
    avg_accel = np.mean(samples, axis=0)
    initial_orientation[2] = avg_accel[2]  # Gravity reference
    print(f"Calibration complete: {initial_orientation}")

# Main Program
def run_tracking():
    global velocity, position, previous_accel

    with SMBus(I2C_BUS) as bus:
        setup_mpu(bus)
        calibrate_sensor(bus)

        while True:
            rate(50)  # Update rate 50 Hz
            accel = read_accel(bus)

            # Gravity compensation
            accel_corrected = accel * gravity
            accel_corrected[2] -= initial_orientation[2]  # Remove gravity from Z-axis

            # Detect stationary and reset velocity
            if np.linalg.norm(accel_corrected) < threshold:
                velocity = np.array([0.0, 0.0, 0.0])

            # Update velocity and position
            velocity += accel_corrected * dt
            position += velocity * dt

            # Track movement along X-axis (reformer movement)
            relative_position_x = position[0]  # Assuming X-axis aligns with reformer track

            # Update visualization
            mpu_box.pos = vector(relative_position_x, 0, 0)
            distance_label.text = f"Distance: {np.round(relative_position_x * 100, 2)} cm"

# Run tracking
run_tracking()
