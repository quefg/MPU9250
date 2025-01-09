import time
import numpy as np
from smbus2 import SMBus
from vpython import vector, box, rate, scene, label, cylinder

# Constants for I2C and MPU9250 setup
I2C_BUS = 8
MPU9250_ADDR = 0x68  

# MPU9250 Register Addresses
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B  # Power management registers

# Constants for calculations
gravity = 9.81  # Acceleration due to gravity in m/s²
noise_threshold = 0.01  # Threshold to filter out noise in acceleration
alpha = 0.95  # Coefficient for complementary filter

# Initial states
velocity_x, velocity_y, velocity_z = 0.0, 0.0, 0.0  # Velocities for X, Y, Z
displacement_x, displacement_y, displacement_z = 0.0, 0.0, 0.0  # Displacements for X, Y, Z
accel_bias = [0.0, 0.0, 0.0]  # Bias correction for accelerometer

# Visualization setup in VPython
scene.background = vector(0.2, 0.2, 0.2)
scene.title = "MPU9250 3D Visualization"
scene.range = 3

# 3D axes and MPU box for orientation
x_axis = cylinder(pos=vector(0, 0, 0), axis=vector(2, 0, 0), radius=0.02, color=vector(1, 0, 0))
y_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 2, 0), radius=0.02, color=vector(0, 1, 0))
z_axis = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 2), radius=0.02, color=vector(0, 0, 1))
mpu_box = box(size=vector(1.004, 0.606, 0.118), color=vector(0, 1, 0))

# Labels for displaying data (one row for each category)
accel_label = label(pos=vector(0, -1, 0), text="Acceleration: X=0.00, Y=0.00, Z=0.00 m/s²", height=14, color=vector(1, 0.5, 0))
distance_label = label(pos=vector(0, -1.3, 0), text="Displacement: X=0.00, Y=0.00, Z=0.00 m", height=14, color=vector(1, 1, 1))

# Functions to read and process data from the MPU9250
def read_i2c_word(bus, addr, reg):
    """Read two bytes from I2C and combine into a signed word."""
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    return value - 65536 if value > 32768 else value

def calibrate_sensor(bus, samples=100):
    """Calibrate the accelerometer to calculate biases."""
    global accel_bias
    ax_total, ay_total, az_total = 0.0, 0.0, 0.0
    for _ in range(samples):
        ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0
        ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
        az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0
        ax_total += ax
        ay_total += ay
        az_total += az
        time.sleep(0.01)
    accel_bias = [ax_total / samples, ay_total / samples, az_total / samples - 1.0]  # Subtract 1.0g for gravity

def read_accel_gyro(bus):
    """Read accelerometer and gyroscope data."""
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0 - accel_bias[0]
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0 - accel_bias[1]
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0 - accel_bias[2]

    accel_label.text = f"Acceleration: X={ax:.2f}, Y={ay:.2f}, Z={az:.2f} m/s²"
    return ax, ay, az

def cancel_gravity(ax, ay, az, pitch, roll):
    """Cancel the effect of gravity from raw acceleration readings."""
    g_x = gravity * np.sin(np.radians(pitch))
    g_y = -gravity * np.sin(np.radians(roll)) * np.cos(np.radians(pitch))
    g_z = gravity * np.cos(np.radians(roll)) * np.cos(np.radians(pitch))

    ax_corrected = ax - g_x
    ay_corrected = ay - g_y
    az_corrected = az - g_z

    return ax_corrected, ay_corrected, az_corrected
# Main loop for visualization and distance estimation
def run_visualization_and_distance_estimation():
    global velocity_x, velocity_y, velocity_z
    global displacement_x, displacement_y, displacement_z

    with SMBus(I2C_BUS) as bus:
        bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up sensor
        calibrate_sensor(bus)  # Perform calibration
        last_time = time.time()

        while True:
           
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            ax, ay, az = read_accel_gyro(bus)

            # Cancel gravitational effect to correct the raw acceleration
            ax_corrected, ay_corrected, az_corrected = ax, ay, az
            ax_corrected, ay_corrected, az_corrected = cancel_gravity(ax, ay, az, pitch=0, roll=0)  # Assuming no tilt for simplicity

            # Apply threshold to filter out noise
            ax_corrected = ax_corrected if abs(ax_corrected) > noise_threshold else 0.0
            ay_corrected = ay_corrected if abs(ay_corrected) > noise_threshold else 0.0
            az_corrected = az_corrected if abs(az_corrected) > noise_threshold else 0.0

            # Update velocity for each axis
            velocity_x += ax_corrected * dt
            velocity_y += ay_corrected * dt
            velocity_z += az_corrected * dt  # Update vertical velocity

            # Update displacement for each axis
            incremental_displacement_x = velocity_x * dt
            incremental_displacement_y = velocity_y * dt
            incremental_displacement_z = velocity_z * dt

            # Update the overall displacement
            displacement_x += incremental_displacement_x
            displacement_y += incremental_displacement_y
            displacement_z += incremental_displacement_z

            # Print incremental displacement
            print(f"Incremental Displacement: X={incremental_displacement_x:+.2f}, "
                  f"Y={incremental_displacement_y:+.2f}, Z={incremental_displacement_z:+.2f} m")

            # Update the displacement label with signs
            distance_label.text = (
                f"Displacement: X={displacement_x:+.2f}, "
                f"Y={displacement_y:+.2f}, Z={displacement_z:+.2f} m"
            )
# Run the visualization and distance estimation
run_visualization_and_distance_estimation()
