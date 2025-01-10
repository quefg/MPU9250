import time
import numpy as np
from smbus2 import SMBus

# Constants for I2C and MPU9250 setup
I2C_BUS = 8
MPU9250_ADDR = 0x68  

# MPU9250 Register Addresses
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
PWR_MGMT_1 = 0x6B  # Power management registers

# Constants for calculations
gravity = 9.81  # Acceleration due to gravity in m/s²
noise_threshold = 0.01  # Threshold to filter out noise in acceleration
velocity_reset_threshold = 0.1  # Threshold to reset velocity when stationary
sampling_interval = 0.05  # Sampling interval in seconds (100 ms)

# Initial states
accel_bias = [0.0, 0.0, 0.0]  # Bias correction for accelerometer
velocity = [0.0, 0.0, 0.0]  # Initial velocity in X, Y, Z
position = [0.0, 0.0, 0.0]  # Initial position in X, Y, Z
previous_acceleration = [0.0, 0.0, 0.0]  # To store acceleration from the previous step

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
        ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_YOUT_H) / 16384.0
        az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_ZOUT_H) / 16384.0
        ax_total += ax
        ay_total += ay
        az_total += az
     
    accel_bias = [ax_total / samples, ay_total / samples, az_total / samples - gravity]  # Subtract gravity for Z-axis

def read_accel(bus):
    """Read accelerometer data."""
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0 - accel_bias[0]
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_YOUT_H ) / 16384.0 - accel_bias[1]
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_ZOUT_H ) / 16384.0 - accel_bias[2]
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

def update_velocity_and_position(ax, ay, az, dt):
    """Update velocity and position using trapezoidal integration."""
    global velocity, position, previous_acceleration

    # Update velocity for each axis
    velocity[0] += (previous_acceleration[0] + ax) * dt / 2
    velocity[1] += (previous_acceleration[1] + ay) * dt / 2
    velocity[2] += (previous_acceleration[2] + az) * dt / 2

    # Reset velocity if the accelerometer readings are below the threshold
    if abs(ax) < velocity_reset_threshold and abs(ay) < velocity_reset_threshold and abs(az) < velocity_reset_threshold:
        velocity = [0.0, 0.0, 0.0]

    # Update position for each axis
    position[0] += (velocity[0] + velocity[0]) * dt / 2
    position[1] += (velocity[1] + velocity[1]) * dt / 2
    position[2] += (velocity[2] + velocity[2]) * dt / 2

    # Update previous acceleration for the next iteration
    previous_acceleration = [ax, ay, az]

def read_acceleration():
    with SMBus(I2C_BUS) as bus:
        bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up sensor
        calibrate_sensor(bus)  # Perform calibration

        global previous_acceleration
        previous_acceleration = [0.0, 0.0, 0.0]

        while True:
            
                # Read acceleration data
                ax, ay, az = read_accel(bus)

                # Cancel gravitational effect to correct the raw acceleration
                ax_corrected, ay_corrected, az_corrected = cancel_gravity(ax, ay, az, pitch=0, roll=0)

                # Apply threshold to filter out noise
                ax_corrected = ax_corrected if abs(ax_corrected) > noise_threshold else 0.0
                ay_corrected = ay_corrected if abs(ay_corrected) > noise_threshold else 0.0
                az_corrected = az_corrected if abs(az_corrected) > noise_threshold else 0.0

                # Update velocity and position
                update_velocity_and_position(ax_corrected, ay_corrected, az_corrected, sampling_interval)

                # Print corrected acceleration, velocity, and position
                print(f"Acceleration: X={ax_corrected:.2f}, Y={ay_corrected:.2f}, Z={az_corrected:.2f} m/s²")
                print(f"Velocity: X={velocity[0]:.2f}, Y={velocity[1]:.2f}, Z={velocity[2]:.2f} m/s")
                print(f"Position: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f} m\n")

                time.sleep(sampling_interval)

if __name__ == "__main__":
    read_acceleration()

