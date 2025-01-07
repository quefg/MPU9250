import time
import numpy as np
import RPi.GPIO as GPIO
from smbus2 import SMBus
from vpython import vector, box, rate, scene, label, cylinder

# I2C setup
I2C_BUS = 8  # Update with your I2C bus number
MPU9250_ADDR = 0x68
# MPU9250 Register Map
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B
INT_ENABLE = 0x38
INT_PIN_CFG = 0x37

# GPIO setup
INT_PIN = 17  # GPIO pin connected to MPU9250 interrupt pin

# Constants
dt = 0.02  # Sampling period in seconds (50 Hz)
alpha = 0.98  # Complementary filter coefficient
gravity = 9.81  # Gravity in m/s²
velocity_threshold = 0.05  # Threshold to detect stationary motion (cm/s)
noise_threshold = 0.02  # Ignore small accelerometer values (g)

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
initial_position = np.array([0.0, 0.0, 0.0])
prev_accel = np.array([0.0, 0.0, 0.0])  # For low-pass filtering
calibrated = False  # Flag to track if calibration is done

def setup_interrupt(bus):
   
    # Wake up MPU9250
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)

    # Enable interrupt on data ready
    bus.write_byte_data(MPU9250_ADDR, INT_ENABLE, 0x01)  # Data ready interrupt enabled

    # Configure INT_PIN_CFG for active high, push-pull
    bus.write_byte_data(MPU9250_ADDR, INT_PIN_CFG, 0x30)  # Interrupt pin configuration

def calibrate_position(bus):
    """Calibrate the initial position and orientation."""
    global initial_position, prev_accel, calibrated

    print("Calibrating... Place the sensor in a stationary position.")
    samples = []
    for _ in range(100):  # Collect multiple samples for better accuracy
        accel, _ = read_accel_gyro(bus)
        samples.append(accel)
        time.sleep(0.01)
    initial_position = np.mean(samples, axis=0) * gravity
    prev_accel = np.mean(samples, axis=0)  # Set initial acceleration for filtering
    calibrated = True
    print(f"Calibration complete. Initial position: {initial_position}")

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
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0

    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / 131.0
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / 131.0
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / 131.0

    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def interrupt_handler(channel):
    """Callback function triggered by MPU9250 interrupt."""
    global velocity, distance, prev_accel, orientation, calibrated

    if not calibrated:
        print("Waiting for calibration...")
        return

    accel, gyro = read_accel_gyro(bus)

    # Filter acceleration
    accel = low_pass_filter(accel, prev_accel)
    prev_accel = accel
    accel_no_gravity = high_pass_filter(accel)
    accel_no_gravity[np.abs(accel_no_gravity) < noise_threshold] = 0

    # Update velocity and distance
    velocity += accel_no_gravity * gravity * dt
    distance += velocity * dt * 100  # Convert to cm

    # Update 3D visualization
    angle_label.text = f"Angles: Pitch={orientation[1]:.2f}, Roll={orientation[0]:.2f}, Yaw={orientation[2]:.2f}"
    distance_label.text = f"Distance: X={distance[0]:.2f} cm, Y={distance[1]:.2f} cm, Z={distance[2]:.2f} cm"

# Setup GPIO for interrupt
GPIO.setmode(GPIO.BCM)
GPIO.setup(INT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.add_event_detect(INT_PIN, GPIO.RISING, callback=interrupt_handler)

# Main loop
def run_visualization():
    global calibrated

    with SMBus(I2C_BUS) as bus:
        setup_interrupt(bus)  # Configure MPU9250 interrupts
        calibrate_position(bus)  # Perform initial calibration

        print("Waiting for interrupt-driven data...")
        try:
            while True:
                time.sleep(1)  # Keep the program running to listen for interrupts
        except KeyboardInterrupt:
            GPIO.cleanup()  # Clean up GPIO on exit

run_visualization()
