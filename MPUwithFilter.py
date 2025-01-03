import sys
import numpy as np
from smbus2 import SMBus
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

# I2C setup
I2C_BUS = 8
MPU9250_ADDR = 0x68

# MPU9250 Register Map
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
PWR_MGMT_1 = 0x6B

# Constants
dt = 20 / 1000.0  # Sampling period in seconds
gravity = 9.81  # Earth's gravity (m/s^2)

# I2C Functions
def read_i2c_word(bus, addr, reg):
    """Read two bytes from I2C and combine into a signed word."""
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    return value - 65536 if value > 32768 else value

def read_accel_gyro(bus):
    """Read accelerometer and gyroscope data."""
    ax = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H) / 16384.0
    ay = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 2) / 16384.0
    az = read_i2c_word(bus, MPU9250_ADDR, ACCEL_XOUT_H + 4) / 16384.0
    gx = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H) / 131.0
    gy = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 2) / 131.0
    gz = read_i2c_word(bus, MPU9250_ADDR, GYRO_XOUT_H + 4) / 131.0
    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def setup_mpu(bus):
    """Initialize MPU9250."""
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # Wake up MPU9250

# PyQtGraph Visualization Class
class MPUVisualization(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPU9250 Real-Time Visualization")
        self.setGeometry(100, 100, 800, 600)

        # Add a grid for reference
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10)
        grid.setSpacing(x=1, y=1)
        self.addItem(grid)

        # Add a 3D box (sensor representation)
        self.sensor = gl.GLMeshItem(
            meshdata=gl.MeshData.cube(),
            color=(0, 1, 0, 0.5),
            smooth=False,
            shader="shaded",
        )
        self.sensor.scale(1.004, 0.606, 0.118)  # Scale to sensor dimensions (in inches)
        self.addItem(self.sensor)

    def update_orientation(self, angles):
        """Update the sensor orientation."""
        self.sensor.resetTransform()
        self.sensor.rotate(angles[0], 1, 0, 0)  # Rotate around x-axis
        self.sensor.rotate(angles[1], 0, 1, 0)  # Rotate around y-axis
        self.sensor.rotate(angles[2], 0, 0, 1)  # Rotate around z-axis

# Main Application
app = QApplication(sys.argv)
viewer = MPUVisualization()

with SMBus(I2C_BUS) as bus:
    setup_mpu(bus)

    velocity = np.array([0.0, 0.0, 0.0])
    position = np.array([0.0, 0.0, 0.0])
    angles = np.array([0.0, 0.0, 0.0])  # Tracks rotation angles

    def update():
        global velocity, position, angles

        accel, gyro = read_accel_gyro(bus)

        # Remove gravity (assuming calibration is done; modify as needed)
        accel -= np.array([0, 0, gravity])

        # Integrate acceleration to calculate velocity
        velocity += accel * dt

        # Integrate velocity to calculate position
        position += velocity * dt

        # Update angles using gyroscope data
        angles += gyro * dt

        # Update visualization
        viewer.update_orientation(angles)

    # Set up a timer for real-time updates
    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(20)  # 50 Hz refresh rate

viewer.show()
sys.exit(app.exec_())
