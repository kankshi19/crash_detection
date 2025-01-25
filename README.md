# Crash Detection Model

This model predicts whether a crash is likely based on sensor data (accelerometer and gyroscope values).

## Input Features:
- accel_x, accel_y, accel_z: Accelerometer data (x, y, z axes)
- gyro_x, gyro_y, gyro_z: Gyroscope data (x, y, z axes)
- timestamp: Unix timestamp of the data

## Output:
- crash_prediction: 1 if a crash is predicted, 0 otherwise

## Example Request:
```json
{
    "accel_x": -1.5,
    "accel_y": 8.2,
    "accel_z": 0.3,
    "gyro_x": 0.15,
    "gyro_y": -0.25,
    "gyro_z": 0.13,
    "timestamp": 1625432933
}