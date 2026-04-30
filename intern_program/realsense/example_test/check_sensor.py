import pyrealsense2 as rs

pipeline = rs.pipeline()
profile = pipeline.start()

device = profile.get_device()

for sensor in device.query_sensors():
    print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")

    for option in sensor.get_supported_options():
        print(f"  {option} = {sensor.get_option(option)}")

pipeline.stop()