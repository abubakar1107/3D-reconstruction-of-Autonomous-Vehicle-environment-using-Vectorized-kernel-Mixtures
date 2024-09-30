import carla
import pygame
import random
import time
import numpy as np
import math

def connect_to_carla():
    for _ in range(10):  # Attempt to connect 10 times
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            world = client.get_world()  # This should not throw an error if the server is ready
            print("Successfully connected to CARLA server!")
            return client, world
        except RuntimeError as e:
            print(f"Connection attempt failed: {e}")
            time.sleep(1)
    raise RuntimeError("Failed to connect to CARLA server after several attempts")

def lidar_callback(lidar_data, display, lidar_file):
    # Points format: x, y, z, intensity
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    # Transform the points to 2D for visualization
    lidar_data_2d = np.array(points[:, :2])
    lidar_data_2d *= min(display.get_width(), display.get_height()) / 100.0  # scale points for visibility
    lidar_data_2d += (0.5 * display.get_width(), 0.5 * display.get_height())
    lidar_data_2d = np.fabs(lidar_data_2d)  # Absolute value to handle flipping
    lidar_data_2d = lidar_data_2d.astype(np.int32)
    lidar_data_2d = np.clip(lidar_data_2d, 0, min(display.get_width() - 1, display.get_height() - 1))

    # Draw the points on Pygame display
    display.fill((0, 0, 0))  # Clear the display
    for point in lidar_data_2d:
        pygame.draw.circle(display, (255, 255, 255), point, 2)

    pygame.display.flip()

    # Log data to file
    with open(lidar_file, 'a') as file:
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")  # Write x, y, z

def drive_to_waypoint(vehicle, destination, world, display):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = vehicle_transform.rotation.yaw
    distance = np.linalg.norm(np.array([vehicle_location.x, vehicle_location.y]) - np.array([destination.x, destination.y]))

    if distance < 2:  # When close enough to the waypoint
        print("Reached waypoint")
        return True  # Signal that the waypoint has been reached

    direction_vector = np.array([destination.x - vehicle_location.x, destination.y - vehicle_location.y])
    direction_norm = np.linalg.norm(direction_vector)
    if direction_norm > 0:
        normalized_direction = direction_vector / direction_norm
        target_yaw = math.degrees(math.atan2(normalized_direction[1], normalized_direction[0]))
        delta_yaw = target_yaw - vehicle_yaw
        delta_yaw = (delta_yaw + 180) % 360 - 180  # Normalize the angle to [-180, 180]

        steer = delta_yaw / 180.0  # Normalize steering
        steer = max(min(steer, 1.0), -1.0)  # Clamp values to [-1, 1]

        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer))  # Apply control
        
        world.tick()
        time.sleep(0.1)
    return False

def main():
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA LiDAR Visualization")

    client, world = connect_to_carla()
    world = client.load_world('Town03')

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
    spawn_point = carla.Transform(carla.Location(x=240, y=-7, z=10), carla.Rotation(pitch=0, yaw=180, roll=0))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Add a LiDAR sensor to the vehicle
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')  # 100 meters range
    lidar_bp.set_attribute('rotation_frequency', '10')  # 10 Hz rotation frequency
    lidar_bp.set_attribute('points_per_second', '100000')  # High resolution
    lidar_location = carla.Transform(carla.Location(x=0.8, z=1.7))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_location, attach_to=vehicle)

    # Prepare the file for storing LiDAR data
    lidar_file = "town3_lidar_data.xyz"
    open(lidar_file, 'w').close()  # Create or clear the file before starting

    lidar_sensor.listen(lambda data: lidar_callback(data, display, lidar_file))

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(carla.Location(x=250, y=-7, z=10), carla.Rotation(pitch=-15)))

    def update_camera():
        camera_location = vehicle.get_transform().location + carla.Location(x=15, z=10)
        camera_rotation = carla.Rotation(pitch=-15, yaw=180)
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))


    waypoints = [
        carla.Location(x=240, y=-7, z=10), carla.Location(x=220, y=-7, z=10), carla.Location(x=200, y=-7, z=10),
        carla.Location(x=170, y=-7, z=10), carla.Location(x=160, y=-7, z=10), carla.Location(x=155, y=-20, z=10),
        carla.Location(x=155, y=-35, z=10)]
    # , carla.Location(x=130, y=-73, z=10), carla.Location(x=110, y=-70, z=10)]

    current_waypoint = 0
    reached = False

    try:
        while True:
            if reached and current_waypoint < len(waypoints) - 1:
                current_waypoint += 1
                reached = False

            if not reached:
                reached = drive_to_waypoint(vehicle, waypoints[current_waypoint], world, display)
                update_camera()
                world.tick()
                time.sleep(0.05)
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    return
            world.tick()
    finally:
        print("Destroying actors.")
        for actor in [lidar_sensor, vehicle]:
            actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
