import rclpy
from rclpy.node import Node

from autoware_auto_perception_msgs.msg import TrackedObjects
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import os
import numpy as np
from lanelet2.projection import LocalCartesianProjector
from lanelet2.io import load, Origin
from autoware_ml_motion_prediction.nodes.path_generator import *
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import joblib
import uuid
import carla


class MotionPredictionNode(Node):
    def __init__(self):
        super().__init__('motion_prediction_node')
        self.subscription = self.create_subscription(
            TrackedObjects,
            '/perception/object_recognition/tracking/objects',
            self.objects_callback,
            10
        )

        self.ego_pose_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.ego_pose_callback,
            1
        )
        
        self.ego_pose = None  # Store the latest ego pose

        # Load map and setup
        map_path = os.path.expanduser('~/Documents/town_10/backup/lanelet2_map.osm')
        assert os.path.exists(map_path), f"Map file not found at {map_path}"
        proj = LocalCartesianProjector(Origin(0, 0, 0))
        self.lanelet_map = load(map_path, proj)

        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)
        self.buffers = {}

        client = carla.Client('localhost', 2000) 
        self.world = client.get_world()

        # Load the scalers
        self.loaded_pos_x_scaler = joblib.load('./scalers_large/pos_x_scaler.save')
        self.loaded_pos_y_scaler = joblib.load('./scalers_large/pos_y_scaler.save')
        self.loaded_vel_x_scaler = joblib.load('./scalers_large/vel_x_scaler.save')
        self.loaded_vel_y_scaler = joblib.load('./scalers_large/vel_y_scaler.save')
        self.loaded_yaw_scaler = joblib.load('./scalers_large/yaw_scaler.save')
        self.loaded_heading_scaler = joblib.load('./scalers_large/heading_scaler.save')
        self.loaded_traj_x_scaler = joblib.load('./scalers_large/traj_x_scaler.save')
        self.loaded_traj_y_scaler = joblib.load('./scalers_large/traj_y_scaler.save')
        self.loaded_boundary_distance_scaler = joblib.load('./scalers_large/boundary_distance_scaler.save')


    def ego_pose_callback(self, msg):
        self.ego_pose = msg.pose.pose

    def is_consecutive(self, timestamps, expected_delta=0.1, tolerance=0.01):
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i-1]
            if abs(delta - expected_delta) > tolerance:
                return False
        return True

    def uuid_to_str(self, uuid_msg):
        # uuid_msg.uuid is a numpy array of 16 bytes
        return str(uuid.UUID(bytes=bytes(uuid_msg.uuid)))
    

    def preprocess_and_vectorize_paths(
        self, 
        paths, 
        reference_point, 
        num_paths=3, 
        path_length=29,
        traj_x_scaler=None,
        traj_y_scaler=None,
        pos_x_scaler=None,
        pos_y_scaler=None):
        """
        Preprocess paths, convert to relative, flatten, and concatenate.
        Args:
            paths: List of list of (x, y) tuples.
            reference_point: (x, y) tuple or np.array of shape (2,)
            num_paths: Number of paths to output (default 3)
            path_length: Number of points per path (default 29)
        Returns:
            np.array of shape (num_paths * path_length * 2,)
        """
        # Step 1: Convert input to numpy arrays for vectorization
        np_paths = [np.array(p, dtype=np.float32)[4::5,:2] for p in paths]

        # Step 2: Normalize x and y in each path using trajectory scalers
        if traj_x_scaler is not None and traj_y_scaler is not None:
            for i, path in enumerate(np_paths):
                if len(path) > 0:
                    path[:, 0] = traj_x_scaler.transform(path[:, 0].reshape(-1, 1)).flatten()
                    path[:, 1] = traj_y_scaler.transform(path[:, 1].reshape(-1, 1)).flatten()
                    np_paths[i] = path

        # Step 3: Ensure exactly num_paths
        if len(np_paths) > num_paths:
            np_paths = np_paths[:num_paths]
        elif len(np_paths) < num_paths:
            while len(np_paths) < num_paths:
                np_paths.append(np_paths[-1].copy())
           
        # Step 4: Pad/truncate each path to path_length
        processed_paths = []
        for path in np_paths:
            if path.shape[0] > path_length:
                path = path[:path_length]
            elif path.shape[0] < path_length:
                if path.shape[0] > 0:
                    pad = np.tile(path[-1], (path_length - path.shape[0], 1))
                    path = np.vstack([path, pad])
                else:
                    path = np.zeros((path_length, 2), dtype=np.float32)
            processed_paths.append(path)

        # Step 4: Convert to relative, flatten, and concatenate
        reference_point = np.array(reference_point, dtype=np.float32)
        if pos_x_scaler is not None and pos_y_scaler is not None:
            ref_x = pos_x_scaler.transform([[reference_point[0]]])[0, 0]
            ref_y = pos_y_scaler.transform([[reference_point[1]]])[0, 0]
            reference_point = np.array([ref_x, ref_y], dtype=np.float32)

        rel_flattened = []
        for path in processed_paths:
            rel_path = path - reference_point  # (29, 2)
            rel_flattened.append(rel_path.flatten())  # (58,)
        result = np.concatenate(rel_flattened)  # (174,)
        return result

    def autoware_to_carla_yaw(self, autoware_yaw_deg):
        """
        Convert Autoware yaw (in degrees, CCW-positive, ENU) to Carla style
        yaw (degrees, CW-positive, Unreal frame).
        """
        carla_yaw = -autoware_yaw_deg
        return (carla_yaw + 180) % 360 - 180



    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        ego_pos = self.ego_pose.position
        self.get_logger().info(f'Received {len(msg.objects)} tracked objects.')
        marker_array = MarkerArray()
        
        print("+"*20)
        for obj in msg.objects:  # iterate through detected objects

            agent_id = obj.object_id
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            agent_id = self.uuid_to_str(obj.object_id)
            if agent_id not in self.buffers:
                self.buffers[agent_id] = []

            #position
            pos = obj.kinematics.pose_with_covariance.pose.position
            # Transform position from Autoware to CARLA coordinate system
            pos_x = pos.x  
            pos_y = pos.y  
            
            #orientation
            ori = obj.kinematics.pose_with_covariance.pose.orientation
            quat = (ori.x,ori.y,ori.z,ori.w)
            _, _, yaw = euler_from_quaternion(quat)
            sim_yaw_match = -math.degrees(yaw)
           
            #velocity
            # Transform velocity from Autoware to CARLA coordinate system
            vx_o = obj.kinematics.twist_with_covariance.twist.linear.x
            vy_o = obj.kinematics.twist_with_covariance.twist.linear.y
            # 3) rotate into map frame
            vx_map = math.cos(yaw) * vx_o - math.sin(yaw) * vy_o
            vy_map = math.sin(yaw) * vx_o + math.cos(yaw) * vy_o

            #heading
            heading = math.atan2(-vy_map, vx_map)
           
            #vehicle blinker light
            right_blinker = int(False)
            left_blinker = int(False)

            #vehicle to boundary distance
            center_lane_boundary_distance = 0.0
            right_lane_boundary_distance = 0.0
            left_lane_boundary_distance = 0.0

            vehicle_at_traffic_light = int(False)

        # scaling vehicle state data
            scaled_pos_x = self.loaded_pos_x_scaler.transform([[pos_x]])[0][0]
            scaled_pos_y = self.loaded_pos_y_scaler.transform([[pos_y]])[0][0]
            scaled_vel_x = self.loaded_vel_x_scaler.transform([[vx_map]])[0][0]
            scaled_vel_y = self.loaded_vel_y_scaler.transform([[vy_map]])[0][0]
            scaled_yaw = self.loaded_yaw_scaler.transform([[sim_yaw_match]])[0][0]
            scaled_heading = self.loaded_heading_scaler.transform([[heading]])[0][0]
            scaled_dist_center_bound = self.loaded_boundary_distance_scaler.transform([[center_lane_boundary_distance]])[0][0]
            scaled_dist_left_bound = self.loaded_boundary_distance_scaler.transform([[left_lane_boundary_distance]])[0][0]
            scaled_dist_right_bound = self.loaded_boundary_distance_scaler.transform([[right_lane_boundary_distance]])[0][0]


            for vehicle in self.world.get_actors().filter('vehicle.tesla.model3'):
                #print("carla vehicle loc", vehicle.get_location())
                #print("autoware vehicle loc", pos_x, pos_y)
                #print("carla vehicle velocity", vehicle.get_velocity())
                #print("autoware vehicle velocity", vx_map, vy_map)
                print("carla yaw", vehicle.get_transform().rotation.yaw)
                print("autoware yaw", -math.degrees(yaw))

        # get possible trajectories--------------------------------------------------------------------
            current_lanelets = find_current_lanelets(self.lanelet_map, pos_x, pos_y)
            graph = build_routing_graph(self.lanelet_map)
            raw_paths = []
            for lanelet in current_lanelets:
                paths = get_candidate_paths(self.lanelet_map, graph, lanelet, max_distance=50.0)
                for path in paths:
                    traj = lanelet_sequence_to_trajectory(path, step=0.5)
                    raw_paths.append(traj)
                    
            ego_yaw = math.radians(yaw)
            filtered_paths = filter_trajectories_by_initial_direction(raw_paths, ego_yaw, max_angle_deg=60.0)
            if not filtered_paths:
                # you just turned — allow up to 120° until you’re fully on the new lane
                filtered_paths = filter_trajectories_by_initial_direction(raw_paths, ego_yaw, max_angle_deg=120.0)

            scaled_possible_trajectories = self.preprocess_and_vectorize_paths(
                filtered_paths, [pos_x, pos_y], num_paths=3, path_length=29,
                traj_x_scaler=self.loaded_traj_x_scaler, traj_y_scaler=self.loaded_traj_y_scaler,
                pos_x_scaler=self.loaded_pos_x_scaler, pos_y_scaler=self.loaded_pos_y_scaler,
            )

        # combining vehicle state data and map info.
            features = [
                scaled_dist_center_bound, scaled_dist_right_bound, scaled_dist_left_bound,
                scaled_pos_x, scaled_pos_y, scaled_vel_x, scaled_vel_y, scaled_yaw, scaled_heading,
                right_blinker, left_blinker, vehicle_at_traffic_light
            ]
            total_scaled_features = np.concatenate([features, scaled_possible_trajectories])

        # Check gap for existing buffer entries
            if self.buffers[agent_id]:
                last_time = self.buffers[agent_id][-1][0]
                delta = timestamp - last_time
                if abs(delta - 0.1) > 0.01:  # Tolerance = 0.01s
                    self.get_logger().info(f"Resetting buffer for {agent_id} due to gap: {delta:.3f}s")
                    self.buffers[agent_id] = []
            
            self.buffers[agent_id].append((timestamp, total_scaled_features))
            # Trim buffer to last N entries (e.g., N=5)
            self.buffers[agent_id] = self.buffers[agent_id][-5:]
            
            # Proceed only if buffer has N consecutive entries
            if len(self.buffers[agent_id]) == 5 and self.is_consecutive([t for t, _ in self.buffers[agent_id]]):
                # Run prediction
                features_to_stack = [entry[1] for entry in self.buffers[agent_id]]
                stacked = np.stack(features_to_stack, axis=0)  # shape will be (5, 186)
                stacked = np.expand_dims(stacked, axis=0)
                print('nn', stacked.shape)

            

            ego_xy = (pos_x, pos_y)
            for idx, traj in enumerate(filtered_paths):
                possible_trajectory = [(p[0], p[1]) for p in traj]
                trimmed = slice_trajectory_ahead_vec(possible_trajectory, ego_xy)
                trimmed = np.array(trimmed)[4::5, :].tolist()

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "trajectories"
                marker.id = idx
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.1  # Line width
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.lifetime.sec = 0  # 0 means forever

                for p in trimmed:
                    pt = Point()
                    pt.x, pt.y = p[0], p[1]
                    pt.z = 0.0  # Set to actual z if available
                    marker.points.append(pt)
                
                marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
        print("-"*20)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
