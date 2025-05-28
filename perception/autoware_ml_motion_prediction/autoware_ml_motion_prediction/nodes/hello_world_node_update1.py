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
import torch
from autoware_ml_motion_prediction.nodes.model.relative_model import EnhancedCombinedEncoder, MotionPredictionDecoder
from autoware_ml_motion_prediction.nodes.model.util import relative_to_global_trajectory_realtime
from autoware_ml_motion_prediction.nodes.publishers import TrajectoryPublisher
from autoware_ml_motion_prediction.nodes.util import uuid_to_str, autoware_to_carla_yaw, is_consecutive, calculate_autoware_lanelet_boundary_dists


class MotionPredictionNode(Node):
    def __init__(self):
        super().__init__('motion_prediction_node')
        
        #subscribers
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

        #publishers
        self.possible_path_pub = self.create_publisher(MarkerArray, "possible_path", 10)
        self.pred_trajectory_pub = TrajectoryPublisher()

        

        # Load map and setup
        map_path = os.path.expanduser('~/Documents/town_10/backup/lanelet2_map.osm')
        assert os.path.exists(map_path), f"Map file not found at {map_path}"
        proj = LocalCartesianProjector(Origin(0, 0, 0))
        self.lanelet_map = load(map_path, proj)
        self.graph = build_routing_graph(self.lanelet_map)


        self.ego_pose = None  # Store the latest ego pose
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

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = EnhancedCombinedEncoder(boundary_input_size=3, 
            vehicle_state_input_size=183, 
            hidden_size=128,
            final_hidden_size=256
        ).to(self.device)
        self.decoder = MotionPredictionDecoder(encoder=encoder,
            input_size=186, 
            hidden_size=256, 
        output_size=20).to(self.device)

        self.decoder.load_state_dict(torch.load("./relative_t_pt_model_ckpt_lr_001_e100_b32_update1.pt", weights_only=True))
        self.decoder.eval()

    def ego_pose_callback(self, msg):
        self.ego_pose = msg.pose.pose

    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        self.get_logger().info(f'Received {len(msg.objects)} tracked objects.')
        
        print("+"*20)
        for obj_id, obj in enumerate(msg.objects):  # iterate through detected objects

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
            temporary_heading = -3.12483163757340
            if (math.sqrt(vx_map**2 + vy_map**2)) < 0.1:
                heading = temporary_heading

            #vehicle blinker light
            right_blinker = int(False)
            left_blinker = int(False)

            #check if vehicle at traffic light
            vehicle_at_traffic_light = int(False)

            #getting vehicle to lane boundary distance
            center_lane_boundary_distance, right_lane_boundary_distance, left_lane_boundary_distance = calculate_autoware_lanelet_boundary_dists(self.lanelet_map, self.graph, pos_x, pos_y, sim_yaw_match)


            # scaling features
            scaled_pos_x = self.loaded_pos_x_scaler.transform([[pos_x]])[0][0]
            scaled_pos_y = self.loaded_pos_y_scaler.transform([[pos_y]])[0][0]
            scaled_vel_x = self.loaded_vel_x_scaler.transform([[vx_map]])[0][0]
            scaled_vel_y = self.loaded_vel_y_scaler.transform([[vy_map]])[0][0]
            scaled_yaw = self.loaded_yaw_scaler.transform([[sim_yaw_match]])[0][0]
            scaled_heading = self.loaded_heading_scaler.transform([[heading]])[0][0]
            scaled_dist_center_bound = self.loaded_boundary_distance_scaler.transform([[center_lane_boundary_distance]])[0][0]
            scaled_dist_left_bound = self.loaded_boundary_distance_scaler.transform([[left_lane_boundary_distance]])[0][0]
            scaled_dist_right_bound = self.loaded_boundary_distance_scaler.transform([[right_lane_boundary_distance]])[0][0]


            # get possible trajectories--------------------------------------------------------------------
            current_lanelets = find_current_lanelets(self.lanelet_map, pos_x, pos_y)
            raw_paths        = [lanelet_sequence_to_trajectory(path, step=0.5)
                                for ll in current_lanelets
                                for path in get_candidate_paths(self.lanelet_map, self.graph, ll, 60.0)]
            # 1) slice full paths first
            sliced_full = [
                slice_trajectory_ahead_vec(np.array(p)[:, :2], (pos_x, pos_y))
                for p in raw_paths
            ]
            # 2) down-sample (or skip entirely)
            sliced_paths = [traj if len(traj) < 5 else traj[4::5] for traj in sliced_full]
            # 3) filter by direction
            filtered = filter_trajectories_by_initial_direction(sliced_paths, sim_yaw_match, 60.0)
            if not filtered:
                filtered = filter_trajectories_by_initial_direction(sliced_paths, sim_yaw_match, 120.0)

            scaled_possible_trajectories = preprocess_and_vectorize_paths(
                filtered, [pos_x, pos_y], num_paths=3, path_length=29,
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
                inp_tensor = torch.tensor(stacked, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    decoder_output, _, _ = self.decoder(inp_tensor, None, mode="test")
                    decoder_output = relative_to_global_trajectory_realtime(inp_tensor, decoder_output)

                    decoder_output_x = decoder_output.cpu().reshape(10,2)[:,0]
                    decoder_output_x = [self.loaded_pos_x_scaler.inverse_transform([[item]])[0][0] for item in decoder_output_x]
                    decoder_output_y = decoder_output.cpu().reshape(10,2)[:,1]
                    decoder_output_y = [self.loaded_pos_y_scaler.inverse_transform([[item]])[0][0] for item in decoder_output_y]
                    self.trajectory_publisher.publish_trajectory([decoder_output_x, decoder_output_y], obj_id)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
