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
from autoware_ml_motion_prediction.nodes.util import calculate_autoware_lanelet_boundary_dists_with_next
from autoware_ml_motion_prediction.nodes.model.carla_functions import calculate_vehicle_land_boundary_distance, get_extended_trajectories, create_trajectory, pad_trajectories, global_to_relative_possible_trajectory


class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        self.publisher = self.create_publisher(Marker, 'future_trajectory', 10)
        
    def publish_trajectory(self, trajectory, obj_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "future_trajectory"
        marker.id = obj_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Line width
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        for x,y in zip(trajectory[0], trajectory[1]):
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            marker.points.append(point)
        
        self.publisher.publish(marker)

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
        self.graph = build_routing_graph(self.lanelet_map)


        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)
        self.buffers = {}
        self.map_cache = {}

        client = carla.Client('localhost', 2000) 
        self.world = client.get_world()

        # Load the scalers
        node_dir = os.path.dirname(os.path.realpath(__file__))
        self.loaded_pos_x_scaler = joblib.load(os.path.join(node_dir,'scalers_large/pos_x_scaler.save'))
        self.loaded_pos_y_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/pos_y_scaler.save'))
        self.loaded_vel_x_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/vel_x_scaler.save'))
        self.loaded_vel_y_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/vel_y_scaler.save'))
        self.loaded_yaw_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/yaw_scaler.save'))
        self.loaded_heading_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/heading_scaler.save'))
        self.loaded_traj_x_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/traj_x_scaler.save'))
        self.loaded_traj_y_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/traj_y_scaler.save'))
        self.loaded_boundary_distance_scaler = joblib.load(os.path.join(node_dir,'./scalers_large/boundary_distance_scaler.save'))

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

        self.decoder.load_state_dict(torch.load(os.path.join(node_dir,"model/relative_t_pt_model_ckpt_lr_001_e100_b32_update1.pt"), weights_only=True))
        self.decoder.eval()

        self.trajectory_publisher = TrajectoryPublisher()



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
        np_paths = [np.array(p, dtype=np.float32) for p in paths]

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
        # … collect state, cache map‐based features …
        to_predict      = []      # list of (agent_id, feature_seq)
        agent_indices   = []      # keep track of agent_ids in order
        
        print("+"*20)
        for obj_id, obj in enumerate(msg.objects):  # iterate through detected objects

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
            sim_yaw_match = (sim_yaw_match + 180) % 360 - 180
           
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


            # 3) Look up or initialize this agent’s cache
            cache = self.map_cache.get(agent_id)
            if cache is None:
                cache = {
                    'x': pos_x,
                    'y': pos_y,
                    'yaw': sim_yaw_match,
                    'distances': None,
                    'paths': None,
                    'agent_active': False,
                    'path_bound_calculation_done' : False
                }
                self.map_cache[agent_id] = cache

            print('xx', cache["agent_active"], agent_id)

            # 4) Compute how much the agent moved/rotated since last map‐work
            dist_moved = math.hypot(pos_x - cache['x'], pos_y - cache['y'])
            yaw_diff   = abs(sim_yaw_match - cache['yaw'])

            for vehicle in self.world.get_actors().filter('vehicle.tesla.model3'):
                '''
                print("carla vehicle loc", vehicle.get_location())
                print("autoware vehicle loc", pos_x, pos_y)
                print("carla vehicle velocity", vehicle.get_velocity())
                print("autoware vehicle velocity", vx_map, vy_map)
                print("carla yaw degree", vehicle.get_transform().rotation.yaw, "carla yaw radian:", -math.radians(vehicle.get_transform().rotation.yaw))
                print("autoware yaw degree:", sim_yaw_match, "autoware yaw radian:", yaw)
                
                
                carla_heading = math.atan2(vehicle.get_velocity().y, vehicle.get_velocity().x)
                vel_x, vel_y = vehicle.get_velocity().x, -vehicle.get_velocity().y
                if (math.sqrt(vel_x**2 + vel_y**2)) < 0.1:
                    carla_heading = temporary_heading

                #print("carla heading", carla_heading)
                #print("autoware heading", heading)
                

                
                pos_x, pos_y = vehicle.get_location().x, -vehicle.get_location().y
                vx_map, vy_map = vehicle.get_velocity().x, -vehicle.get_velocity().y
                vehicle_transform = vehicle.get_transform()
                orientation = vehicle_transform.rotation
                roll, pitch, sim_yaw_match = orientation.roll, orientation.pitch, orientation.yaw
                heading = math.atan2(vehicle.get_velocity().y, vehicle.get_velocity().x)
                if (math.sqrt(vel_x**2 + vel_y**2)) < 0.1:
                    heading = temporary_heading
                '''

            '''
            RIGHT_BLINKER_POS = 4
            LEFT_BLINKER_POS = 5
            light_state = vehicle.get_light_state()
            right_blinker = bool(light_state & (0x1 << RIGHT_BLINKER_POS))
            left_blinker = bool(light_state & (0x1 << LEFT_BLINKER_POS))
            '''

            #vehicle blinker light
            right_blinker = int(False)
            left_blinker = int(False)
            vehicle_at_traffic_light = int(False)


            #center_lane_boundary_distance, right_lane_boundary_distance, left_lane_boundary_distance = calculate_vehicle_land_boundary_distance(vehicle, self.world)
            
            '''
            pos_x_scaled = self.loaded_pos_x_scaler.transform([[pos_x]])
            pos_y_scaled = self.loaded_pos_y_scaler.transform([[pos_y]])
            possible_trajectories = get_extended_trajectories(vehicle, self.world)
            possible_trajectories = create_trajectory(possible_trajectories)
            possible_trajectories = pad_trajectories(possible_trajectories)
            possible_trajectories = np.array(possible_trajectories)[:, :29, :]
            x_coords = possible_trajectories[:, :, 0].reshape(-1, 1)
            y_coords = possible_trajectories[:, :, 1].reshape(-1, 1)
            scaled_x = self.loaded_traj_x_scaler.transform(x_coords)
            scaled_y = self.loaded_traj_y_scaler.transform(y_coords)
            scaled_trajectories = np.hstack((scaled_x, scaled_y))
            possible_trajectories_scaled = scaled_trajectories.reshape(3, 29, 2)
            relative_possible_t1 = global_to_relative_possible_trajectory(possible_trajectories_scaled[0, :, :].flatten(), np.array([pos_x_scaled[0], pos_y_scaled[0]]).squeeze(1))
            relative_possible_t2 = global_to_relative_possible_trajectory(possible_trajectories_scaled[1, :, :].flatten(), np.array([pos_x_scaled[0], pos_y_scaled[0]]).squeeze(1))
            relative_possible_t3 = global_to_relative_possible_trajectory(possible_trajectories_scaled[2, :, :].flatten(), np.array([pos_x_scaled[0], pos_y_scaled[0]]).squeeze(1))
            total_relative_possible_t = np.concatenate([relative_possible_t1, relative_possible_t2, relative_possible_t3])
            '''

            
            #If moved > thresh or rotated > thresh, recompute expensive features--------------------------------------------------------------------
            if dist_moved > 0.5 or yaw_diff > 5.0 or cache["agent_active"] == False:
                cache["agent_active"] = True
                #get possible trajectories---------
                current_lanelets = find_current_lanelets(self.lanelet_map, pos_x, pos_y)
                graph            = build_routing_graph(self.lanelet_map)
                raw_paths        = [lanelet_sequence_to_trajectory(path, step=0.5)
                                    for ll in current_lanelets
                                    for path in get_candidate_paths(self.lanelet_map, graph, ll, 60.0)]
                
                # 1) slice full paths first
                sliced_full = [
                    slice_trajectory_ahead_vec(np.array(p)[:, :2], (pos_x, pos_y))
                    for p in raw_paths
                ]
                # 2) down-sample (or skip entirely)
                sliced_paths = [traj if len(traj) < 5 else traj[4::5] for traj in sliced_full]
                # 3) filter by direction
                filtered = filter_trajectories_by_initial_direction(sliced_paths, yaw, 60.0)
                if not filtered:
                    filtered = filter_trajectories_by_initial_direction(sliced_paths, yaw, 120.0)

                if not filtered:
                    print("cont check...")
                    cache.update({'path_bound_calculation_done': False})
                    continue
                

                scaled_possible_trajectories = self.preprocess_and_vectorize_paths(
                    filtered, [pos_x, pos_y], num_paths=3, path_length=29,
                    traj_x_scaler=self.loaded_traj_x_scaler, traj_y_scaler=self.loaded_traj_y_scaler,
                    pos_x_scaler=self.loaded_pos_x_scaler, pos_y_scaler=self.loaded_pos_y_scaler,
                )

                #vehicle to boundary distance---------
                center_lane_boundary_distance, right_lane_boundary_distance, left_lane_boundary_distance = calculate_autoware_lanelet_boundary_dists_with_next(self.lanelet_map, self.graph, pos_x, pos_y, -sim_yaw_match)
                print("yy", center_lane_boundary_distance, right_lane_boundary_distance, left_lane_boundary_distance)

                # store back in cache
                cache.update({
                    'x': pos_x,
                    'y': pos_y,
                    'yaw': sim_yaw_match,
                    'distances': (center_lane_boundary_distance, right_lane_boundary_distance, left_lane_boundary_distance),
                    'paths': scaled_possible_trajectories,
                    'path_bound_calculation_done': True
                })


            # 6) Now pull from cache for the rest of your pipeline
            if cache["path_bound_calculation_done"]:
                center_lane_boundary_distance, right_lane_boundary_distance, left_lane_boundary_distance = cache['distances']
                scaled_possible_trajectories = cache['paths']
            else:
                continue

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
                
                '''
                features_to_stack = [entry[1] for entry in self.buffers[agent_id]]
                stacked = np.stack(features_to_stack, axis=0)  # shape will be (5, 186)
                stacked = np.expand_dims(stacked, axis=0)
                inp_tensor = torch.tensor(stacked, dtype=torch.float32).to(self.device)
                '''
                
                feat_seq = [entry[1] for entry in self.buffers[agent_id]]  # five (feat_dim,) arrays
                to_predict.append(feat_seq)
                agent_indices.append(agent_id)

                '''
                with torch.no_grad():
                    decoder_output, _, _ = self.decoder(inp_tensor, None, mode="test")
                    decoder_output = relative_to_global_trajectory_realtime(inp_tensor, decoder_output)

                    decoder_output_x = decoder_output.cpu().reshape(10,2)[:,0]
                    decoder_output_x = [self.loaded_pos_x_scaler.inverse_transform([[item]])[0][0] for item in decoder_output_x]
                    decoder_output_y = decoder_output.cpu().reshape(10,2)[:,1]
                    decoder_output_y = [self.loaded_pos_y_scaler.inverse_transform([[item]])[0][0] for item in decoder_output_y]
                    self.trajectory_publisher.publish_trajectory([decoder_output_x, decoder_output_y], obj_id)
                '''


            '''
            marker_array.markers.clear()
            for idx, traj in enumerate(filtered):
                trimmed = np.array(traj).tolist()
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"trajectories_{idx + obj_id}"
                marker.id = idx + obj_id
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
            '''


        # if no one is ready, bail out
        if not to_predict:
            return

        # 1) Build a batch: shape (N_agents, 5, feat_dim)
        batch = torch.tensor(to_predict, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            decoder_output, _, _ = self.decoder(batch, None, mode="test")
            global_preds = relative_to_global_trajectory_realtime(batch, decoder_output)

        # 4) Publish per agent
        for i, agent_id in enumerate(agent_indices):
            traj = global_preds[i].cpu().numpy()  # shape (future_len, 2)

            '''
            xs = [self.loaded_pos_x_scaler.inverse_transform([[x]])[0][0] for x in traj[:,0]]
            ys = [self.loaded_pos_y_scaler.inverse_transform([[y]])[0][0] for y in traj[:,1]]
            print("traj shape", traj.reshape(10,2).shape)
            self.trajectory_publisher.publish_trajectory([xs, ys], i)
            '''
            decoder_output_x = traj.reshape(10,2)[:,0]
            decoder_output_x = [self.loaded_pos_x_scaler.inverse_transform([[item]])[0][0] for item in decoder_output_x]
            decoder_output_y = traj.reshape(10,2)[:,1]
            decoder_output_y = [self.loaded_pos_y_scaler.inverse_transform([[item]])[0][0] for item in decoder_output_y]
            self.trajectory_publisher.publish_trajectory([decoder_output_x, decoder_output_y], i)
        
        print("-"*20)



def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
