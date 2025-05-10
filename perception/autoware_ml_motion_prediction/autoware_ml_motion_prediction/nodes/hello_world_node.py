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


    def ego_pose_callback(self, msg):
        self.ego_pose = msg.pose.pose


    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        ego_pos = self.ego_pose.position
        self.get_logger().info(f'Received {len(msg.objects)} tracked objects.')
        marker_array = MarkerArray()
        
        print("+"*20)
        for obj in msg.objects:  # iterate through detected objects
            #position
            pos = obj.kinematics.pose_with_covariance.pose.position
            # Transform position from Autoware to CARLA coordinate system
            pos_x = pos.x  # Keep x as is
            pos_y = -pos.y  # Flip y sign
            
            #orientation
            ori = obj.kinematics.pose_with_covariance.pose.orientation
            quat = (ori.x,ori.y,ori.z,ori.w)
            _, _, yaw = euler_from_quaternion(quat)
            yaw = -yaw
            
            #velocity
            # Transform velocity from Autoware to CARLA coordinate system
            vx_o = obj.kinematics.twist_with_covariance.twist.linear.x
            vy_o = obj.kinematics.twist_with_covariance.twist.linear.y
            # 3) rotate into map frame
            vx_map = math.cos(yaw) * vx_o - math.sin(yaw) * vy_o
            vy_map = math.sin(yaw) * vx_o + math.cos(yaw) * vy_o

            # get possible trajectories
            current_lanelets = find_current_lanelets(self.lanelet_map, pos_x, -pos_y)
            graph = build_routing_graph(self.lanelet_map)
            raw_paths = []
            for lanelet in current_lanelets:
                paths = get_candidate_paths(self.lanelet_map, graph, lanelet, max_distance=50.0)
                for path in paths:
                    traj = lanelet_sequence_to_trajectory(path, step=0.5)
                    raw_paths.append(traj)
                    
            ego_yaw = math.radians(-yaw)
            filtered_paths = filter_trajectories_by_initial_direction(raw_paths, ego_yaw, max_angle_deg=60.0)
            if not filtered_paths:
                # you just turned — allow up to 120° until you’re fully on the new lane
                filtered_paths = filter_trajectories_by_initial_direction(raw_paths, ego_yaw, max_angle_deg=120.0)

            ego_xy = (pos_x, -pos_y)
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
