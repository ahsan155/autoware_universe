import rclpy
from rclpy.node import Node

from autoware_perception_msgs.msg import TrackedObjects
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import os
import numpy as np
from lanelet2.projection import LocalCartesianProjector
from lanelet2.io import load, Origin
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import joblib
import uuid
import carla
import torch
from collections import deque

from model_lane_input import get_centerline
from model_agent_input import build_model_inputs

from util import uuid_to_str

class MotionPredictionNode(Node):
    def __init__(self):
        super().__init__('motion_prediction_node')
        self.get_logger().info("Welcome to main node!")

        self.ego_pose = None  # Store the latest ego pose 
        self.agent_buffers = {}

        client = carla.Client('localhost', 2000) 
        self.world = client.get_world()


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

        PREDICTION_HZ = 10.0  # how often to run inference
        MAP_LOAD_HZ = 1.0
        self.prediction_timer = self.create_timer(1.0 / PREDICTION_HZ, self.prediction_callback)
        self.map_loader = self.create_timer(10.0, self.map_load_callback)
        self.lane_centerlines, self.lane_attrs, self.is_intersections = None, None, None

        # Load map and setup
        map_path = os.path.expanduser('~/Documents/town_10/backup/lanelet2_map.osm')
        assert os.path.exists(map_path), f"Map file not found at {map_path}"
        proj = LocalCartesianProjector(Origin(0, 0, 0))
        self.lanelet_map = load(map_path, proj)
        
        self.iteration = 1

    def get_buffer_template(self, last_timestamp, max_hist=50):
        buffer = {
            "pos": deque(maxlen=max_hist),
            "heading": deque(maxlen=max_hist),
            "velocity": deque(maxlen=max_hist),
            "last_timestamp": last_timestamp
        }
        return buffer

    def update_agent_history(self, agent_id, pos, heading, velocity, last_timestamp):
        """Update or create rolling history for an agent."""
        if agent_id not in self.agent_buffers:
            self.get_logger().info('creating buffer for new agent')
            self.agent_buffers[agent_id] = self.get_buffer_template(last_timestamp)

        self.agent_buffers[agent_id]["pos"].append(pos)
        self.agent_buffers[agent_id]["heading"].append(heading)
        self.agent_buffers[agent_id]["velocity"].append(velocity)
        self.agent_buffers[agent_id]["last_timestamp"] = last_timestamp

    def ego_pose_callback(self, msg):
        self.ego_pose = msg.pose.pose

    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        self.get_logger().info(f'Received {len(msg.objects)} tracked objects. {self.iteration}')
        self.iteration += 1
        
        for obj_id, obj in enumerate(msg.objects):  # iterate through detected objects
            agent_id = obj.object_id
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            agent_id = uuid_to_str(obj.object_id)

            if agent_id in self.agent_buffers.keys():
                last_time = self.agent_buffers[agent_id]["last_timestamp"]
                delta = timestamp - last_time
                if abs(delta - 0.1) > 0.5:  # Tolerance = 0.01s
                    self.get_logger().info(f"Resetting buffer for {agent_id} due to gap: {delta:.3f}s")
                    self.agent_buffers[agent_id] = self.get_buffer_template(timestamp)

            #position
            pos = obj.kinematics.pose_with_covariance.pose.position
            # Transform position from Autoware to CARLA coordinate system
            pos_x = pos.x  
            pos_y = pos.y  
            
            #orientation
            ori = obj.kinematics.pose_with_covariance.pose.orientation
            quat = (ori.x,ori.y,ori.z,ori.w)
            _, _, yaw = euler_from_quaternion(quat)
            yaw *= -1
            
            #velocity
            # Transform velocity from Autoware to CARLA coordinate system
            vx_o = obj.kinematics.twist_with_covariance.twist.linear.x
            vy_o = obj.kinematics.twist_with_covariance.twist.linear.y
            # 3) rotate into map frame
            vx_map = math.cos(yaw) * vx_o - math.sin(yaw) * vy_o
            vy_map = math.sin(yaw) * vx_o + math.cos(yaw) * vy_o
            vel_norm = np.linalg.norm([vx_map, vy_map])

            self.update_agent_history(
                agent_id,
                pos=np.array([pos_x, pos_y]),
                heading=yaw,
                velocity=vel_norm,
                last_timestamp=timestamp
            )


            
            #for vehicle in self.world.get_actors().filter('vehicle.tesla.model3'):
            #    print('tesla found')
            #    break

    def map_load_callback(self):
        ego_pos_x = self.ego_pose.position.x
        ego_pos_y = self.ego_pose.position.y
        self.lane_centerlines, self.lane_attrs, self.is_intersections = get_centerline(self.lanelet_map, ego_pos_x=ego_pos_x, ego_pos_y=ego_pos_y)

        

    def prediction_callback(self):
        succ, agent_state_data, lanelet_data = build_model_inputs(self.agent_buffers, self.lane_centerlines, self.lane_attrs, self.is_intersections)
        print('prediction data ready? ', succ)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
