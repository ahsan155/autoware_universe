import rclpy
from rclpy.node import Node

from autoware_auto_perception_msgs.msg import TrackedObjects
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math


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

    def ego_pose_callback(self, msg):
        self.ego_pose = msg.pose.pose


    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        ego_pos = self.ego_pose.position
        self.get_logger().info(f'Received {len(msg.objects)} tracked objects.')

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
                

            # carla velocity calculation-------------------------------------
            import carla
            client = carla.Client('localhost', 2000) 
            world = client.get_world()
            bp_lib = world.get_blueprint_library() 
            spawn_points = world.get_map().get_spawn_points() 
            for vehicle in world.get_actors().filter('vehicle.tesla.model3'):
                break

           

            self.get_logger().info(
                f'yaw from carla: {vehicle.get_transform().rotation.yaw} | '
                f'yaw from perception: {math.degrees(yaw)} | '  # Convert to degrees for comparison
                f'position from perception: ({pos_x:.2f}, {pos_y:.2f}) | '
                f'position from carla: ({vehicle.get_transform().location.x:.2f}, {vehicle.get_transform().location.y:.2f}) | '
                f'velocity from perception: ({vx_map:.2f}, {vy_map:.2f}) | '
                f'velocity from carla: ({vehicle.get_velocity().x}, {vehicle.get_velocity().y})'
            )
        print("-"*20)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
