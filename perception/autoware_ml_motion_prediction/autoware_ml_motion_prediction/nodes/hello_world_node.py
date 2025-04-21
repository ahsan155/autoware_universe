import rclpy
from rclpy.node import Node

from autoware_auto_perception_msgs.msg import TrackedObjects
from autoware_auto_vehicle_msgs.msg import VehicleKinematicState

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
            VehicleKinematicState,
            '/localization/kinematic_state',
            self.ego_pose_callback,
            10
        )

        self.ego_pose = None  # Store the latest ego pose

    def ego_pose_callback(self, msg):
        self.ego_pose = msg.state.pose


    def objects_callback(self, msg):
        if self.ego_pose is None:
            self.get_logger().info("Waiting for ego pose...")
            return

        ego_pos = self.ego_pose.position

        self.get_logger().info(f'Received {len(msg.objects)} tracked objects.')

        print("+"*20)
        for obj in msg.objects:
            pos = obj.kinematics.pose_with_covariance.pose.position
            rel_x = pos.x - ego_pos.x
            rel_y = pos.y - ego_pos.y
            rel_z = pos.z - ego_pos.z

            ori = obj.kinematics.pose_with_covariance.pose.orientation
            vel = obj.kinematics.twist_with_covariance.twist.linear

            self.get_logger().info(
                f'Object ID: {obj.object_id} | '
                f'Relative Pos: ({rel_x:.2f}, {rel_y:.2f}, {rel_z:.2f}) | '
                f'Orientation: ({ori.x:.2f}, {ori.y:.2f}, {ori.z:.2f}, {ori.w:.2f}) | '
                f'Velocity: ({vel.x:.2f}, {vel.y:.2f}, {vel.z:.2f})'
            )
        print("-"*20)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPredictionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
