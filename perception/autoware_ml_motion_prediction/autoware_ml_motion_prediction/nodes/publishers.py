from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


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