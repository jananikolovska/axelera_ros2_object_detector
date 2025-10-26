import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import sys
import os

class VideoStreamer(Node):
    """
    Streams video file frame-by-frame to simulate live camera feed.
    Automatically loops when video ends, maintains original FPS.
    """
    def __init__(self):
        super().__init__('video_streamer')
        self.publisher_ = self.create_publisher(Image, 'camera_frame', 10)
        self.bridge = CvBridge()

        # Path to your video file
        self.video_path = '../media/traffic1_1080p.mp4'
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {self.video_path}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # fallback in case FPS is not detected
        
        # Frame skipping for 60->30 FPS conversion
        self.frame_skip = 2 if fps == 60.0 else 1
        target_fps = fps / self.frame_skip
        
        self.get_logger().info(f"Original video FPS: {fps}, Publishing at: {target_fps} FPS (skip every {self.frame_skip} frames)")

        # Timer interval based on target FPS
        self.timer = self.create_timer(1.0 / target_fps, self.timer_callback)

    def timer_callback(self):
        # Read and skip frames according to frame_skip
        for _ in range(self.frame_skip):
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().info("End of video file reached. Rewinding...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

        # Publish the last frame read
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        
        # Add timestamp to header for latency calculation
        msg.header.stamp = self.get_clock().now().to_msg()
        
        self.publisher_.publish(msg)
        self.get_logger().info("Published video frame")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
