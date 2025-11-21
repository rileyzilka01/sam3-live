#!/usr/bin/python3

import rospy
import zmq
import json
import numpy as np
import message_filters
from sensor_msgs.msg import PointCloud2, JointState, Image
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Int16
import sys
import ros_numpy
import msgpack
import cv2
from cv_bridge import CvBridge

import time
import math
import threading
from queue import Queue, Empty


class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node("image_segmentation_node")

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:4444")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.socket.setsockopt(zmq.RCVTIMEO, -1)  # indefinite wait
        self.socket.setsockopt(zmq.LINGER, 0)       # Don't hang on close if server is dead

        self.bridge = CvBridge()

        self.img = None
        self.busy = False

        # Send a ping test message
        try:
            ping = {"ping": True}
            self.socket.send_json(ping)

            resp = self.socket.recv_json()
            rospy.loginfo(f"✅ Connected to inference server: {resp}")
        except zmq.Again:
            rospy.logerr("❌ Could not connect to inference server (timeout)")
            sys.exit(1)

        # ROS publishers and subscribers
        self.cam_sub = rospy.Subscriber("/cam/color/image_raw", Image, self.callback)
        self.seg_pub = rospy.Publisher("/my_gen3/segment_mask", Image, queue_size=1)

        rospy.loginfo("🤖 Segmentation node initialized")


    def callback(self, img_msg):
        self.img = img_msg
        rospy.loginfo("Received image")

    def process_image(self):
        if self.img is None:
            rospy.loginfo("No image yet")
            self.busy = False
            return

        try: 
            # Convert ROS Image -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(self.img, "bgr8")
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape
            img_bytes = img_rgb.tobytes()

            header = {"height": h, "width": w, "channels": c}

            prompts = ["black end effector on robot arm", "cube", "plate", "screwdriver"]
            prompts_json = json.dumps({"prompts": prompts}).encode("utf-8")

            rospy.loginfo("Sending...")
            self.socket.send_multipart([
                json.dumps(header).encode("utf-8"),
                img_bytes,
                prompts_json
            ])

            reply_parts = self.socket.recv_multipart()
            reply_header = json.loads(reply_parts[0].decode("utf-8"))
            mask_bytes = reply_parts[1]

            rospy.loginfo(f"Server reply header: {reply_header}")
            rospy.loginfo(f"Mask bytes length: {len(mask_bytes)}")

            mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)  # or dtype=header['dtype']
            mask_array = mask_array.reshape(header['height'], header['width'])
            mask_array = (mask_array * 255).astype(np.uint8)

            ros_image_msg = self.bridge.cv2_to_imgmsg(mask_array, encoding="mono8")
            self.seg_pub.publish(ros_image_msg)

            self.busy = False

        except Exception as e:
            rospy.logerr(f"Error during ZMQ send/recv: {e}")


if __name__ == '__main__':
    try:
        node = ImageSegmentationNode()

        rate = rospy.Rate(30)  # 30 Hz loop, adjust as needed
        while not rospy.is_shutdown():
            # Grab the latest image and process it
            if node.img is not None:
                node.process_image()
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    except rospy.ROSInterruptException:
        pass