#!/usr/bin/env python

import rospy
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

# See lab4/homography_data for images and data used

#The following collection of pixel locations and corresponding relative
#ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
PTS_IMAGE_PLANE = [[223, 321],
                   [366, 294],
                   [515, 328],
                   [155, 274],
                   [285, 260],
                   [379, 246],
                   [524, 253],
                   [300, 218],
                   [388, 219]]
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
PTS_GROUND_PLANE = [[15.5,   4.0],
                    [17.75, -3.5],
                    [13.0,  -9.5],
                    [22.0,   10.75],
                    [24.5,   2.0],
                    [28.0,  -5.5],
                    [24.5,  -16.0],
                    [41.75,  2.25],
                    [39.5,  -8.0]]
######################################################

METERS_PER_INCH = 0.0254


class HomographyTransformer:
    def __init__(self):
        self.line_px_sub = rospy.Subscriber("/line_px", Point, self.line_detection_callback)
        self.lookahead_pub = rospy.Publisher("/lookaheadpoint", PointStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/viz", Marker, queue_size=1)

        if not len(PTS_GROUND_PLANE) == len(PTS_IMAGE_PLANE):
            rospy.logerr("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

        #Initialize data into a homography matrix
        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        # Prepare for transforms between base_link and left_zed_camera
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def line_detection_callback(self, msg):
        """
        """
        # Pixel coordinates
        u = msg.x
        v = msg.y

        # Transform and viz world coordinates
        x, y = self.pixel_to_world(u, v)
        self.draw_marker(x, y, "base_link")
        pt_cam = PointStamped()
        pt_cam.header.stamp = rospy.Time.now()
        pt_cam.header.frame_id = "left_zed_camera"
        pt_cam.x = x
        pt_cam.y = y
        pt_world = self.transform_to_car(pt_cam)

        self.lookahead_pub.publish(pt_world)

    def pixel_to_world(self, u, v):
        """
        Transform pixel coordinates (u,v) to world coordinates (x,y)
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

    def transform_to_car(self, point):
        """
        Takes a PointStamped message and transforms it from the its frame_id frame
        into the base_link frame, which is where the controller expects it.
        """
        try:
            point_transformed = self.tf_buffer.transform(point, "base_link", rospy.Duration(0.1))
            return point_transformed
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            raise

    def draw_marker(self, x, y, message_frame):
        """
        Publish a marker to represent the lookahead point in rviz.
        """
        marker = Marker()
        marker.header.frame_id = message_frame
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        self.marker_pub.publish(marker)


if __name__ == "__main__":
    rospy.init_node('homography')
    homography_transformer = HomographyTransformer()
    rospy.spin()
