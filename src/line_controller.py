#!/usr/bin/env python

import rospy
import numpy as np

from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped


class LineFollowingController():
    """
    """
    def __init__(self):
        rospy.Subscriber("/lookaheadpoint", Point, self.line_callback)

        DRIVE_TOPIC = rospy.get_param("drive_topic")
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

        self.parking_distance = 0.75  # meters
        self.relative_x = 0
        self.relative_y = 0

        self.prev_time = rospy.Time.now()

        self.prev_theta_err = 0.0
        self.prev_dist_err = 0.0

        self.running_theta_err = 0.0
        self.running_dist_err = 0.0

    def line_callback(self, msg):
        time = rospy.Time.now()
        dt = (time - self.prev_time).to_sec()
        self.prev_time = time

        self.relative_x = msg.x # x in front
        self.relative_y = msg.y # y to left

        dist = np.sqrt((self.relative_x**2.0)+(self.relative_y**2.0))
        dist_err = dist

        theta_err = np.arctan2(self.relative_y, self.relative_x) # theta to left

        self.running_theta_err += theta_err
        self.running_dist_err += dist_err

        d_theta_dt = (theta_err - self.prev_theta_err) / dt
        d_dist_dt = (dist_err - self.prev_dist_err) / dt

        self.prev_dist_err = dist_err
        self.prev_theta_err = theta_err

        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = "base_link"

        drive_cmd.drive.steering_angle = 0.4 * theta_err + \
            0.01 * d_theta_dt + 0.0 * self.running_theta_err

        drive_cmd.drive.speed = 1.0

        drive_cmd.drive.steering_angle_velocity = 0.0
        drive_cmd.drive.acceleration = 0.0
        drive_cmd.drive.jerk = 0.0

        self.drive_pub.publish(drive_cmd)


if __name__ == '__main__':
    try:
        rospy.init_node('line_follower', anonymous=True)
        lfc = LineFollowingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
