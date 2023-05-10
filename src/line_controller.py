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

        self.SPEED = rospy.get_param("speed")
        self.MAX_STEERING_ANGLE = np.deg2rad(rospy.get_param("max_steering_angle"))
        self.STEERING_ANGLE_VELOCITY = np.deg2rad(rospy.get_param("steering_angle_velocity"))
        self.Kp = rospy.get_param("Kp")
        self.Ki = rospy.get_param("Ki")
        self.Kd = rospy.get_param("Kd")

        self.prev_time = rospy.Time.now()

        self.prev_theta_err = 0.0
        self.prev_dist_err = 0.0

        self.running_theta_err = 0.0
        self.running_dist_err = 0.0

    def line_callback(self, msg):
        time = rospy.Time.now()
        dt = (time - self.prev_time).to_sec()
        self.prev_time = time

        x = msg.x # x in front
        y = msg.y # y to left

        dist = np.sqrt((x**2.0)+(y**2.0))
        dist_err = dist

        theta_err = np.arctan2(y, x) # theta to left

        self.running_theta_err += theta_err
        self.running_dist_err += dist_err

        d_theta_dt = (theta_err - self.prev_theta_err) / dt
        d_dist_dt = (dist_err - self.prev_dist_err) / dt

        self.prev_dist_err = dist_err
        self.prev_theta_err = theta_err

        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = "base_link"

        # Find angle
        angle = self.Kp * theta_err + \
            self.Kd * d_theta_dt + self.Ki * self.running_theta_err
        if self.MAX_STEERING_ANGLE != 0:
            # If there is a max steering angle, we limit it
            angle = np.sign(angle)*max(abs(angle), self.MAX_STEERING_ANGLE)
        drive_cmd.drive.steering_angle = angle

        drive_cmd.drive.speed = self.SPEED
        drive_cmd.drive.steering_angle_velocity = self.STEERING_ANGLE_VELOCITY

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
