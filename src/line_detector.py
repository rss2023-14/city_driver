#!/usr/bin/env python

import numpy as np
import rospy
import math 

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

class LineDetector():
    """
    """

    def __init__(self):
        # Subscribe to ZED camera RGB frames
        self.line_pub = rospy.Publisher("/line_px", Point, queue_size=10)
        self.debug_pub = rospy.Publisher("/debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.SEARCH_SIDE_THRESHOLD = rospy.get_param("search_side_threshold")
        self.TURN = 0 # -1 for left, 1 for right
        self.COUNTER = 0
        self.STRAIGHT_FRAME_THRESHOLD = 2

    def image_callback(self, image_msg):
        """
        """
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        top_blacked_portion = 0.60
        bottom_blacked_portion = 0.10
        hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        kernel = np.ones((3,3), np.uint8)

        min_orange = np.array([5,60,105]) # HSV
        max_orange = np.array([15,255,255]) # HSV

        height,width, _ = hsv_img.shape
        num_r_top = int(math.ceil(top_blacked_portion*height))
        num_r_bot = int(math.ceil(bottom_blacked_portion*height))
        mask_top = np.ones_like(hsv_img) * 255
        mask_top[:num_r_top,:,:] = 0 
        mask_top[height-num_r_bot:height,:,:] = 0

        # Erode and dilate image
        hsv_img = cv.bitwise_and(hsv_img,mask_top)
        hsv_img = cv.erode(hsv_img,kernel,iterations=1)
        hsv_img = cv.dilate(hsv_img,kernel,iterations=3)
        mask = cv.inRange(hsv_img,min_orange,max_orange)

        # Find largest contour and its bounding box
        im2, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        try:
            line_contour = max(contours, key=cv.contourArea)
        except ValueError:
            return # No point published!!
        x,y,w,h = cv.boundingRect(line_contour)
        boundingbox = ((x,y),(x+w,y+h))

        msg = Point()
        if self.TURN != 0:
            # Was previously committed to a turn, keep turning
            if self.TURN == -1: # Left
                msg.x = x
                msg.y = y
            elif self.TURN == 1: # Right
                msg.x = x+w
                msg.y = y

            if 1.3*h > w:
                # Detecting a straight line!
                self.COUNTER += 1
            if self.COUNTER > self.STRAIGHT_FRAME_THRESHOLD:
                # Commit to a straight turn now
                self.TURN = 0
                self.COUNTER = 0
        else:
            # Was going straight last frame, are we still?
            if w > 1.3*h: # No
                if x < width*self.SEARCH_SIDE_THRESHOLD:
                    self.TURN = -1 # Commit to a left turn!
                    msg.x = x
                    msg.y = y
                else:
                    self.TURN = 1 # Commit to a right turn!
                    msg.x = x+w
                    msg.y = y
            else: # Yes
                x_bot = (2*x+w)/2
                y_bot = y+h
                msg.x = int(x_bot)
                msg.y = int(y_bot)
        self.line_pub.publish(msg)

        # Plot point and bounding box in debug image
        debug_img = cv.rectangle(img, boundingbox[0], boundingbox[1], color=(255,0,0), thickness=2)
        debug_img = cv.circle(debug_img, (int(msg.x), int(msg.y)), 0, color=(0,0,255), thickness=12)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8") # bgr8 for img, 8UC1 for mask
        self.debug_pub.publish(debug_msg)

        # OLD CODE
        """
        try:
            line_contour = max(contours, key=cv2.contourArea)
        except ValueError:
            lineFound = False

        if lineFound:
            x,y,w,h = cv2.boundingRect(line_contour)

            boundingbox = ((x,y),(x+w,y+h))
            x_bot = (2*x+w)/2
            y_bot = y+h

            msg.x = x_bot
            msg.y = y_bot
        else:
            row_index = int(0.9*height)
            row = hsv_img[row_index,:]
            center = np.argmax(row)
            msg.x = center
            msg.y = row_index        
        self.line_pub.publish(msg)
        """

if __name__ == '__main__':
    try:
        rospy.init_node('line_detector', anonymous=True)
        ld = LineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
