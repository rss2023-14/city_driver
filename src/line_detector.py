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

    def image_callback(self, image_msg):
        """
        """
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        debug_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.debug_pub.publish(debug_msg)
        top_blacked_portion = .5
        bottom_blacked_portion = .1
        hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        kernel = np.ones((3,3), np.uint8)
        min_orange = np.array([5,80,120])  #hsv
        max_orange = np.array([50,255,255]) #hsv
        height,width, _ = hsv_img.shape
        num_r_top = int(math.ceil(top_blacked_portion*height))
        num_r_bot = int(math.ceil(bottom_blacked_portion*height))
        mask_top = np.ones_like(hsv_img) * 255
        mask_top[:num_r_top,:,:] = 0 
        mask_top[height-num_r_bot:height,:,:] = 0


        hsv_img = cv.bitwise_and(hsv_img,mask_top)
        # erode and dilate image
        hsv_img = cv.erode(hsv_img,kernel,iterations=1)
        hsv_img = cv.dilate(hsv_img,kernel,iterations=3)
        mask = cv.inRange(hsv_img,min_orange,max_orange)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        try:
            cone_contour = max(contours, key=cv.contourArea)
        except ValueError:
            return # No point published!!
        x,y,w,h = cv.boundingRect(cone_contour)
        boundingbox = ((x,y),(x+w,y+h))
        # try:
        #     line_contour = max(contours, key=cv2.contourArea)
        # except ValueError:
        #     lineFound = False

        msg = Point()
        # if lineFound:
        #     x,y,w,h = cv2.boundingRect(line_contour)

        #     boundingbox = ((x,y),(x+w,y+h))
        #     x_bot = (2*x+w)/2
        #     y_bot = y+h

        #     msg.x = x_bot
        #     msg.y = y_bot
        # else:
        #     row_index = int(0.9*height)
        #     row = hsv_img[row_index,:]
        #     center = np.argmax(row)
        #     msg.x = center
        #     msg.y = row_index        
        # self.line_pub.publish(msg)
        search_side_thresh = 0.35

        if w > h:
            if x < width*search_side_thresh:
                msg.x = x+0.5*w
                msg.y = y
                self.line_pub.publish(msg)
            else:
                msg.x = x+0.5*w
                msg.y = y
                self.line_pub.publish(msg)

        else:
            x_bot = (2*x+w)/2
            y_bot = y+h
            msg.x = int(x_bot)
            msg.y = int(y_bot)
            self.line_pub.publish(msg)

if __name__ == '__main__':
    try:
        rospy.init_node('line_detector', anonymous=True)
        ld = LineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
