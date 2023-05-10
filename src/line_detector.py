#!/usr/bin/env python

import numpy as np
import rospy
import math 

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


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
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

        hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        height,width, _ = hsv_img.shape

        min_orange = np.array([5,100,160])  # hsv
        max_orange = np.array([25,255,255]) # hsv

        # how much of the image do we want to black out?
        portion_top = 0.6
        bottom_height = int(math.ceil(0.15*height))
        hsv_img[height-bottom_height:height,:,:]=0
            
        #f ilter out designated top portion of image
        num_r = int(math.ceil(portion_top*height))
        mask_top = np.ones_like(hsv_img) * 255
        mask_top[:num_r,:,:] = 0 
        hsv_img = cv2.bitwise_and(hsv_img,mask_top)
        kernel =  np.ones((3,3), np.uint8)

        # erode and dilate image
        hsv_img = cv2.erode(hsv_img,kernel,iterations=1)
        hsv_img = cv2.dilate(hsv_img,kernel,iterations=3)

        mask = cv2.inRange(hsv_img,min_orange,max_orange)  # hsv
        im2, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        lineFound = True
        try:
            line_contour = max(contours, key=cv2.contourArea)
        except ValueError:
            lineFound = False

        msg = Point()
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


if __name__ == '__main__':
    try:
        rospy.init_node('line_detector', anonymous=True)
        ld = LineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
