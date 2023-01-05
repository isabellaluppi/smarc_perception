#!/usr/bin/env python2.7
#from typing import Counter
from tkinter import Y
from sklearn import linear_model
from sklearn.cluster import KMeans
import rospy
from sss_object_detection.consts import ObjectID
#from tf.transformations import euler_from_quaternion
#import tf2_py
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
import tf2_ros
import tf2_geometry_msgs
from smarc_msgs.msg import GotoWaypoint
from nav_msgs.msg import Odometry

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import math
# import statistics
import matplotlib
import collections
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from std_msgs.msg import Float64, Header, Bool, Empty


class sss_detection_listener:
    def __init__(self, robot_name):

        #kristineberg coordinates
        #self.MOB1 = [643815, 6459258]    #top left
        #self.MOB2 = [643806, 6459228]   #bottom left
        #self.MOB3 = [643816, 6459230]   #bottom center
        #self.MOB4 = [643832, 6459225]   #bottom right
        #self.MOB5 = [643830, 6459257]   #top right

        #algae_small_world coordinates
        self.MOB1 = [643800.54 + 4, 6459252.08 -5]    #buoy corner1
        self.MOB2 = [643800.54 + 4, 6459252.08 +5]    #buoy corner2
        self.MOB4 = [643800.54 + 14, 6459252.08 -5]   #bottom corner3
        self.MOB5 = [643800.54 + 14, 6459252.08 +5]   #bottom corner4
        #self.MOB3 = [643800.54 + 9, 643800.54 -5]   #mid buoy

        self.slope_algae_rope = (self.MOB2[1]-self.MOB4[1])/(self.MOB2[0]-self.MOB4[0])

        self.clustering_flag = False

        self.rope_pose_x = []
        self.rope_pose_y = []
        self.detection_x = []
        self.detection_y = []
        self.class0 = []
        self.class1 = []

        self.wp_x = []
        self.wp_y = []
        self.no_detection = 0
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.current_pose = None
        self.odom_sub = rospy.Subscriber('/{}/dr/odom'.format(robot_name), Odometry,self.update_pose)
        self.yaw_sub = rospy.Subscriber('/{}/dr/yaw'.format(robot_name), Float64, self.raw)

        self.detection_topic = '/{}/payload/sidescan/detection_hypothesis'.format(robot_name)
        self.detection_sub = rospy.Subscriber(self.detection_topic, Detection2DArray, self.detection_callback)
        self.waypoint_topic = '/{}/algae_farm/wp'.format(robot_name)#/sam/algae_farm/wp
        self.waypoint_topic_type = GotoWaypoint #ROS topic type
        self.waypoint_pub = rospy.Publisher(self.waypoint_topic, self.waypoint_topic_type,queue_size=5)
        self.counter=1
        self.enable_pub = rospy.Publisher('/sam/algae_farm/enable', Bool, queue_size=1)
        self.enable = Bool()
        self.enable.data = False
        print(self.waypoint_topic)

        #self.fig, self.ax = plt.subplots()


    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
        return trans

    def raw(self,msg):
        self.rawr=float(msg.data)
        #print('rawr = {} .\n'.format(self.rawr))
        
    def transform_pose(self, pose, from_frame, to_frame):
        trans = self.wait_for_transform(from_frame=from_frame,to_frame=to_frame)
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def update_pose(self, msg):
        # might need to transform pose to another frame
        to_frame = 'utm'
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=to_frame)
        self.current_pose = transformed_pose
        #print('Current pose:')
        #print(type(self.current_pose))

    def detection_callback(self, msg):
        # Assume each Detection2DArray only contains one Detection2D message
        # Further assume each Detection2D only contains one ObjectHypothesisWithPose
        for detection in msg.detections:
            object_hypothesis = detection.results[0]
            object_frame_id = msg.header.frame_id[1:]
            # Pose msg
            object_pose = object_hypothesis.pose
            detection_confidence = object_hypothesis.score
            object_id = object_hypothesis.id

            to_frame = 'utm'
            object_pose = self.transform_pose(object_pose, from_frame=object_frame_id, to_frame=to_frame)

            if object_id == ObjectID.ROPE.value:
                #print('1')
                # print('Detected rope at frame {}, pose {}, with confidence {}'.format(
                #     object_frame_id, object_pose, detection_confidence))
                # Do whatever you want to do
                self.rope_pose_x.append(object_pose.pose.position.x)
                self.rope_pose_y.append(object_pose.pose.position.y)
                self.detection_x.append(object_pose.pose.position.x)
                self.detection_y.append(object_pose.pose.position.y)
                self.publish_waypoint_switch()
            if not object_id == ObjectID.ROPE.value:
                self.detection_x.append(None)
                self.detection_y.append(None)
                self.publish_waypoint_switch()
            if object_id == ObjectID.BUOY.value:
                pass
                # print('Detected buoy at frame {}, pose {}, with confidence {}'.format(
                #     object_frame_id, object_pose, detection_confidence))
            if object_id == ObjectID.NADIR.value:
                pass
                #print('Detected nadir at frame {}, pose {}, with confidence {}'.format(
                #    object_frame_id, object_pose, detection_confidence))

    def publish_waypoint_switch(self):
        #if len(self.rope_pose_x) > 20:
        #print('in publish_waypoint_switch')
        if len(self.detection_x) < 30:
            print('here1')
            pass
        else:
            print('here2')
            for i in range(30):
                if self.detection_x[-i] == None:
                    self.no_detection += 1
            if self.no_detection > 25:
                print('here3')
                self.no_detection = 0
                self.enable.data = False
                self.enable_pub.publish(self.enable)
                pass
            else:
                self.no_detection = 0
                self.publish_waypoint()

    def publish_waypoint(self):
        self.counter += 1

        if len(self.rope_pose_x) <=  50 and len(self.rope_pose_x) >= 10:
            if self.clustering_flag == False:
                X_noclustering= np.array(self.rope_pose_x)
                y_noclustering= np.array(self.rope_pose_y)
            else:
                X = [[x, y] for x, y in zip(self.rope_pose_x, self.rope_pose_y)]
        else: 
            if self.clustering_flag == False:
                X_noclustering = np.array(self.rope_pose_x[-50:])
                y_noclustering = np.array(self.rope_pose_y[-50:])
            else:
                X = [[x, y] for x, y in zip(self.rope_pose_x[-50:], self.rope_pose_y[-50:])]

        if self.clustering_flag == True:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            prediction_class = kmeans.predict(X)
            for km in range(len(X)):
                if prediction_class[km] == 0:
                    self.class0.append(X[km])
                if prediction_class[km] == 1:
                    self.class1.append(X[km])

            X0, y0 = zip(*self.class0)
            X1, y1 = zip(*self.class1)
        #convert into array (necessary format for ransac)
            X0 = np.array(X0)
            y0 = np.array(y0)
            X1 = np.array(X1)
            y1 = np.array(y1)

            slope0, c0 = self.ransac_slope(X0, y0)
            slope1, c1 = self.ransac_slope(X1, y1)

            if (self.slope_algae_rope - slope0) < (self.slope_algae_rope - slope1):
                m_slope = slope0
                c_intercept = c0
            else:
                m_slope = slope1
                c_intercept = c1
        else:
            m_slope, c_intercept = self.ransac_slope(X_noclustering, y_noclustering)
        print('SLOPE OF LINE',m_slope)
        print('SLOPE OF KRISTINEBERGS ALGAE ROPE',self.slope_algae_rope)

        #if (self.slope_algae_rope - m_slope) > 1:
            #return
        
        # get pose of SAM
        x_auv=self.current_pose.pose.position.x
        y_auv=self.current_pose.pose.position.y
        auv_yaw=self.rawr*180/math.pi
  
        Waypoint_x, Waypoint_y = self.calculate_waypoint(x_auv, y_auv, auv_yaw, m_slope, c_intercept)
        print('counter',self.counter)
        print('wp',Waypoint_x)

        self.wp_x.append(float(Waypoint_x))
        self.wp_y.append(float(Waypoint_y))
        #msg.pose.pose.position.x = float(Waypoint_x)
        #msg.pose.pose.position.y = float(Waypoint_y)
        
        #msg.pose = self.rope_pose[-1]

        #msg.pose.pose.orientation.x = 0
        #msg.pose.pose.orientation.y = 0


        if self.counter % 10 == 0:
            msg = GotoWaypoint()
            msg.travel_depth = -1
            msg.goal_tolerance = 2
            msg.z_control_mode = GotoWaypoint.Z_CONTROL_DEPTH
            #msg.speed_control_mode = GotoWaypoint.SPEED_CONTROL_RPM
            #msg.travel_rpm = 1000
            msg.speed_control_mode = GotoWaypoint.SPEED_CONTROL_SPEED
            msg.travel_speed = 1.0
            msg.pose.header.frame_id = 'utm'
            msg.pose.header.stamp = rospy.Time(0)

            msg.pose.pose.position.x = np.mean(self.wp_x)
            msg.pose.pose.position.y = np.mean(self.wp_y)
        #msg.pose.pose.position.x = Waypoint_x
        #msg.pose.pose.position.y = Waypoint_y
            self.waypoint_pub.publish(msg)
            print('PUBLISHED WAYPOINT')
            print(msg)
            #self.ax.scatter(msg.pose.pose.position.x,msg.pose.pose.position.y)
            #self.fig.show()
        #if self.counter % 10 == 0:
            self.wp_x = []
            self.wp_y = []

        self.enable.data = True
        self.enable_pub.publish(self.enable)

    def ransac_slope(self, X, y):
        ransac=linear_model.RANSACRegressor()
        Xr=X.reshape(-1,1)
        #print(Xr[0])
        print('Xr',Xr)
        ransac.fit(Xr,y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        yr=ransac.predict(Xr)
        max_x=max(Xr)
        #max_xindex=Xr.index(max(Xr))
        min_x=min(Xr)
        #min_xindex=Xr.index(min(Xr))
        maximum = np.max(Xr)
        minimum = np.min(Xr)
        #print('maximum:{}'.format(maximum))
        max_xindex = np.where(Xr==maximum)[0]
        min_xindex = np.where(Xr==minimum)[0]
        print('max_xindex:{}'.format(max_xindex[0]))
        m_slope=(yr[max_xindex[0]]-yr[min_xindex[0]])/(Xr[max_xindex[0]]-Xr[min_xindex[0]])
        c_intercept=yr[max_xindex[0]]-m_slope*Xr[max_xindex[0]]

        return m_slope, c_intercept

    def calculate_waypoint(self, x_auv, y_auv, auv_yaw, m_slope, c_intercept):
        if m_slope<20 and m_slope>-20:
            x_distanGo=5/(1+m_slope**2)
            if y_auv > m_slope*x_auv + c_intercept:
                c_move=c_intercept+3*(1+m_slope**2)**0.5
            else:
                c_move=c_intercept-3*(1+m_slope**2)**0.5
            # ax+by+c m,n
            # ((b*b*m-a*b*n-a*c)/(a*a+b*b),(a*a*n-a*b*m-b*c)/(a*a+b*b))
            vpoint_x=(x_auv+m_slope*y_auv-m_slope*c_move)/(m_slope*m_slope+1)
            vpoint_y=(m_slope*m_slope*y_auv+m_slope*x_auv+c_move)/(m_slope*m_slope+1)
            if auv_yaw > 180:
                Waypoint_x=(vpoint_x+x_distanGo)*2/3+x_auv*1/3
                Waypoint_y=((vpoint_x+x_distanGo)*m_slope+c_move)*2/3+y_auv*1/3
            else:
                Waypoint_x=(vpoint_x-x_distanGo)*2/3+x_auv*1/3
                Waypoint_y=((vpoint_x-x_distanGo)*m_slope+c_move)*2/3+y_auv*1/3
        else:
            x_togo=0
            y_togo=5
            
            if y_auv > m_slope*x_auv + c_intercept:
                c_move=c_intercept+3*(1+m_slope**2)**0.5
            else:
                c_move=c_intercept-3*(1+m_slope**2)**0.5
            #vpoint=(x_auv+m_slope*y_auv-m_slope*c_move)/(m_slope*m_slope+1)
            vpoint_x=(x_auv+m_slope*y_auv-m_slope*c_move)/(m_slope*m_slope+1)
            vpoint_y=(m_slope*m_slope*y_auv+m_slope*x_auv+c_move)/(m_slope*m_slope+1)
            if auv_yaw < 90 or auv_yaw > 270:
                Waypoint_x=(x_togo)*2/3+vpoint_x
                Waypoint_y=(y_togo)*2/3+vpoint_y
            else:
                Waypoint_x=(x_togo)*2/3+vpoint_x
                Waypoint_y=-(y_togo)*2/3+vpoint_y

        return Waypoint_x, Waypoint_y

def main():
    rospy.init_node('sss_detection_listener', anonymous=True)
    #rospy.Rate(5)  # ROS Rate at 5Hz
    rospy.Rate(2)
    robot_name_param = '~robot_name'
    if rospy.has_param(robot_name_param):
        robot_name = rospy.get_param(robot_name_param)
        print('Getting robot_name = {} from param server'.format(robot_name))
    else:
        robot_name = 'sam'
        print('{} param not found in param server.\n'.format(robot_name_param))
        print('Setting robot_name = {} default value.'.format(robot_name))

    print('entering ssss_detection_listner...')
    
    listner = sss_detection_listener(robot_name)

    plt.ion()
    plt.show()

    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main()