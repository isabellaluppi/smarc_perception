#!/usr/bin/env python2.7
#from typing import Counter
from sklearn import linear_model
from sklearn.cluster import DBSCAN
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

        self.rope_pose_x = []
        self.rope_pose_y = []
        self.detection_x = []
        self.detection_y = []


        self.list_x = []
        self.list_y =[]
        self.list_wp_x = []
        self.list_wp_y =[]

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

        self.TOSAVE_AUVx = [None]
        self.TOSAVE_AUVy = [None]
        self.TOSAVE_ROPEx = [None]
        self.TOSAVE_ROPEy = [None]
        self.TOSAVE_WPx = [None]
        self.TOSAVE_WPy = [None]


    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
                print('from_frame: {}, to_frame: {}, current time: {}'.format(from_frame, to_frame, rospy.Time.now()))
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
                # print('Detected rope at frame {}, pose {}, with confidence {}'.format(
                #      object_frame_id, object_pose, detection_confidence))
                self.rope_pose_x.append(object_pose.pose.position.x)
                self.rope_pose_y.append(object_pose.pose.position.y)
                self.detection_x.append(object_pose.pose.position.x)
                self.detection_y.append(object_pose.pose.position.y)
                self.publish_waypoint_switch()
            if not object_id == ObjectID.ROPE.value:
                self.detection_x.append(None)
                self.detection_y.append(None)
                #self.publish_waypoint_switch()
            if object_id == ObjectID.BUOY.value:
                pass
                # print('Detected buoy at frame {}, pose {}, with confidence {}'.format(
                #     object_frame_id, object_pose, detection_confidence))
            if object_id == ObjectID.NADIR.value:
                pass
                #print('Detected nadir at frame {}, pose {}, with confidence {}'.format(
                #    object_frame_id, object_pose, detection_confidence))

    def publish_waypoint_switch(self):
        if len(self.detection_x) < 50:
            rospy.loginfo('Not enough detections')
            pass
        else:
            for i in range(50):
                if self.detection_x[-i] == None:
                    self.no_detection += 1
            if self.no_detection > 40:
                rospy.loginfo('Not enough rope detections in the last 50 detections')
                self.no_detection = 0
                pass
            else:
                rospy.loginfo('Sufficient rope detections!')
                self.no_detection = 0
                self.publish_waypoint()

    def publish_waypoint(self):
        rospy.loginfo('Calculating WP...')
        self.counter += 1

        if len(self.rope_pose_x) <=  50 and len(self.rope_pose_x) >= 10:
            X= np.array(self.rope_pose_x)
            y= np.array(self.rope_pose_y)
            #print('self.rope_pose: {}'.format(self.rope_pose))
        else: 
            X= np.array(self.rope_pose_x[-50:])
            y= np.array(self.rope_pose_y[-50:])
            #print('self.rope_pose: {}'.format(self.rope_pose))

        rope = self.select_rope_to_follow(self, [self.rope_pose_x, self.rope_pose_x])

        X,y = rope

        ransac=linear_model.RANSACRegressor()
        Xr=X.reshape(-1,1)
        ransac.fit(Xr,y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        yr=ransac.predict(Xr)
        max_x=max(Xr)
        min_x=min(Xr)
        maximum = np.max(Xr)
        minimum = np.min(Xr)
        max_xindex = np.where(Xr==maximum)[0]
        min_xindex = np.where(Xr==minimum)[0]
        m_slope=(yr[max_xindex[0]]-yr[min_xindex[0]])/(Xr[max_xindex[0]]-Xr[min_xindex[0]])
        c_intercept=yr[max_xindex[0]]-m_slope*Xr[max_xindex[0]]
        
        # get pose of SAM
        x_auv=self.current_pose.pose.position.x
        y_auv=self.current_pose.pose.position.y
        auv_yaw=self.rawr*180/math.pi
                
        # caulate waypoint      
        if m_slope<20 and m_slope>-20:
            x_distanGo=5/(1+m_slope**2)
            if y_auv > m_slope*x_auv + c_intercept:
                c_move=c_intercept+3*(1+m_slope**2)**0.5
            else:
                c_move=c_intercept-3*(1+m_slope**2)**0.5
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
            vpoint_x=(x_auv+m_slope*y_auv-m_slope*c_move)/(m_slope*m_slope+1)
            vpoint_y=(m_slope*m_slope*y_auv+m_slope*x_auv+c_move)/(m_slope*m_slope+1)
            if auv_yaw < 90 or auv_yaw > 270:
                Waypoint_x=(x_togo)*2/3+vpoint_x
                Waypoint_y=(y_togo)*2/3+vpoint_y
            else:
                Waypoint_x=(x_togo)*2/3+vpoint_x
                Waypoint_y=-(y_togo)*2/3+vpoint_y
        print('wp',Waypoint_x)

        self.wp_x.append(float(Waypoint_x))
        self.wp_y.append(float(Waypoint_y))


        #if self.counter % 10 == 0:
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

        # msg.pose.pose.position.x = np.mean(self.wp_x)
        # msg.pose.pose.position.y = np.mean(self.wp_y)

        self.list_wp_x.append(msg.pose.pose.position.x)
        self.list_wp_y.append(msg.pose.pose.position.y)
        self.list_x.append(x_auv)
        self.list_y.append(y_auv)

        msg.pose.pose.position.x = Waypoint_x
        msg.pose.pose.position.y = Waypoint_y

        self.waypoint_pub.publish(msg)
        rospy.loginfo('PUBLISHED WAYPOINT')
        print(msg)
        self.wp_x = []
        self.wp_y = []

        self.TOSAVE_AUVx.append(self.current_pose.pose.position.x)
        self.TOSAVE_AUVy.append(self.current_pose.pose.position.y)
        self.TOSAVE_ROPEx = [self.rope_pose_x]
        self.TOSAVE_ROPEy = [self.rope_pose_y]
        self.TOSAVE_WPx.append(msg.pose.pose.position.x)
        self.TOSAVE_WPy.append(msg.pose.pose.position.y)

        print('TOSAVE_AUVx = ' + ', '.join(map(str, self.TOSAVE_AUVx)))
        print('TOSAVE_AUVy = ' + ', '.join(map(str, self.TOSAVE_AUVy)))
        print('TOSAVE_ROPEx = ' + ', '.join(map(str, self.TOSAVE_ROPEx)))
        print('TOSAVE_ROPEy = ' + ', '.join(map(str, self.TOSAVE_ROPEy)))
        print('TOSAVE_WPx = ' + ', '.join(map(str, self.TOSAVE_WPx)))
        print('TOSAVE_WPy = ' + ', '.join(map(str, self.TOSAVE_WPy)))

        self.enable.data = True
        self.enable_pub.publish(self.enable)

        

    def select_rope_to_follow(self, detections):
        # Convert detections to numpy array
        X = np.array(detections)

        # Use DBSCAN to cluster detections into ropes
        dbscan = DBSCAN(eps=2, min_samples=5)
        clusters = dbscan.fit_predict(X)

        # Initialize dictionary to store predicted locations of ropes
        self.rope_predictions = {}

        # Predict future locations of all ropes
        for i in range(len(X)):
            if clusters[i] == -1:
                # Skip noise points
                continue
            if clusters[i] not in self.rope_predictions:
                self.rope_predictions[clusters[i]] = self.predict_rope_location(X[i])
            else:
                # Update existing prediction with new data
                self.rope_predictions[clusters[i]] = self.predict_rope_location(X[i], self.rope_predictions[clusters[i]])

        # Calculate distances from AUV to predicted locations of ropes
        distances = {}
        for rope_id in self.rope_predictions:
            x_diff = self.rope_predictions[rope_id][0] - self.current_pose.pose.position.x
            y_diff = self.rope_predictions[rope_id][1] - self.current_pose.pose.position.y
            distances[rope_id] = math.sqrt(x_diff**2 + y_diff**2)

        # Select rope with shortest distance
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        selected_rope_id = sorted_distances[0][0]
        selected_rope_prediction = self.rope_predictions[selected_rope_id]

        # Return predicted location of selected rope
        return selected_rope_prediction



    
    def predict_rope_location(self, rope_detections, rope_id, dt):
        # Initialize state and covariance estimates
        x = np.array([rope_detections[0][0], rope_detections[0][1], 0, 0])
        P = np.eye(4)
        
        # Set state transition matrix
        F = np.eye(4) * 0.1
        # Set process noise covariance matrix
        Q = np.eye(4) * 0.1
        # Set measurement matrix
        H = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])
        # Set measurement noise covariance matrix
        R = np.eye(2) * 0.1
        
        # Initialize list to store predicted rope locations
        predicted_rope_locations = []
        
        # Iterate over each detection and perform Kalman filter prediction
        for detection in rope_detections:
            x, P = self.kalman_filter(x, P, F, Q, H, R)
            predicted_rope_locations.append((rope_id, x[0], x[1]))
        
        return predicted_rope_locations


    def kalman_filter(self, x, P, F, Q, H, R):
        # Prediction
        x = np.dot(F, x)
        P = np.dot(F, P).dot(F.T) + Q
        
        # Rope measurement
        z = [self.rope_pose_x, self.rope_pose_y]

        # Measurement update
        y = z - np.dot(H, x)
        S = np.dot(H, P).dot(H.T) + R
        K = np.dot(P, H.T).dot(np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = P - np.dot(K, H).dot(P)
        
        return x, P
       




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