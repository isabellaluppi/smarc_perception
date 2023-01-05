#!/usr/bin/env python2.7
#from typing import Counter
from sklearn import linear_model
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

        self.TOSAVE_AUVx = []
        self.TOSAVE_AUVy = []
        self.TOSAVE_ROPEx = []
        self.TOSAVE_ROPEy = []
        self.TOSAVE_clusterx = []
        self.TOSAVE_clustery = []
        self.TOSAVE_WPx = []
        self.TOSAVE_WPy = []


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

        # cluster rope detections
        clusters = self.cluster_detections()

        # select cluster closest to AUV
        closest_cluster = self.select_closest_cluster(clusters)

        #if len(closest_cluster[0]) <=  50 and len(closest_cluster[1]) >= 10:
        X,y = closest_cluster
            #print('self.rope_pose: {}'.format(self.rope_pose))
        #else: 
            #X= np.array(self.rope_pose_x[-50:])
            #print('self.rope_pose: {}'.format(self.rope_pose))

        ransac=linear_model.RANSACRegressor()
        Xr=np.array(X).reshape(-1,1)
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
        self.TOSAVE_clusterx = X
        self.TOSAVE_clustery = y
        self.TOSAVE_ROPEx = self.rope_pose_x
        self.TOSAVE_ROPEy = self.rope_pose_y
        self.TOSAVE_WPx.append(msg.pose.pose.position.x)
        self.TOSAVE_WPy.append(msg.pose.pose.position.y)

        print('TOSAVE_AUVx = ' + ', '.join(map(str, self.TOSAVE_AUVx)))
        print('TOSAVE_AUVy = ' + ', '.join(map(str, self.TOSAVE_AUVy)))
        print('TOSAVE_ROPEx = ' + ', '.join(map(str, self.TOSAVE_ROPEx)))
        print('TOSAVE_ROPEy = ' + ', '.join(map(str, self.TOSAVE_ROPEy)))
        print('TOSAVE_CLUSTERx = ' + ', '.join(map(str, self.TOSAVE_clusterx)))
        print('TOSAVE_CLUSTERy = ' + ', '.join(map(str, self.TOSAVE_clustery)))
        print('TOSAVE_WPx = ' + ', '.join(map(str, self.TOSAVE_WPx)))
        print('TOSAVE_WPy = ' + ', '.join(map(str, self.TOSAVE_WPy)))

        self.enable.data = True
        self.enable_pub.publish(self.enable)

    def cluster_detections(self):
        # compute distance between all pairs of detections
        distances = []
        for i in range(len(self.rope_pose_x)):
            for j in range(i+1, len(self.rope_pose_x)):
                dx = self.rope_pose_x[i] - self.rope_pose_x[j]
                dy = self.rope_pose_y[i] - self.rope_pose_y[j]
                distances.append((i, j, np.sqrt(dx**2 + dy**2)))

        # sort distances in ascending order
        distances.sort(key=lambda x: x[2])

        # initialize clusters with each detection as a separate cluster
        clusters = [[ropex,ropey] for ropex,ropey in zip(self.rope_pose_x, self.rope_pose_y)]

        # MAD (Median Absolute Deviation) method for determining the threshold distance for merging clusters
        # median_x = np.median(self.rope_pose_x)
        # median_y = np.median(self.rope_pose_y)
        # mad_x = np.median(np.abs(self.rope_pose_x - median_x))
        # mad_y = np.median(np.abs(self.rope_pose_y - median_y))
        # mad = np.sqrt(mad_x**2 + mad_y**2)

        d_distances = [d[2] for d in distances]
        median_distance = np.median(d_distances)
        mad = np.median([np.abs(d - median_distance) for d in d_distances])

        # merge clusters that are close enough
        merge_distance = np.mean(d_distances) # 2*mad  # threshold distance for merging clusters
        for i, j, d in reversed(distances):
            print('d')
            print(d)
            print('merge_distance')
            print(merge_distance)
            if d > merge_distance:
                break
            clusters.append(clusters[i] + clusters[j])
            clusters.pop(i)
            clusters.pop(j-1)
        print('CLUSTERS')
        print(np.shape(clusters))
        return clusters


    def select_closest_cluster(self, clusters):
        # select cluster with centroid closest to AUV
        min_distance = float('inf')
        closest_cluster = clusters[0] # initialize with first cluster
        for cluster in clusters:
            if len(cluster) <= 4:  # skip clusters with only a single element, deletes outliers
                continue
            cluster_x, cluster_y = cluster
            dx = self.current_pose.pose.position.x - np.mean(cluster_x)
            dy = self.current_pose.pose.position.y - np.mean(cluster_y)
            distance = np.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster
        return closest_cluster


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