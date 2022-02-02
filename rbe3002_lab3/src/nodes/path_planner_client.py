#!/usr/bin/env python2

import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.srv import GetPlan, GetMap
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from std_msgs.msg import Bool



class PathPlannerClient:

    def __init__(self):
        """
        Class constructor
        """
        self.position = None
        self.orientation = None

        # Initialize node
        rospy.init_node("path_planner_client")

        ## PUBLISHERS

        # Tell ROS that this node publishes Path messages on the '/paths' topic
        # lab2 is subscribed to this topic, and will call follow_path()
        self.path_publish = rospy.Publisher('/paths', Path, queue_size=10)

        # Tell ROS that this node publishes Bool messages on the '/need_new_target' topic
        # frontier is subscribed to this topic, and will call construct_new_target()
        self.new_target_publish = rospy.Publisher('/need_new_target', Bool, queue_size=10)

        ## SUBSCRIBERS

        # Tell ROS that this node subscribes to PoseWithCovarianceStamped messages on the '/initialpose' topic
        # When a message is received, call self.update_pose_estimate
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.update_pose_estimate)

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber('/odom', Odometry, self.update_odometry)

        # Tell ROS that this node subscribes to PoseStamped messages on the '/move_base_simple/goal' topic
        # When a message is received, call self.plan_path_client
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.plan_path_client)

        # Tell ROS that this node subscribes to PoseStamped messages on the '/move_to' topic
        # When a message is received, call self.plan_path_client
        rospy.Subscriber('/move_to', PoseStamped, self.plan_path_client)

        # Initialization complete
        rospy.sleep(1)
        print("PathPlannerClient ready")

    def plan_path_client(self, msg):
        """
        Requests path from the path service
        :param msg  [PoseStamped]   The goal of the path 
        """
        rospy.loginfo("Requesting the path")    

        # Create a PostStamped Object to store the start pose
        start = PoseStamped()
        start.header.frame_id = '/initialpose'
        start.pose.position = self.position
        start.pose.orientation = self.orientation

        # Get goal pose from message passed into this function
        goal = msg
        rospy.wait_for_service('plan_path')
        TOLERANCE = 0.0001      # Path tolerance
        print('start ', start)
        print('goal ', goal)

        try:
            plan_path = rospy.ServiceProxy('plan_path', GetPlan)    # Requesting a path
            resp1 = plan_path(start, goal, TOLERANCE)               # Receive a GetPlan message
            path = resp1.plan

            # If Path is empty, indicating that A* cannot find a path
            if len(path.poses) == 0:
                needToPop = Bool()
                needToPop.data = True
                self.new_target_publish.publish(needToPop)          # Trigger construct_target function again, with boolean message = True
            # If path is valid, publish the path to /paths -> lab2
            else:
                self.path_publish.publish(resp1.plan)           
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def update_pose_estimate(self, msg):
        """
        Initialize the localization system used by the navigation stack by setting the pose of the robot in the world. 
        This method is a callback bound to a Subscriber.
        :param msg [PoseWithCovarianceStamped] The current odometry information.
        """
        # Update pose of robot using message passed in
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
            
    def update_odometry(self, msg):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        # Update pose of robot using message passed in
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        
    def run(self):
        self.__init__
        rospy.spin()
        

if __name__ == '__main__':
    PathPlannerClient().run()


