#!/usr/bin/env python
import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseWithCovariance
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from tf.transformations import euler_from_quaternion


class Lab2:

    def __init__(self):
        """
        Class constructor
        """
        self.px = None
        self.py = None
        self.pth = None

        # PID
        self.Kp = 1.1  # Kp = 1
        self.Ki = 0

        # Initialize node
        rospy.init_node('lab2', anonymous=True)

        ## PUBLISHERS
        # Tell ROS that this node publishes Twist messages on the '/cmd_vel' topic
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Create a publisher to notify frontier when no path has been found 
        # The topic is "/arrival", the message type is PoseWithCovarianceStamped
        # frontier is subscribed to this topic, and will call frontier_exploration()
        self.arrival = rospy.Publisher('/has_arrived', PoseWithCovarianceStamped, queue_size=10)

        ## SUBSCRIBERS
        
        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber('/odom', Odometry, self.update_odometry)
        
        # Tell ROS that this node subscribes to PoseStamped messages on the '/paths' topic
        # When a message is received, call self.follow_path
        rospy.Subscriber('/paths', Path, self.follow_path)
    
        # Initialization complete
        rospy.sleep(1)
        rospy.loginfo("Lab2 ready")

    def follow_path(self, plan):
        """
        Takes a path and instruct the robot to follow it
        :param  plan  [Path]  The path from start to goal  
        """
        rospy.loginfo("Following the path")    
        waypoint = plan.poses       # Get the array of Poses (PoseStamped)
        for i in range(len(waypoint)):
            if i > 0:               # If the current pose is not the start pose
                going_to = waypoint[i]
                if i == len(waypoint) - 1:      # And if the current pose is the last pose
                    self.go_to(going_to, True)  # Instruct robot to go to given pose and turn to target orientation
                else:
                    self.go_to(going_to, False) # Else, instruct the robot to go to the given pose
         
        # Create PoseWithCovarianceStamped message to publish
        last_waypoint = waypoint[0]    # PoseStamped
        print('last waypoint : ', last_waypoint)
        posecovstamped = PoseWithCovarianceStamped()
        posecovstamped.header.frame_id = 'map'
        posecovstamped.pose = PoseWithCovariance()
        #posecovstamped.pose.pose = last_waypoint.pose
        #posecovstamped.pose.covariance = 0
        print(posecovstamped)
        self.arrival.publish(posecovstamped)      # Publish to let frontier node knows, it has arrived at a goal, so it can explore again
        rospy.loginfo("Published msg to /arrived")  
        rospy.loginfo("Arrived at Goal") 




    def send_speed(self, linear_speed, angular_speed):
        """
        Sends the speeds to the motors.
        :param linear_speed  [float] [m/s]   The forward linear speed.
        :param angular_speed [float] [rad/s] The angular speed for rotating around the body center.
        """
        # Create twist message
        msg_cmd_vel = Twist()
        # Linear velocity
        msg_cmd_vel.linear.x = linear_speed
        msg_cmd_vel.linear.y = 0.0
        msg_cmd_vel.linear.z = 0.0
        # Angular velocity
        msg_cmd_vel.angular.x = 0.0
        msg_cmd_vel.angular.y = 0.0
        msg_cmd_vel.angular.z = angular_speed

        ### Publish the message
        self.cmd_vel.publish(msg_cmd_vel)

    def driveToPoint(self, target_x, target_y, linear_speed):
        """
        Drives the robot in a straight line to target pose
        :param target       [PoseStamped] 
        :param linear_speed [float] [m/s] The forward linear speed.
        """
        # Save the initial pose
        init_x = self.px
        init_y = self.py
        init_angle = self.pth

        #target_x = target.pose.position.x
        #target_y = target.pose.position.y
        print("Target x: ", target_x, ", Target y: ", target_y)

        # Send speed to instruct robot to drive forward
        self.send_speed(linear_speed,0)

        # Initialize current position
        current_x = init_x
        current_y = init_y

        TOLERANCE_DIST = 0.075   # Tolerance in position ## 0.02  ## with the glitching it works with 0.1
        TOLERANCE_ANGLE = 0.1   #0.1   # Tolerance in angle for PID controller #0.01 ## with the glitching it works with 0.08

        # Continue driving forward until robot arrives at a position within tolerance
        while (not (abs(target_x - current_x) < TOLERANCE_DIST)) or (not (abs(target_y - current_y) < TOLERANCE_DIST)):
            # Update current position
            current_x = self.px
            current_y = self.py
            print("Current x ", current_x, ", Current y ", current_y)
            
            # Adjust angular speed when encountering angular drift
            if(abs(init_angle - self.pth) > TOLERANCE_ANGLE):
                angSpd =  self.Kp*(init_angle - self.pth)
                # Limit angular speed with the given linear speed
                if angSpd > linear_speed: angSpd = linear_speed
                elif angSpd < -linear_speed: angSpd = -linear_speed
                
                # Send speed to instruct robot to drive forward with given linear speed and angular speed
                self.send_speed(linear_speed, angSpd)
                print("AngSpd = ", angSpd, ", Angle: ", self.pth)
            rospy.sleep(0.025)   # delay # 0.05

        self.send_speed(0,0)    # Stop robot when robot reaches target position
        
    def drive(self, distance, linear_speed):
        """
        Drives the robot in a straight line.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The forward linear speed.
        """
        # Save the initial pose
        init_x = self.px
        init_y = self.py
        init_angle = self.pth
        
        # Calculate required distance in x and y axis to the target 
        # rx is the distance between target and current position on x-axis
        # ry is the distance between target and current position on y-axis
        if (init_angle > 0) and (init_angle > math.pi/2):
            beta = math.pi - init_angle                     # angle with respect of -x axis
            ry = distance*(math.sin(beta))                  
            rx = -distance*(math.cos(beta))              
        elif (init_angle > 0) and (init_angle < math.pi/2):
            ry = distance*(math.sin(init_angle))
            rx = distance*(math.cos(init_angle))
        elif (init_angle < 0) and (init_angle > -math.pi/2):
            beta = math.pi + init_angle                     # angle with respect of -x axis
            ry = -distance*(math.sin(beta))
            rx = -distance*(math.cos(beta))
        elif (init_angle < 0 ) and (init_angle < -math.pi/2):
            ry = distance*(math.sin(init_angle))
            rx = -distance*(math.cos(init_angle))
        
        # Calculate the position of the target in world coordinate
        target_x = rx + init_x
        target_y = ry + init_y
        print("Target x: ", target_x, ", Target y: ", target_y)

        # Send speed to instruct robot to drive forward
        self.send_speed(linear_speed,0)

        # Initialize current position
        current_x = init_x
        current_y = init_y
        #print("Initial x: ", current_x, ", Initial y: ", current_y)

        TOLERANCE_DIST = 0.075    # Tolerance in position ## 0.02  ## with the glitching it works with 0.1
        TOLERANCE_ANGLE = 0.1   # Tolerance in angle for PID controller #0.01 ## with the glitching it works with 0.08

        # Continue driving forward until robot arrives at a position within tolerance
        while (not (abs(target_x - current_x) < TOLERANCE_DIST)) or (not (abs(target_y - current_y) < TOLERANCE_DIST)):
            # Update current position
            current_x = self.px
            current_y = self.py
            print("Current x ", current_x, ", Current y ", current_y)
            
            # Adjust angular speed when encountering angular drift
            if(abs(init_angle - self.pth) > TOLERANCE_ANGLE):
                angSpd =  self.Kp*(init_angle - self.pth)
                # Limit angular speed with the given linear speed
                if angSpd > linear_speed: angSpd = linear_speed
                elif angSpd < -linear_speed: angSpd = -linear_speed
                
                # Send speed to instruct robot to drive forward with given linear speed and angular speed
                self.send_speed(linear_speed, angSpd)
                print("AngSpd = ", angSpd, ", Angle: ", self.pth)
            rospy.sleep(0.025)   # delay # 0.05

        self.send_speed(0,0)    # Stop robot when robot reaches target position
    


    def rotate(self, angle, aspeed):
        """
        Rotates the robot around the body center by the given angle.
        :param angle         [float] [rad]   The distance to cover.
        :param angular_speed [float] [rad/s] The angular speed.
        """
        #print("***Rotating now by ",angle, " rad***")
        init_angle = self.pth
        #print("Initial angle: ", init_angle)

        ## Converting target angle from robot frame to world frame
        # Calculate the total angle needed to rotate
        sum_angle = init_angle + angle
        sum_angle2 = init_angle - angle
        #print("Sum angle: ", sum_angle)
        
        # Convert sum angle in (0,2pi) into target angle in (0,pi,-pi,0)
       
        if sum_angle > math.pi:
            target_angle = -math.pi + (sum_angle % math.pi) 
        elif sum_angle < - math.pi:
            target_angle = math.pi + (sum_angle % -math.pi)
        else:
            target_angle = sum_angle
        print("Target angle: ", target_angle)
        # Initialize current angle
        current_angle = init_angle

        # Send speed to instruct robot to rotate
        if angle > 0:
            self.send_speed(0, aspeed)
        else:
            self.send_speed(0, -aspeed)

        

        TOLERANCE_ANGLE = 0.008         # Tolerance in angle

        # Continue rotating until robot arrives at an angle within tolerance
        while not (abs(target_angle - current_angle) < TOLERANCE_ANGLE):
            # Update current angle
            current_angle = self.pth
            print("Current angle: ", current_angle)
            rospy.sleep(0.08)           # delay
        
        self.send_speed(0,0)            # Stop robot when robot reaches target angle



    def go_to(self, msg, isLast):
        """
        Calls rotate(), drive##(), and rotate() to attain a given pose.
        This method is a callback bound to a Subscriber.
        :param msg    [PoseStamped] The target pose.
        :param isLast [Boolean] If this pose is the last one in a trajectory
        """
        # Get target x, y, and angle from message
        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        print("GO_TO: Moving to ",target_x, " and ", target_y)
        quat_orig = msg.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        target_pth = yaw

        # Save current pose
        init_x = self.px
        init_y = self.py
        theta = self.pth
        
        ### Rotate to face target position
        # Convert target position in world frame to robot frame
        pointRobot = np.matmul([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]],
                                 [[target_x - init_x],[target_y - init_y]])
        yRT = pointRobot[1][0]          # y position in robot frame
        xRT = pointRobot[0][0]          # x position in robot frame
        newTheta = math.atan(yRT/xRT)   # angle to target position with respect to x-axis
        
        # Calculate rotation angle 
        if xRT > 0:
            rot = newTheta
        elif yRT > 0 and xRT < 0:
            rot = math.pi + newTheta
        elif yRT < 0 and xRT < 0:
            rot = -math.pi + newTheta

        # Rotate to face target position
        self.rotate(rot, 0.18)  ## OG 0.2
        rospy.sleep(0.7)                  # delay

        ### Drive straight to target position
        r = math.sqrt(pow(xRT, 2) + pow(yRT, 2))    # total distance 
        #self.drive(r, 0.12)
        self.driveToPoint(target_x, target_y, 0.12) # Testing speed 0.12
        rospy.sleep(0.2)                  # delay
        
        if isLast:  # If the pose is not the last pose in a path
            ### Rotate to target orientation
            current_pth = self.pth               # Save current orientation
            # Calculate rotation needed
            if (target_pth >= 0 and current_pth >= 0) or (target_pth <= 0 and current_pth <= 0):
                orient = target_pth - current_pth
            elif (target_pth < 0 and current_pth > 0):
                orient = -(abs(target_pth) + abs(current_pth))
            elif (target_pth > 0 and current_pth < 0):
                orient = abs(target_pth) + abs(current_pth)

            # Rotate to face target orientation        
            self.rotate(orient, 0.2)
        
        rospy.sleep(0.5)                  # delay


    def update_odometry(self, msg):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        # Update pose of robot using message passed in
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        quat_orig = msg.pose.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        self.pth = yaw

    def arc_to(self, position):
        """
        Drives to a given position in an arc.
        :param msg [PoseStamped] The target pose.
        """
        ### EXTRA CREDIT
        # TODO
        pass # delete this when you implement your code



    def smooth_drive(self, distance, linear_speed):
        """
        Drives the robot in a straight line by changing the actual speed smoothly.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The maximum forward linear speed.
        """
        ### EXTRA CREDIT
        # TODO
        pass # delete this when you implement your code

    def run(self):
        self.__init__
        rospy.spin()
        
        
if __name__ == '__main__':
    Lab2().run()
    
