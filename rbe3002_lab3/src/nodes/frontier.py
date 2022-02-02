#!/usr/bin/env python
import math
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from priority_queue import PriorityQueue
from path_planner import PathPlanner
from lab2 import Lab2
from nav_msgs.srv import GetPlan, GetMap
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Bool

#frontier_list = {}
#centroidList = []
#weightList = []
THRESHOLD = 50

class frontier:

    def __init__(self):
        """
        Class constructor
        """
        self.px = None
        self.py = None
        self.pth = None
        self.position = None    # Point object
        self.orientation = None # Quaterion object
        self.initial_pose = None # Pose object storing initialpose

        self.frontier_list = {}
        self.centroidList = []
        self.weightList = []

        self.classmap = None    # Map holder

        ## Initialize node
        rospy.init_node("frontier")

        ## PUBLISHERS

        # Tell ROS that this node publishes Path messages on the '/move_to' topic
        # path_planner_client is subscribed to this topic, and will call plan_path_client()
        self.frontier_publish = rospy.Publisher('/move_to', PoseStamped, queue_size=10)

        # Tell ROS that this node publishes Path messages on the '/paths' topic
        # lab2 is subscribed to this topic, and will call follow_path()
        self.path_publish = rospy.Publisher('/paths', Path, queue_size=10)

        # Create publishers for frontier visualization
        # The topic is "/visual_gridcells", the message type is GridCells
        self.disp_frontier = rospy.Publisher('/visual_gridcells', GridCells, queue_size=10)

        # Create publishers for frontier visualization
        # The topic is "/visual_centroid", the message type is GridCells
        self.disp_centroid = rospy.Publisher('/visual_centroid', GridCells, queue_size=10)

        ## SUBSCRIBERS       

        # Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        # When a message is received, call self.update_odometry
        rospy.Subscriber('/odom', Odometry, self.update_odometry)

        # Tell ROS that this node subscribes to PoseWithCovarianceStamped messages on the '/initialpose' topic
        # When a message is received, call self.frontier_exploration
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.call_frontier_exploration)

        # Tell ROS that this node subscribes to PoseWithCovarianceStamped messages on the '/has_arrived' topic
        # When a message is received, call self.frontier_exploration
        rospy.Subscriber('/has_arrived', PoseWithCovarianceStamped, self.frontier_exploration)

        # Tell ROS that this node subscribes to Bool messages on the '/need_new_target' topic
        # When a message is received, call self.construct_new_target
        rospy.Subscriber('/need_new_target', Bool, self.construct_new_target)

        # Initialization complete
        rospy.sleep(1.0)
        rospy.loginfo("Frontier ready")

    def call_frontier_exploration(self, msg):
        rospy.loginfo("call_frontier_exploration callback") 
        self.initial_pose = msg.pose.pose
        self.frontier_exploration(msg)

    def cspacemap_client(self):
        """
        Requests map that have been padded from the /cspace_map service
        :return  [OccupancyGrid]  Return the map that has been padded 
        """
        rospy.loginfo("1. Requesting the cspaced map")   

        # Get cspaced map 
        rospy.wait_for_service('cspace_map')
        try:
            request_cspacemap = rospy.ServiceProxy('cspace_map', GetMap)    # Requesting a path
            resp1 = request_cspacemap()                                     # Receive a Getmap message
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e) 
        return resp1.map


    def update_odometry(self, msg):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        # Update pose of robot using message passed in
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        quat_orig = msg.pose.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        self.pth = yaw

    def calc_weight_centroid(self, mapdata):
        """
        Calculate the weight and the centroid of the frontier regions
        Call construct_target to find the best frontier
        """
        frontier_gridcells = []

        self.classmap = mapdata
        print('Frontier list = ')
        print(self.frontier_list)

        # Calculate the centroid and its weight with our heuristic function
        for key in self.frontier_list:
            cell_list = self.frontier_list[key]                      # List of frontiers at the given key region
            frontier_gridcells.extend(cell_list)                # Append list of frontiers to main list for gridcells publisher

            currCentroid = frontier.calc_centroid(cell_list)    # Calculate centroid of the list of frontier
            weight = self.heuristic(currCentroid, cell_list)    # Calculate weight of the list of frontier

            self.centroidList.append(currCentroid)                   # Append centroid to centroid list
            self.weightList.append(weight)                           # Append weight to weight list

        print('centroid list = ', self.centroidList)
        print('weight list = ', self.weightList)

        # Display frontiers in Rviz
        self.frontier_display(mapdata, frontier_gridcells)

        # Call next function to move to best centroid with the assumption that there is no need to pop best centroid
        needtopop = Bool()
        needtopop.data = False
        self.construct_new_target(needtopop)
        

    def construct_new_target(self, msg):
        """
        Find the best centroid to go, and construct the target to publish to /move_to
        :param  msg  [Bool]
        """
        needToPop = msg.data
        mapdata = self.classmap

        # Find the best centroid to go
        min_weight = max(self.weightList)                            # Find the minimum weight
        target_index = self.weightList.index(min_weight)                # Find the frontier region with the minimum weight
        target_centroid = self.centroidList[target_index]               # Get target centroid

        if needToPop:
        # If needToPop is True
            # Pop the invalid centroid
            self.centroidList.pop(target_index)
            self.weightList.pop(target_index)
            print(target_centroid, ' is unwalkable')
            print('after pop, centroid list = ', self.centroidList)
            print('after pop, weight list = ', self.weightList)



            if len(self.centroidList) == 0:  # If there is no more frontier available
                rospy.loginfo('No more available frontier to go to, moving to initial pose')
                self.move_to_initial()
                return
            # Find next best centroid to go
            min_weight = max(self.weightList)                            # Find the minimum weight
            min_index = self.weightList.index(min_weight)                # Find the frontier region with the minimum weight
            target_centroid = self.centroidList[min_index]               # Get target centroid

            print('after pop, target centroid is ', target_centroid)

        # If centroid is unwalkable, find the closest free cell that it can go
        is_centroid_walkable = PathPlanner.is_cell_walkable(mapdata, target_centroid[0], target_centroid[1])
        if not is_centroid_walkable:
            target_centroid = self.closest_open_cell(mapdata, target_centroid[0], target_centroid[1])

        # Convert centroid from grid to world (Point)
        target_centroid_world = PathPlanner.grid_to_world(mapdata, target_centroid[0], target_centroid[1]) 

        ## Display centroid on Rviz
        point_centroid = Point()
        point_centroid.x = target_centroid_world.x
        point_centroid.y = target_centroid_world.y

        visual_centroid = GridCells()
        visual_centroid.header.frame_id = 'map'
        visual_centroid.cell_width = mapdata.info.resolution
        visual_centroid.cell_height = mapdata.info.resolution
        visual_centroid.cells = [point_centroid]

        self.disp_centroid.publish(visual_centroid)     # Publish

        ## Publish goal to instruct robot to move
        posestamp = PoseStamped()                       # Create PoseStamped message for target frontier
        posestamp.header.frame_id = 'map'
        posestamp.pose.position.x = target_centroid_world.x
        posestamp.pose.position.y = target_centroid_world.y
        posestamp.pose.position.z = 0
       
        self.frontier_publish.publish(posestamp)        # Publish to /move_to -> path_planner_client


    def frontier_display(self, mapdata, all_frontier):
        rospy.loginfo('Displaying frontier')
        # Create GridCells message for frontier visualization
        point_arr = []
        for frontier in all_frontier:
            world_coord_frontier = PathPlanner.grid_to_world(mapdata, frontier[0], frontier[1])
            point = Point()
            point.x = world_coord_frontier.x
            point.y = world_coord_frontier.y
            point_arr.append(point)

        visual_frontiers = GridCells()
        visual_frontiers.header.frame_id = 'map'
        visual_frontiers.cell_width = mapdata.info.resolution
        visual_frontiers.cell_height = mapdata.info.resolution
        visual_frontiers.cells = point_arr

        # Publish frontier GridCells message to topic and display in Rviz
        self.disp_frontier.publish(visual_frontiers)    


    def heuristic(self, centroid, frontiers):
        """
        Calculate the weight given a centroid according to the heuristic function
        ;param  centroid  [(int, int)]  The centroid of a frontier region
        :return           [int]         The weight of the frontier region
        """
        length = len(frontiers)
        distance = PathPlanner.euclidean_distance(self.px, self.py, centroid[0], centroid[1])
        return length/distance


    def floodfill(self, mapdata, x, y, regionID):
        """
        Perform floodfill algorithm to find all the connecting frontiers with the given frontier
        ;param  mapdata  [OccupancyGrid]  The map data
        :param     x     [int]            The x coordinate of the frontier
        :param     y     [int]            The y coordinate of the frontier
        :param  regionID [int]            The region ID of this frontier region       
        """
        grid = mapdata.data             
        width = mapdata.info.width
        length = mapdata.info.height

        index = PathPlanner.grid_to_index(mapdata, x, y)

        if x < 0 or x >= width or y < 0 or y >= length: return                  # Base cases: if cell is out-of-bound
        if self.frontier_exists(x, y): return                               #             if cell is in frontier list already
        if not frontier.is_cell_frontier(mapdata, x, y): return                    #             if cell is not a frontier

        self.frontier_list[regionID].append((x,y))                                   # append to the frontier list according to the region
        all_frontier_neighbors = frontier.neighbors_of_8(mapdata, x, y) # find 8-neighbor of this cell that are frontier

        for neighbor in all_frontier_neighbors:
            self.floodfill(mapdata, neighbor[0], neighbor[1], regionID) # find all the connecting frontiers   

    def frontier_exists(self, x, y):
        """
        Check if the given frontier already exists on the frontier list
        :param  x  [int]        The x coordinate of the frontier
        :param  y  [int]        The y coordinate of the frontier
        :return    [boolean]    True if the frontier exists on the list already
        """
        if any([True for key,value in self.frontier_list.items() if (x,y) in value]): return True
        else: return False

    @staticmethod
    def calc_centroid(listOfCells):
        """
        Calculate the centroid of the given list of frontiers
        :param  listOfCells  [[(int,int)]]     The list of frontiers
        :return              [(int, int)]      The centroid of the list of frontiers
        """
        cx = 0
        cy = 0 
        n = len(listOfCells)        # Number of frontieres

        for cell in listOfCells:
            cx = cx + cell[0]       # Accumulate values of x coordinate
            cy = cy + cell[1]       # Accumulate values of y coordinate

        cx = cx/n
        cy = cy/n
        return (cx, cy)
    
    @staticmethod
    def is_cell_frontier(mapdata, x, y):
        """
        A cell is frontier if all of these conditions are true:
        1. It is within the boundaries of the grid;
        2. It is free (not unknown, not occupied by an obstacle)
        3. It is next to an unknown cell
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The X coordinate in the grid.
        :param y       [int]           The Y coordinate in the grid.
        :return        [boolean]       True if the cell is walkable, False otherwise
        """
        # Get map information: width, height, and the occupancy grid array
        width = mapdata.info.width
        height = mapdata.info.height
        occupancy = mapdata.data
        num_data = len(occupancy)
        
        index = PathPlanner.grid_to_index(mapdata, x, y)  # Convert the coordinate in occupancy grid to index
        # Determine if cell is walkable
        if (x < 0 or x > width) or (y < 0 or y > height): # 1: Cell is out-of-bound
            return False
        elif index > (num_data - 1) or index < 0:         # 2: Cell index is out-of-range
            return False
        elif (occupancy[index] == -1):                    # 3: Cell has not been explored yet: unknown 
            return False
        elif (occupancy[index] >= THRESHOLD):             # 4: Cell is occupied/obstacle0
            return False
        elif frontier.has_unknown_neighbors(mapdata, x, y) == False: # 5: It does not have unknown neighbor
            return False
        return True

    @staticmethod
    def has_unknown_neighbors(mapdata, x, y):
        """
        Returns true if the cell has an unknown neighbor
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The X coordinate in the grid.
        :param y       [int]           The Y coordinate in the grid.
        :return        [boolean]       True if the cell has an unknown neighbor
        """
        width = mapdata.info.width                      # Get map width
        index = PathPlanner.grid_to_index(mapdata,x,y)  # Convert occupancy grid coordinate to index
        arr = mapdata.data
        
        # Find the index of the four neighbors 
        top = index - width 
        left = index - 1
        right = index + 1
        low = index + width
        index_4_neighbors = [top, left, right, low]

        # Return true if it has an unknown neighbor
        for i in index_4_neighbors:
            if arr[i] == -1:    
                return True
        return False 
    
    
    @staticmethod
    def neighbors_of_8(mapdata, x, y):
        """
        Returns the frontier 8-neighbors cells of (x,y) in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The X coordinate in the grid.
        :param y       [int]           The Y coordinate in the grid.
        :return        [[(int,int)]]   A list of frontier 8-neighbors.
        """
        width = mapdata.info.width                      # Get map width
        neighbors = []
        
        index = PathPlanner.grid_to_index(mapdata,x,y)  # Convert occupancy grid coordinate to index

        # Find the index of the eight neighbors 
        top_left = index - width - 1
        top = index - width
        top_right = index - width + 1
        left = index - 1
        right = index + 1
        low_left = index + width - 1
        low = index + width
        low_right = index + width + 1
        index_8_neighbors = [top_left, top, top_right, left, right, low_left, low, low_right]

        # Append the frontier neighbors to a list and return
        for i in index_8_neighbors:
            coordinate = PathPlanner.index_to_grid(mapdata,i)
            if frontier.is_cell_frontier(mapdata, coordinate[0], coordinate[1]):
                neighbors.append(coordinate)
        return neighbors 

    def next_ungroup_frontier(self, mapdata):
        """
        Return the next frontier that is not in a group
        :param mapdata [OccupancyGrid] The map information.
        :return        [(int, int)]    The next uncheck frontier
        """
        rospy.loginfo('2b: Looping to find ungroup frontier')
        occupancy = mapdata.data

        for i in range(len(occupancy)):                                 # For every cell in map
            coordinate = PathPlanner.index_to_grid(mapdata, i)          # Get coordinate of the cell
            x = coordinate[0]
            y = coordinate[1]
            if frontier.is_cell_frontier(mapdata, x, y):                # If cell is a frontier
                if not self.frontier_exists(x, y):                  # and if it has not been added in the total frontier list
                    rospy.loginfo('2c: Found next uncheck frontier: (%d, %d)', x, y)
                    return (x,y)

        rospy.loginfo('2c: Found 0 uncheck frontier')

    def identify_frontier_region(self, mapdata):
        """
        Return a dictionary of all frontier organized by region (key, value):(regionID, frontiers in the region) 
        :param mapdata [OccupancyGrid] The map information.
        :return        [Dictionary]    The frontiers dictionary
        """
        rospy.loginfo('2a: Identifying all frontier regions')
        occupancy = mapdata.data        # Get OccupancyGrid from mapdata
        checked_all = False
        region_count = 1                # Initialize region ID
        
        while not checked_all:
            next = self.next_ungroup_frontier(mapdata)                 # Find the next unchecked frontier
            if next == None: return self.frontier_list                      # If there is no more unchecked frontier, return

            self.frontier_list[region_count] = []                           # Initialize dictionary key for the specific region
            self.floodfill(mapdata, next[0], next[1], region_count)    # Use floodfill to find all the connecting frontiers
            region_count += 1                                          # Move on to the next region
        
        return self.frontier_list

    def frontier_exploration(self, msg):
        """
        Explore frontier once this function has been called.
        This will be called in 2 situations:
            1. When intial pose has been selected in Rviz
            2. When robot arrive at a goal
        :param  msg  [PoseWithCovarianceStamped]   Trigger message
        """
        rospy.loginfo("Exploring frontiers")   
        mapdata = self.cspacemap_client()                       # 1. Request map from cspacemap service
        
        # 2. Check if robot is currently in padding
        worldPos = PathPlanner.world_to_grid(mapdata, self.position)                 # Convert current world to grid
        worldIndex = PathPlanner.grid_to_index(mapdata, worldPos[0], worldPos[1])       # Convert current grid to index
        print("WorldIndex:", worldIndex)
        print("mapdata", mapdata.data[worldIndex])
        if(not mapdata.data[worldIndex] == 0):                  # If current cell is not open: Is robot in padding?
            self.escape_from_cspace(mapdata)                    # Escape to the closest free cell
            return                                              # Terminate frontier exploration until arrived at open cell

        # 3. If robot is not stuck in padding
        self.frontier_list = {}                                  # Clear frontiers list from last iteration
        self.centroidList = []
        self.weightList = []

        # 4. Identify frontier regions
        all_frontiers = self.identify_frontier_region(mapdata)                 
        if len(all_frontiers) == 0: 
            #No more frontier, stop frontier exploration and go back to intial pose
            self.move_to_initial()
            return

        # 5. Find and go to the closet centroid
        self.calc_weight_centroid(mapdata)                      
            
    def escape_from_cspace(self, mapdata):
        """
        This function is called when it is determined that robot is stuck in the cspace padding.
        Robot will be instructed to move to the closest open cell.
        :param mapdata [OccupancyGrid] The map information.
        """
        print("IN C-SPACE")
        # Take current position in world and convert to grid (Point -> Tuple)
        current_grid = PathPlanner.world_to_grid(mapdata, self.position)
        # Get closest open cell
        closest_open = frontier.closest_open_cell(mapdata, current_grid[0], current_grid[1])

        neighbors = PathPlanner.neighbors_of_8(mapdata, closest_open[0], closest_open[1])
        dist = 0
        cell = None

        for n in neighbors:
            currDist = PathPlanner.euclidean_distance(n[0], n[1], current_grid[0], current_grid[1])
            if currDist > dist:
                cell = n
        closest_open = cell

        neighbors = PathPlanner.neighbors_of_8(mapdata, closest_open[0], closest_open[1])

        dist = 0
        cell = None

        for n in neighbors:
            currDist = PathPlanner.euclidean_distance(n[0], n[1], current_grid[0], current_grid[1])
            if currDist > dist:
                cell = n
        closest_open = cell

        # Convert grid to world (Tuple > Point)
        closest_world = PathPlanner.grid_to_world(mapdata, closest_open[0], closest_open[1])
        # Construct Path message and publish to /paths
        # Two waypoints: start and goal
        start = PoseStamped()
        start.header.frame_id = 'map'
        start.pose.position = self.position
        start.pose.orientation = self.orientation

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = closest_world

        path = Path()
        path.header.frame_id = 'map'
        path.poses = [start, goal]

        print("PLANNED ESCAPE")

        self.path_publish.publish(path)



    @staticmethod
    def closest_open_cell(mapdata, x, y):
        """
        Find the closest open cell from the current position
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The X coordinate in the grid.
        :param y       [int]           The Y coordinate in the grid.
        :return        [(int,int)]     A walkable neighbor.
        """
        found = False
        i = 1
        while not found:
            star_neighbors = [(x,y-i), (x+i,y), (x,y+i), (x-i,y), (x+i,y-i), (x+i,y+i), (x-i,y+i), (x-i,y-i)]
            for cell in star_neighbors:
                if PathPlanner.is_cell_walkable(mapdata, cell[0], cell[1]): return cell
            i += 1


    def any_unknown_cell(self, mapdata):
        """
        Check if there is unknown (-1) cell in the OccupancyGrid
        :param  mapdata  [OccupancyGrid]  
        :return          [boolean]       True if there is unknown cell
        """
        for cell in mapdata.data:
            if cell == -1:
                #rospy.loginfo('There is unknown cell')
                return True
        #rospy.loginfo('There is no unknown cell')
        return False

    def move_to_initial(self):
        """
        Publish initial position to /move_to and instruct robot to move back to initial position,
        set by Rviz 2D Pose Estimate
        """
        rospy.loginfo('Moving to intial pose')
        init_posestamped = PoseStamped()
        init_posestamped.header.frame_id = 'map'
        init_posestamped.pose = self.initial_pose
        self.frontier_publish.publish(init_posestamped)

    def run(self):
        self.__init__   
        rospy.spin()                    

if __name__ == '__main__':
    frontier().run()

