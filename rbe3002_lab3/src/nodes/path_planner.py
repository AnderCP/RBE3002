#!/usr/bin/env python

import math
import rospy
import numpy as np
from priority_queue import PriorityQueue
from nav_msgs.srv import GetPlan, GetMap
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

THRESHOLD = 50
# unknown = -1, open = 0, occupied = 100

class PathPlanner:
    
    def __init__(self):
        """
        Class constructor
        """
        self.heading = None

        ## Initialize the node and call it "path_planner"
        rospy.init_node("path_planner")

        ## SERVICES

        ## Create a new service called "plan_path" that accepts messages of
        ## type GetPlan and calls self.plan_path() when a message is received
        rospy.Service('plan_path', GetPlan, self.plan_path)

        ## Create a new service called "cspace_map" that accepts messages of
        ## type GetMap and calls self.request_cspacemap() when a message is received
        rospy.Service('cspace_map', GetMap, self.request_cspacemap)

        ## PUBLISHERS

        # Create a publisher for the visualization of C-space (the enlarged occupancy grid)
        # The topic is "/path_planner/cspace", the message type is GridCells
        self.cspace = rospy.Publisher('/path_planner/cspace', GridCells, queue_size=10)

        # Create publishers for visualization of A* (expanded cells, frontier, ...)
        # The topic is "/path_planner/astar", the message type is GridCells
        self.astar = rospy.Publisher('/path_planner/astar', GridCells, queue_size=10)

        ## Initialize the request counter
        self.counter = 0
        ## Sleep to allow roscore to do some housekeeping
        rospy.sleep(1.0)
        ## Initialization complete
        rospy.loginfo("PathPlanner ready")

    def update_heading(self, msg):
        """
        Updates the current heading of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        # Update heading of the robot based off the messagge
        quat_orig = msg.pose.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        self.heading = yaw

    @staticmethod
    def grid_to_index(mapdata, x, y):
        """
        Returns the index corresponding to the given (x,y) coordinates in the occupancy grid.
        :param x [int] The cell X coordinate.
        :param y [int] The cell Y coordinate.
        :return  [int] The index.
        """
        ### REQUIRED CREDIT
        width = mapdata.info.width  # Get width of occupancy grid from mapdata
        index = y * width + x       # Calclate corresponding index of the given coordinates
        return index

    @staticmethod
    def index_to_grid(mapdata, index):
        """
        Returns the grid coordinate corresponding to the given index in the occupancy grid.
        :param index [int] The index.
        :return      [(int,int)]     The cell position as a tuple.
        """
        width = mapdata.info.width  # Get width of occupancy grid from mapdata
        x = index % width           # Calculate corresponding x-coordinate value of the given index
        y = int(index / width)      # Calculate corresponding y-coordinate value of the given index
        return (x,y)

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        """
        Calculates the Euclidean distance between two points.
        :param x1 [int or float] X coordinate of first point.
        :param y1 [int or float] Y coordinate of first point.
        :param x2 [int or float] X coordinate of second point.
        :param y2 [int or float] Y coordinate of second point.
        :return   [float]        The distance.
        """
        x = abs(x2 - x1)
        y = abs(y2 - y1)

        dis = math.sqrt(x**2 + y**2) # Calculate euclidean distance between the two points with Pythagorean Theorem
        return dis
        

    @staticmethod
    def grid_to_world(mapdata, x, y):
        """
        Transforms a cell coordinate in the occupancy grid into a world coordinate.
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The cell X coordinate.
        :param y       [int]           The cell Y coordinate.
        :return        [Point]         The position in the world.
        """
        resolution = mapdata.info.resolution
        origin_x = mapdata.info.origin.position.x   # Get the location of the map origin
        origin_y = mapdata.info.origin.position.y
        world_x = (x + 0.5) * resolution + origin_x # Calculate the corresponding coordinate in world frame
        world_y = (y + 0.5) * resolution + origin_y
        world_point = Point()                       # Create a Point object to return the position
        world_point.x = world_x
        world_point.y = world_y
        return world_point


        
    @staticmethod
    def world_to_grid(mapdata, wp):
        """
        Transforms a world coordinate into a cell coordinate in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param wp      [Point]         The world coordinate.
        :return        [(int,int)]     The cell position as a tuple.
        """
        ### REQUIRED CREDIT
        #pass
        resolution = mapdata.info.resolution
        origin_x = mapdata.info.origin.position.x   # Get the location of the map origin
        origin_y = mapdata.info.origin.position.y

        grid_x = int((wp.x - origin_x)/resolution)  # Calculate the corresponding coordinate in occupancy grid
        grid_y = int((wp.y - origin_y)/resolution)
        return (grid_x,grid_y)                      # Return position as a tuple

    @staticmethod
    def is_cell_walkable(mapdata, x, y):
        """
        A cell is walkable if all of these conditions are true:
        1. It is within the boundaries of the grid;
        2. It is free (not unknown, not occupied by an obstacle)
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
        elif (occupancy[index] >= THRESHOLD):             # 4: Cell is occupied/obstacle
            return False
        return True

               

    @staticmethod
    def neighbors_of_4(mapdata, x, y):
        """
        Returns the walkable 4-neighbors cells of (x,y) in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The X coordinate in the grid.
        :param y       [int]           The Y coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable 4-neighbors.
        """
        width = mapdata.info.width                      # Get map width
        neighbors = []
        index = PathPlanner.grid_to_index(mapdata,x,y)  # Convert occupancy grid coordinate to index
        
        # Find the index of the four neighbors 
        top = index - width 
        left = index - 1
        right = index + 1
        low = index + width
        index_4_neighbors = [top, left, right, low]

        # Append the walkable neighbors to a list and return
        for i in index_4_neighbors:
            coordinate = PathPlanner.index_to_grid(mapdata,i)
            if PathPlanner.is_cell_walkable(mapdata, coordinate[0], coordinate[1]):
                neighbors.append(coordinate)
        return neighbors 

    
    
    @staticmethod
    def neighbors_of_8(mapdata, x, y):
        """
        Returns the walkable 8-neighbors cells of (x,y) in the occupancy grid.
        :param mapdata [OccupancyGrid] The map information.
        :param x       [int]           The X coordinate in the grid.
        :param y       [int]           The Y coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable 8-neighbors.
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

        # Append the walkable neighbors to a list and return
        for i in index_8_neighbors:
            coordinate = PathPlanner.index_to_grid(mapdata,i)
            if PathPlanner.is_cell_walkable(mapdata, coordinate[0], coordinate[1]):
                neighbors.append(coordinate)
        return neighbors 


    
    
    @staticmethod
    def request_map():
        """
        Requests the map from the gmapping
        :return [OccupancyGrid] The grid if the service call was successful,
                                None in case of error.
        """
        rospy.loginfo("Requesting the map")         
        rospy.wait_for_service('/dynamic_map')            # Waiting for map from the map server
        map = rospy.ServiceProxy('/dynamic_map', GetMap) 
        try:
            resp1 = map()                                # Obtain GetMap which contains the map information
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return resp1.map


    def calc_cspace(self, mapdata, padding):
        """
        Calculates the C-Space, i.e., makes the obstacles in the map thicker.
        Publishes the list of cells that were added to the original map.
        :param mapdata [OccupancyGrid] The map data.
        :param padding [int]           The number of cells around the obstacles.
        :return        [OccupancyGrid] The C-Space.
        """
        rospy.loginfo("Calculating C-Space")
        width = mapdata.info.width          # Get map width
        arr = mapdata.data                  # Get occupancy grid array
        
        ## Inflate the obstacles where necessary
        for layer in range(padding):
            #rospy.loginfo("Inflating CSpace")
            padded_map = PathPlanner.dilation(mapdata)   # Inflate obstacle using helper function
            mapdata.data = padded_map       # Update mapdata with the padded map
            #self.print_index_prob(mapdata)  # print the newly padded map

       
        
        border = []                         # Array storing newly padded obstacles in occupancy grid array
        for i in range(len(padded_map)):    # If the obstacle does not appear both on the padded map and the original map,
            if padded_map[i] == arr[i]:     # the obstacle is on the newly padded layer
                border.append(0)
            else:
                border.append(100)

        # If cell is obstacle, convert the index to grid to world, and append to the array
        point_arr = []                      # Array storing newly padded obstacles in world coordinate
        for cell in range(len(border)):     
            if border[cell] >= 50:
                grid = PathPlanner.index_to_grid(mapdata, cell)
                world = PathPlanner.grid_to_world(mapdata, grid[0], grid[1]) # Point
                point_arr.append(world)

        
        ## Create a GridCells message and publish it
        grid_msg = GridCells()
        grid_msg.header.frame_id = mapdata.header.frame_id
        grid_msg.cell_width = mapdata.info.resolution
        grid_msg.cell_height = mapdata.info.resolution
        grid_msg.cells = point_arr        
        self.cspace.publish(grid_msg)
        
        ## Return the C-space as an OccupancyGrid object
        cspace = OccupancyGrid()
        cspace.header.frame_id = 'map'
        cspace.info = mapdata.info
        cspace.data = padded_map
        return cspace 


    def request_cspacemap(self, msg):
        """
        Requests a map that has been padded
        :return [OccupancyGrid] The grid if the service call was successful,
                                None in case of error.
        """
        rospy.loginfo("Requesting the cspace map")         
        mapdata = self.request_map()
        cspacedata = self.calc_cspace(mapdata, 3)
        
        return cspacedata
    
    @staticmethod
    def dilation(mapdata):
        """
        Pad the given map by 1 layer with the given threshold to determine whether cell is an obstacle or not
        Return the padded map
        :param mapdata   [OccupancyGrid] The map data.
        :param threshold [int]           The threshold that determines if cell is an obstacle or not
        """
        arr = mapdata.data  # Get map from mapdata
        
        # Create a new map to for mapping the padded version of the map
        new_map = np.empty(len(arr))
        new_map.fill(-1)

        for i in range(len(arr)):
            # If current cell is an obstacle, and is unknown in new map
            if(arr[i] >= THRESHOLD and new_map[i] == -1): 
                new_map[i] = 100                                # Set it to obstacle (100) in the new map
                coord = PathPlanner.index_to_grid(mapdata, i)   # Get the coordinate of the cell
                
                # Pad the neighbors: find all the non-obstacle neighbors 
                neighbors = PathPlanner.neighbors_of_8(mapdata, coord[0], coord[1])  
                for n in neighbors:             # For every non-obstacle neighbors
                    nIndex = PathPlanner.grid_to_index(mapdata, n[0], n[1]) # Get the coordinate of that neighbor
                    new_map[nIndex] = 100       # Set neighbor to obstacle (100)
            # If current cell is not an obstacle, and is unknown in new map
            elif(arr[i] < THRESHOLD and arr[i] >= 0 and new_map[i] == -1): 
                new_map[i] = 0          # Set it as free space (0) in new map
        return new_map


    
    def a_star(self, mapdata, start, goal):
        """
        Using A* Algorithm to calculate path from start to goal 
        :param mapdata   [OccupancyGrid] The map data.
        :param start (x,y) Starting point
        :param goal  (x,y) Ending point
        """
        rospy.loginfo("Executing A* from (%d,%d) to (%d,%d)" % (start[0], start[1], goal[0], goal[1]))
        frontier = PriorityQueue()      # Create a PriorityQueue object named frontiers
        frontier.put(start, 0)          # Insert the first frontier in the queue
        came_from = {}                  
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        while not frontier.empty():     # When there are still frontiers
            rospy.sleep(0.05)           # Slight delay
            current = frontier.get()    # Get the element with top priority 

            if current == goal:         # Break when algorithm has reached the goal
                break
            
            ## Perform A* algorithm to find the path from start to goal
            for next in PathPlanner.neighbors_of_4(mapdata, current[0], current[1]):   # Every walkable neighor of current cell
                new_cost = cost_so_far[current] + mapdata.data[PathPlanner.grid_to_index(mapdata, next[0], next[1])]    # Calculate cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:                       # Compare cost to check if it needs update
                    cost_so_far[next] = new_cost
                    # Calculate priority with heuristic function: the eculidean distance from current cell to goal
                    priority = new_cost + PathPlanner.euclidean_distance(goal[0], goal[1], next[0],next[1]) 
                    frontier.put(next, priority)                                                  # Insert current cell back to priority queue
                    came_from[next] = current                                                     # Update where it came from
                    
                    
                    ## WAVEFRONT VISUALIZATION
                    # Create a GridCells message and publish the frontier to /path_planner/astar topic
                    frontier_arr = frontier.get_queue()
                    grid_msg = GridCells()
                    grid_msg.header.frame_id = 'map'
                    grid_msg.cell_width = mapdata.info.resolution
                    grid_msg.cell_height = mapdata.info.resolution

                    point_arr = []
                    for cell in frontier_arr:       # Convert every cell from grid to world coordinate
                        cell_grid = cell[1]         # Derive grid coordinate from priority queue structure
                        world = PathPlanner.grid_to_world(mapdata, cell_grid[0], cell_grid[1]) 
                        point_arr.append(world)     # Append the Point object to the array
                    grid_msg.cells = point_arr
                    self.astar.publish(grid_msg)    # Publish the GridCells message to topic

        
        ## Backtracking the path
        came_from_path = []                 # Array storing path
        from_cell = goal                    # Starting from goal
        #print('came from = ', came_from[from_cell])
        try:
            # If A* CAN find a valid path
            while not start == from_cell:       # If the path has not reached start
                visited = came_from[from_cell]  # Get where the current cell came from
                came_from_path.insert(0, visited)   # Insert it to the path array
                from_cell = visited             # Set the current cell to came from cell
        except:
            # If A* CANNOT find a valid path
            rospy.loginfo('A* path not found')
            return None

        came_from_path.append(goal)         # Append to array containing path
        return came_from_path

    @staticmethod
    def optimize_path(path):
        """
        Optimizes the path, removing unnecessary intermediate nodes.
        :param path [[(x,y)]] The path as a list of tuples (grid coordinates)
        :return     [[(x,y)]] The optimized path as a list of tuples (grid coordinates)
        """
        rospy.loginfo("Optimizing path")
        # Initiating array and conditions for path optimization loop
        check_index = 0
        optimized = [path[0]]   # Insert the start waypoints to optimzed path array
        end = False

        if len(path) == 1: return optimized # Immediately return start waypoint if path only contains one waypoint
        while not end:
            next_waypoint = PathPlanner.next_waypoint_index(check_index, path)  # Get the next necessary waypoint
            optimized.append(path[next_waypoint])                               # Append waypoint to optimized path array
            check_index = next_waypoint                                         # Update current index 
            if next_waypoint == len(path) - 1: end = True                       # Exit loop if next waypoint is the last element of given path
        return optimized

    @staticmethod
    def next_waypoint_index(start_index, path):
        """
        Return the next waypoint following the given waypoint in the list of points
        :param start_index [int] The starting waypoint
        :param path        [[(x,y)]] The path as a list of tuples (grid coordinates)
        :return            [int] The next waypoint following the given starting waypoint
        """
        found_distinct = False      
        start = path[start_index]           # Starting waypoint
        index = start_index + 1             # Index of next waypoint
        # Cond #1: return next point if it is the last one
        if index == len(path) - 1: return index 
             
        while not found_distinct:           # If we have not found a distinct waypoint (which mean x and y are both different)
            next_cell = path[index]         # Next waypoint
            # Cond #2: Encounter redundant point, look for next distinct point
            if start[0] == next_cell[0] or start[1] == next_cell[1]: 
                # Cond #3: It's redudant but the last one in list
                if index == len(path) - 1: return index    
                index = index + 1           # Checking the next index
            # Cond #4: Next point is distinct
            elif abs(start[0] - next_cell[0]) == 1 and abs(start[1] - next_cell[1]) == 1:
                found_distinct = True
            else:
                found_distinct = True       # Found a distinct point that break the strike, 
                index = index - 1           # return the previous one as waypoint
        return index

    @staticmethod
    def remove_redundant_turn(mapdata, path):
        """
        Remove excessive turns in the path
        :param mapdata   [OccupancyGrid] The map data.
        :param path      [[(x,y)]] The path as a list of tuples (grid coordinates)
        :return          [[(x,y)]] The optimized path as a list of tuples (grid coordinates)
        """
        # Check if there is a walkable direct path from current cell 'j' to next cell 'i'
        j = 0                                   # Initialize index of current cell
        while j <= len(path)-2:                 # Don't check the last cell
            current = path[j]
            closest_waypoint_index = j+1        # Initialize the closest waypoint index as the next index
            i = j + 1                           # Initialize index of next cell

            while i <= len(path) - 1:           # Check all cell after current cell 
                next = path[i]      

                # If a walkable direct path exists between current cell and next cell
                if PathPlanner.can_go_straight_to(mapdata, current, next):  
                    closest_waypoint_index = i  # Update the closest waypoint index 
                i += 1                          # Update i index to check next cell
            del path[j+1:closest_waypoint_index]    # Delete intermediate cells inbetween current cell and closest waypoint
            j+=1
        return path

    @staticmethod
    def can_go_straight_to(mapdata, start, goal):
        """
        Check if there is a walkable direct path between start and goal
        :param mapdata   [OccupancyGrid] The map data.
        :param start     [(x,y)] The start waypoint
        :param goal      [(x,y)] The goal waypoint
        :return          [boolean] True if there exists a walkable direct path
        """
        start_x = start[0]          # x coordinate of start waypoint
        start_y = start[1]          # y coordinate of start waypoint
        goal_x = goal[0]            # x coordinate of goal waypoint
        goal_y = goal[1]            # y coordinate of goal waypoint

        diff_x = abs(start_x - goal_x)                      # Difference in x

        if start_x == goal_x and start_y < goal_y:          # 1: top
            for i in range (1, goal_y - start_y - 1):
                # If the intermediate waypoint between start and goal are all walkable, return True
                if not PathPlanner.is_cell_walkable(mapdata, start_x, start_y + i): return False
            return True
        elif start_x == goal_x and start_y > goal_y:        # 2: down
            for i in range (1, start_y - goal_y - 1):
                if not PathPlanner.is_cell_walkable(mapdata, start_x, start_y - i): return False
            return True
        elif start_y == goal_y and goal_x > start_x:        # 3: right
            for i in range (1, goal_x - start_x - 1):
                if not PathPlanner.is_cell_walkable(mapdata, start_x + i, start_y): return False
            return True
        elif start_y == goal_y and goal_x < start_x:        # 4: left
            for i in range (1, start_x - goal_x - 1):
                if not PathPlanner.is_cell_walkable(mapdata, start_x - i, start_y): return False
            return True
        # Checking the diagonals
        elif diff_x == abs(start_y - goal_y):               
            if goal_x > start_x and goal_y > start_y:       # 5: top left
                for i in range(1, diff_x - 1):
                    if not PathPlanner.is_cell_walkable(mapdata, start_x + i, start_y + i): return False
                return True
            elif goal_x < start_x and goal_y < start_y:     # 6: bottom right
                for i in range(1, diff_x - 1):
                    if not PathPlanner.is_cell_walkable(mapdata, start_x - i, start_y - i): return False
                return True
            elif goal_x > start_x and goal_y < start_y:     # 7: bottom left
                for i in range(1, diff_x - 1):
                    if not PathPlanner.is_cell_walkable(mapdata, start_x + i, start_y - i): return False
                return True
            elif goal_x < start_x and goal_y > start_y:     # 8: top right
                for i in range(1, diff_x - 1):
                    if not PathPlanner.is_cell_walkable(mapdata, start_x - i, start_y + i): return False
                return True
        return False

    def print_index_prob(self, mapdata):
        """
        Takes in an OccupancyGrid
        Print the 1D OccupancyGrid in a 2D format
        :param mapdata   [OccupancyGrid] The map data.
        """
        a = 0
        map = []
        st = ""
        width = mapdata.info.width
        for n in mapdata.data:
            
            if n >= 50:
                n = "#"
            if n == 0:
                n = "."
            if n == -1:
                n = "?"

            if a == width:
                for el in map:
                    st += el
                print(st)
                a = 0
                map = []
                st = ""
            
            map.append(n)
            a = a + 1
        for el in map:
            st += el       
        print(st)


    @staticmethod
    def path_to_poses(mapdata, path):
        """
        Converts the given path into a list of PoseStamped.
        :param mapdata [OccupancyGrid] The map information.
        :param  path   [[(int,int)]]   The path as a list of tuples (cell coordinates).
        :return        [[PoseStamped]] The path as a list of PoseStamped (world coordinates).
        """
        posestamped_list = []           # Array storing PostStamped 
        for cell in path:
            posestamp = PoseStamped()   # Create a PostStamped Object for every waypoint
            coord = PathPlanner.grid_to_world(mapdata, cell[0], cell[1])    # Get world coordinate of waypoint
            posestamp.header.frame_id = 'map'
            posestamp.pose.position.x = coord.x
            posestamp.pose.position.y = coord.y
            posestamp.pose.position.z = coord.z
            posestamped_list.append(posestamp)  # Append to the array of PostStamped
        return posestamped_list
        

    def path_to_message(self, mapdata, path):
        """
        Takes a path on the grid and returns a Path message.
        :param path [[(int,int)]] The path on the grid (a list of tuples)
        :return     [Path]        A Path message (the coordinates are expressed in the world)
        """
        rospy.loginfo("Returning a Path message")
        path_msg = Path()                   # Create a Path Object storing all waypoint
        path_msg.header.frame_id = 'map'    # Set header frame = map
        path_msg.poses = self.path_to_poses(mapdata,path)   # Convert waypoint to PostStamped
        return path_msg

    def plan_path(self, msg):
        """
        Plans a path between the start and goal locations in the requested.
        Internally uses A* to plan the optimal path.
        :param req 
        """
        ## Request the map
        ## In case of error, return an empty path
        mapdata = PathPlanner.request_map()
        if mapdata is None:
            return Path()
        ## Calculate the C-space and publish it
        cspacedata = self.calc_cspace(mapdata, 3)
        ## Execute A*
        start = PathPlanner.world_to_grid(cspacedata, msg.start.pose.position)
        goal  = PathPlanner.world_to_grid(cspacedata, msg.goal.pose.position)
        path  = self.a_star(cspacedata, start, goal)
        # If A* cannot construct a path, return an empty path
        print("A* path = ", path)
        if path is None or len(path) == 1:
            return Path()
        
        ## Optimize waypoints
        waypoints = PathPlanner.optimize_path(path)
        fewer_turns_waypoints = PathPlanner.remove_redundant_turn(cspacedata, waypoints)
        #print("Optimized path = ", waypoints)
        ## Return a Path message
        messages = self.path_to_message(cspacedata, fewer_turns_waypoints)
        #print("Path in world: ", messages)
        return messages

    
    def run(self):
        """
        Runs the node until Ctrl-C is pressed.
        """
        self.__init__
        rospy.spin()


        
if __name__ == '__main__':
    PathPlanner().run()
    
