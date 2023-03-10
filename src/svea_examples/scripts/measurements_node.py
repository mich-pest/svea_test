#! /usr/bin/env python3

import rospy

import math
from svea_msgs.msg import VehicleState as VehicleStateMsg
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.patches import Ellipse

__license__ = "MIT"
__maintainer__ = "Michele Pestarino, Federico Sacco"
__email__ = "micpes@kth.se, fsacco@ug.kth.se"
__status__ = "Development"

__all__ = [
    'MeasurementsNode',
]

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class MeasurementsNode:
    """Interface handling the reception of state information from the
    localization stack and the mocap system.

    This object instatiates two subscribers (i.e. one for the state topic given by the 
    localization algorithm, one for the pose topic given by the ground truth, in this case the
    mocap system).

    Args:
        vehicle_name: Name of vehicle being controlled; The name will be
            effectively be added as a namespace to the topics used by the
            corresponding localization node i.e `namespace/vehicle_name/state`.
    """
    def __init__(self, vehicle_name: str = ''):
        """
        Init method for class MeasuremenstNode

        :param vehicle_name: vehicle name, defaults to ''
        :type vehicle_name: str, optional
        """
        # Offset angle between the mocap frame (of the real world) and the map frame
        self.OFFSET_ANGLE = -math.pi/2
        # Create rotation matrix given the offset angle
        self.ROTATION_MATRIX = [[math.cos(self.OFFSET_ANGLE), -math.sin(self.OFFSET_ANGLE)],
                                [math.sin(self.OFFSET_ANGLE), math.cos(self.OFFSET_ANGLE)]]
        # Instatiate figure
        self.fig, self.ax = plt.subplots()
        # First line for the localization visualization
        self.line1, = self.ax.plot([], [], color = "r", alpha=0.5)
        # Second line for the mocap visualization
        self.line2, = self.ax.plot([], [], color = "g", alpha=0.5)
        # Ellipse for visualization localization covariance
        self.covariance_ellipse = Ellipse(xy = (0, 0), width=0, height=0, alpha=0.5, edgecolor='r', fc='r')

        # Get vehicle name from parameters
        self.vehicle_name = vehicle_name
        # Set svea on board localization topic's name
        self._svea_state_topic = load_param('localization_topic', '/state')
        # Set mocap topic's name
        self._mocap_state_topic = load_param('ground_truth_topic', f'/qualisys/{vehicle_name}/pose')
        # current states
        self.svea_state = None
        self.mocap_state = None

        # list of measurements
        self.svea_measurements = []
        self.mocap_measurements = []
    
    def init_and_start_listeners(self):
        """
        Inits the node and start the two listeners
        """
        rospy.loginfo("Starting Measurements Node for " + self.vehicle_name)
        self.node_name = 'measurement_node'
        # Instatiates subscribers
        self._start_listen()
        rospy.loginfo("{} Measurements Interface successfully initialized"
                      .format(self.vehicle_name))

    def _start_listen(self):
        """
        Instatiates the two listeners
        """
        # First subcriber for svea localization algorithm's pose topic
        rospy.Subscriber(self._svea_state_topic,
                         VehicleStateMsg,
                         self._svea_read_state_msg,
                         queue_size=1)
        # Second subscriber for mocap localization pose
        rospy.Subscriber(self._mocap_state_topic,
                         PoseStamped,
                         self._mocap_read_pose_msg,
                         queue_size=1)

    def _svea_read_state_msg(self, msg):
        """
        Callback method for the svea's localization algorithnm 

        :param msg: message from the localization algorithm's topic
        :type msg: VehicleStateMsg
        """
        # Append new svea state message to corresponding list
        self.svea_measurements.append(msg)

    def _mocap_read_pose_msg(self, msg):
        """
        Callback method for the ground truth localization algorithnm

        :param msg: message ground truth localization algorithnm
        :type msg: PoseStamped
        """
        # Append new mocap pose message to corresponding list
        self.mocap_measurements.append(msg)

    def plot_init(self):
        """
        Inits plot

        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # Set axis' limits to the plot
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        # Set legend for the two lines
        self.ax.legend(['Localization', 'Mocap'])
        # Add covariance ellipse
        self.ax.add_patch(self.covariance_ellipse)
        # Returns the two lines
        return [self.line1, self.line2]
    
    def update_plot(self, frame):
        """
        Method called by the FuncAnimation for updating the plot

        :param frame: frame of the animation
        :type frame: _type_
        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # If mocap measurements were received
        if self.mocap_measurements:
            # Instatiate lists for xs and ys to be plotted on the figure
            mocap_xs = []
            mocap_ys = []
            # Iterate over every mocap measurements (possibly too computationally demanding over long timespans)
            for mocap_pose in self.mocap_measurements:
                # Get x and y positions
                x = mocap_pose.pose.position.x 
                y = mocap_pose.pose.position.y
                # Apply rotation matrix
                rotated_point = np.matmul(self.ROTATION_MATRIX, np.transpose(np.array([x,y])))
                # Append rotated coordinates
                mocap_xs.append(rotated_point[0])
                mocap_ys.append(rotated_point[1])
            # Set data for line2
            self.line2.set_data(mocap_xs, mocap_ys)
            
        # If svea's localization measurments were received
        if self.svea_measurements:
            # Instatiate lists for xs and ys to be plotted on the figure
            svea_xs = []
            svea_ys = []
            # Iterate over every svea localization measurements (possibly too computationally demanding over long timespans)
            for svea_pose in self.svea_measurements:
                # Append coordinates
                svea_xs.append(svea_pose.x)
                svea_ys.append(svea_pose.y)
            
            # Update covariance Ellipse (center position, width and height based off of the variance)
            self.covariance_ellipse.set_center(xy = (self.svea_measurements[len(self.svea_measurements) - 1].x, self.svea_measurements[len(self.svea_measurements) - 1].y))
            self.covariance_ellipse.width = math.sqrt(self.svea_measurements[len(self.svea_measurements) - 1].covariance[0])
            self.covariance_ellipse.height = math.sqrt(self.svea_measurements[len(self.svea_measurements) - 1].covariance[5])  
            # Set data for line1  
            self.line1.set_data(svea_xs, svea_ys)
        # Return the two lines
        return [self.line1, self.line2]

   
if __name__ == '__main__':
    ## Start node ##
    rospy.init_node('measurements_node')
    # Instiate object of class MeasurementNode
    measurement_node = MeasurementsNode(vehicle_name='svea7')
    # Init node and start listeners
    measurement_node.init_and_start_listeners()
    # Create animation for the plot
    ani_svea = FuncAnimation(measurement_node.fig, measurement_node.update_plot, init_func=measurement_node.plot_init)
    # Show the figure
    plt.show(block=True)
    # Spin node 
    rospy.spin()
    
    