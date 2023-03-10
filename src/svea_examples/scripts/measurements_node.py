#! /usr/bin/env python3

import rospy

import tf
import math
from svea.states import VehicleState
from svea_msgs.msg import VehicleState as VehicleStateMsg
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

__license__ = "MIT"
__maintainer__ = "Michele Pestarino, Federico Sacco"
__email__ = "micpes@kth.se, fsacco@ug.kth.se"
__status__ = "Development"

__all__ = [
    'MeasurementsNode',
]

class MeasurementsNode:
    """Interface handling the reception of state information from the
    localization stack.

    This object can take on several callback functions and execute them as soon
    as state information is available.

    Args:
        vehicle_name: Name of vehicle being controlled; The name will be
            effectively be added as a namespace to the topics used by the
            corresponding localization node i.e `namespace/vehicle_name/state`.
    """
    def __init__(self, vehicle_name: str = ''):

        self.OFFSET_ANGLE = -math.pi/2
        self.ROTATION_MATRIX = [[math.cos(self.OFFSET_ANGLE), -math.sin(self.OFFSET_ANGLE)],
                                [math.sin(self.OFFSET_ANGLE), math.cos(self.OFFSET_ANGLE)]]
        self.COVARIANCE_THRESHOLD = 7
        self.SCATTER_SCALE_COEFF = 350

        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], color = "r", alpha=0.5)
        self.line2, = self.ax.plot([], [], color = "g", alpha=0.5)
        self.covariance_circle = self.ax.scatter([], [], s=1000, color='r', alpha=0.5)

        self.vehicle_name = vehicle_name
        sub_namespace = vehicle_name + '/' if vehicle_name else ''
        self._svea_state_topic = sub_namespace + 'state'
        self._mocap_state_topic = '/qualisys/svea7/pose'

        # current states
        self.svea_state = None
        self.mocap_state = None

        # list of measurements
        self.svea_measurements = []
        self.mocap_measurements = []
    
    def init_and_start_listeners(self):
        rospy.loginfo("Starting Measurements Node for " + self.vehicle_name)
        self.node_name = 'measurement_node'
        self._start_listen()
        rospy.loginfo("{} Measurements Interface successfully initialized"
                      .format(self.vehicle_name))

    def _start_listen(self):
        rospy.Subscriber(self._svea_state_topic,
                         VehicleStateMsg,
                         self._svea_read_state_msg,
                         queue_size=1)
        rospy.Subscriber(self._mocap_state_topic,
                         PoseStamped,
                         self._mocap_read_pose_msg,
                         queue_size=1)

    def _svea_read_state_msg(self, msg):
        self.svea_measurements.append(msg)

    def _mocap_read_pose_msg(self, msg):
        self.mocap_measurements.append(msg)

    def plot_init(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.legend(['Localization', 'Mocap'])
        return [self.line1, self.line2, self.covariance_circle]
    
    def update_plot(self, frame):
        if self.mocap_measurements:
            mocap_xs = []
            mocap_ys = []
            for mocap_pose in self.mocap_measurements:
                x = mocap_pose.pose.position.x 
                y = mocap_pose.pose.position.y
                rotated_point = np.matmul(self.ROTATION_MATRIX, np.transpose(np.array([x,y])))
                mocap_xs.append(rotated_point[0])
                mocap_ys.append(rotated_point[1])
            self.line2.set_data(mocap_xs, mocap_ys)
            
        if self.svea_measurements:
            svea_xs = []
            svea_ys = []
            for svea_pose in self.svea_measurements:
                svea_xs.append(svea_pose.x)
                svea_ys.append(svea_pose.y)
            #if self.svea_measurements[len(self.svea_measurements) - 1].covariance[0] > self.COVARIANCE_THRESHOLD and self.svea_measurements[len(self.svea_measurements) - 1].covariance[5] > self.COVARIANCE_THRESHOLD:
            self.covariance_circle.set_offsets((self.svea_measurements[len(self.svea_measurements) - 1].x, self.svea_measurements[len(self.svea_measurements) - 1].y))
            scatter_area = self.SCATTER_SCALE_COEFF * (self.svea_measurements[len(self.svea_measurements) - 1].covariance[0] + self.svea_measurements[len(self.svea_measurements) - 1].covariance[1])
            print("scatter_area = " + str(scatter_area))
            self.covariance_circle.set_sizes([scatter_area])
            self.line1.set_data(svea_xs, svea_ys)

        return [self.line1, self.line2, self.covariance_circle]

   
if __name__ == '__main__':
    ## Start node ##
    rospy.init_node('measurement_node')
    measurement_node = MeasurementsNode()
    measurement_node.init_and_start_listeners()
    ani_svea = FuncAnimation(measurement_node.fig, measurement_node.update_plot, init_func=measurement_node.plot_init)
    plt.show(block=True) 
    rospy.spin()
    
    