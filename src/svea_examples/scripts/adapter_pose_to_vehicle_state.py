#! /usr/bin/env python3

import rospy

import math
from svea_msgs.msg import VehicleState as VehicleStateMsg
from geometry_msgs.msg import PoseStamped, TwistWithCovarianceStamped
from tf.transformations import quaternion_matrix, euler_from_matrix, euler_matrix
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.patches import Ellipse

__license__ = "MIT"
__maintainer__ = "Michele Pestarino, Federico Sacco"
__email__ = "micpes@kth.se, fsacco@ug.kth.se"
__status__ = "Development"

__all__ = [
    'AdapterPoseToVehicleState',
]

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class AdapterPoseToVehicleState:

    def __init__(self, vehicle_name: str = ''):
        """
        Init method for class AdapterPoseToVehicleState

        :param vehicle_name: vehicle name, defaults to ''
        :type vehicle_name: str, optional
        """
        # Offset angle between the mocap frame (of the real world) and the map frame
        self.OFFSET_ANGLE = -math.pi/2
        self.ROTATION_MATRIX_4 = euler_matrix(0, 0, self.OFFSET_ANGLE)
        self.ROTATION_MATRIX_2 = [[math.cos(self.OFFSET_ANGLE), -math.sin(self.OFFSET_ANGLE)],
                                [math.sin(self.OFFSET_ANGLE), math.cos(self.OFFSET_ANGLE)]]

        # Get vehicle name from parameters
        self.vehicle_name = vehicle_name
        # Set svea on board localization topic's name
        self._mocap_pose_topic = load_param('~input_topic', f'/qualisys/{vehicle_name}/pose')
        # Set mocap topic's name
        self._svea_state_topic = load_param('~output_topic', '/state')
        self._velocity_topic = load_param('~velocity_topic', '/actuation_twist')
        self._verbose = load_param('~state_verbose', False)
        # Last velocity published 
        self._current_velocity = 0

        # Define publisher for VehicleState topic
        self._state_pub = None
        # Define subscriber for mocap pose topic
        self._pose_sub = None
        self._vel_sub = None
        # Initialize state message
        self._state_msg = VehicleStateMsg()

    def init_ros_interface(self):
        # Define mocap pose subscriber
        self._pose_sub = rospy.Subscriber(self._mocap_pose_topic,
                         PoseStamped,
                         self._pose_callback,
                         queue_size=1)
        
        self._vel_sub = rospy.Subscriber(self._velocity_topic,
                         TwistWithCovarianceStamped,
                         self._vel_callback,
                         queue_size=1)

        # Define vehicle state publisher 
        self._state_pub = rospy.Publisher(self._svea_state_topic, VehicleStateMsg, queue_size=1)
    
    def _vel_callback(self, msg):
        self._current_velocity = msg.twist.twist.linear.x

    def _pose_callback(self, msg):
        # Correct pose based on the angle offset between map and mocap frames
        x_corr, y_corr, yaw_corr = self._correct_mocap_coordinates(msg.pose.position.x, msg.pose.position.y, msg.pose.orientation)
        # Assing vehicle state message fields
        self._state_msg.header = msg.header
        self._state_msg.header.frame_id = 'map'
        self._state_msg.child_frame_id = 'base_link'
        self._state_msg.x = x_corr
        self._state_msg.y = y_corr
        self._state_msg.yaw = yaw_corr

        if self._verbose:
            print("x = " + str(self._state_msg.x))
            print("y = " + str(self._state_msg.y))
            print("yaw = " + str(self._state_msg.yaw) + "\n")
            
        self._state_msg.v = self._current_velocity
        self._state_msg.covariance = [0] * 16

        # Publish state message
        self._state_pub.publish(self._state_msg)

    def _correct_mocap_coordinates(self, x, y, quaternion):
        """
        Method used to correct the mocap pose (if some misalignment between its frame and the map frame is present)
        
        :param x: x coordinate to be corrected 
        :type x: float
        :param y: y coordinate to be corrected 
        :type y: float
        :param quaternion: quaternion used to extract and correct yaw angle 
        :type quaternion: Quaternion

        :return: rotate_point[0] corrected x coordinate
        :rtype: float
        :return: rotate_point[1] corrected y coordinate
        :rtype: float
        :return: mocap_yaw corrected yaw angle
        :rtype: float
        """
        # Apply rotation matrix
        rotated_point = np.matmul(self.ROTATION_MATRIX_2, np.transpose(np.array([x,y])))
        # Get svea's rotation matrix from pose quaternion
        svea_rotation_matrix = quaternion_matrix([quaternion.x, 
                                                  quaternion.y, 
                                                  quaternion.z, 
                                                  quaternion.w])
        # Apply 4 dimension square rotation matrix 
        rotation_matrix = np.matmul(self.ROTATION_MATRIX_4, svea_rotation_matrix)
        # Get correct yaw
        (_, _, mocap_yaw) = euler_from_matrix(rotation_matrix)
        return rotated_point[0], rotated_point[1], mocap_yaw
        
        
     

if __name__ == '__main__':
    ## Start node ##
    rospy.init_node('adapter_pose_to_vehicle_state')
    # Instiate object of class MeasurementNode
    adapter_node = AdapterPoseToVehicleState(vehicle_name='svea7')
    # Init node and start listeners
    adapter_node.init_ros_interface()
    
    # Spin node 
    rospy.spin()