from threading import Thread, Event
from typing import Callable

import rospy

from svea.states import VehicleState
from svea_msgs.msg import VehicleState as VehicleStateMsg
from geometry_msgs.msg import PoseStamped

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

    def __init__(self, vehicle_name: str = 'svea7'):
        self.vehicle_name = vehicle_name
        sub_namespace = vehicle_name + '/' if vehicle_name else ''
        self._svea_state_topic = sub_namespace + 'state'
        self._mocap_state_topic = '/qualisys/svea7/pose'

        self.state = VehicleState()
        self.last_time = float('nan')

        self.is_ready = False
        self._ready_event = Event()

        # list of functions to call whenever a new state comes in
        self.callbacks_svea = []
        self.callbacks_mocap = []
        self.svea_measurements = []
        self.mocap_measurements = []

    def start(self) -> 'MeasurementsNode':
        """Spins up ROS background thread; must be called to start receiving
        data.
        """
        Thread(target=self._init_and_spin_ros, args=()).start()
        return self
    
    def _wait_until_ready(self, timeout=20.0):
        tic = rospy.get_time()
        self._ready_event.wait(timeout)
        toc = rospy.get_time()
        wait = toc - tic
        return wait < timeout
    
    def _init_and_spin_ros(self):
        rospy.loginfo("Starting Measurements Node for " + self.vehicle_name)
        self.node_name = 'measurement_node'
        self._start_listen()
        
        self.is_ready = self._wait_until_ready()
        if not self.is_ready:
            rospy.logwarn("Measurements node not responding during start of "
                          "Measurements Interface. Setting ready anyway.")
        self.is_ready = True

        rospy.loginfo("{} Measurements Interface successfully initialized"
                      .format(self.vehicle_name))

        self.add_callback_mocap(self._mocap_read_state_msg_specific_cbk)
        self.add_callback_svea(self._svea_read_state_msg_specific_cbk)

        # Debug
        self.add_callback_svea(self.print_measurements)

        rospy.spin()

    def _start_listen(self):
        rospy.Subscriber(self._svea_state_topic,
                         VehicleStateMsg,
                         self._svea_read_state_msg,
                         tcp_nodelay=True,
                         queue_size=1)
        
        rospy.Subscriber(self._mocap_state_topic,
                         PoseStamped,
                         self._mocap_read_state_msg,
                         tcp_nodelay=True,
                         queue_size=1)
        
    def _svea_read_state_msg(self, msg):
        self.state.state_msg = msg
        self.last_time = rospy.get_time()
        self._ready_event.set()
        self._ready_event.clear()

        for cb in self.callbacks_svea:
            cb(self.state)

    def _mocap_read_state_msg(self, msg):
        self.state.state_msg = msg
        self.last_time = rospy.get_time()
        self._ready_event.set()
        self._ready_event.clear()

        for cb in self.callbacks_mocap:
            cb(self.state)

    def _svea_read_state_msg_specific_cbk(self, msg):
        self.state.state_msg = msg
        self.last_time = rospy.get_time()
        self.svea_measurements.append(msg)

    def _mocap_read_state_msg_specific_cbk(self, msg):
        self.state.state_msg = msg
        self.last_time = rospy.get_time()
        self.mocap_measurements.append(msg.pose)

    def add_callback_svea(self, cb: Callable[[VehicleState], None]):
        """Add svea state callback.

        Every function passed into this method will be called whenever new
        state information comes in from the localization stack.

        Args:
            cb: A callback function intended for responding to the reception of
                state info.
        """
        self.callbacks_svea.append(cb)

    def remove_callback_svea(self, cb: Callable[[VehicleState], None]):
        """Remove svea state callback so it will no longer be called when state
        information is received.

        Args:
            cb: A callback function that should be no longer used in response
            to the reception of state info.
        """
        while cb in self.callbacks_svea:
            self.callbacks_svea.pop(self.callbacks_svea.index(cb))

    def add_callback_mocap(self, cb: Callable[[PoseStamped], None]):
        """Add mocap pose callback.

        Every function passed into this method will be called whenever new
        state information comes in from the localization stack.

        Args:
            cb: A callback function intended for responding to the reception of
                state info.
        """
        self.callbacks_mocap.append(cb)

    def remove_callback_mocap(self, cb: Callable[[PoseStamped], None]):
        """Remove svea state callback so it will no longer be called when state
        information is received.

        Args:
            cb: A callback function that should be no longer used in response
            to the reception of state info.
        """
        while cb in self.callbacks_mocap:
            self.callbacks_mocap.pop(self.callbacks_mocap.index(cb))

    def print_measurements(self):
        print("svea_meaurements = " + str(self.svea_measurements))
        print("mocap_meaurements = " + str(self.mocap_measurements))
    
