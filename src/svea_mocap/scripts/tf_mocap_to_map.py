#! /usr/bin/env python3  
import roslib

import rospy
import tf
import math

# TODO: angle between map and mocap as a parameter passed to the main function

if __name__ == '__main__':
	rospy.init_node('tf_mocap_to_map')
	br = tf.TransformBroadcaster()
	rate = rospy.Rate(10.0)
	while not rospy.is_shutdown():
		br.sendTransform((0.06, -0.06, 0.0), tf.transformations.quaternion_from_euler(0, 0, math.pi/2),  rospy.Time.now(), "map", "mocap")
		rate.sleep()
