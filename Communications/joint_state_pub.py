#!/usr/bin/env python3

"""
A socket server that contains a ros node using ROS NOETIC 
to publish joint data as a standalone away form simulator

A subscriber will be able to read the joint positions theoretically to another like for like robot.
Currently works in Isaac Sim running the UR3GripperROS.usd file that makes use of action graphs.
"""

import rospy
from sensor_msgs.msg import JointState
import socket

HOST = '127.0.0.1'  
PORT = 65432   

def main():
    
    # Create server socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen()

    print("[INFO] Server is waiting for connections...")

    while True:
        conn, addr = sock.accept()  # Accept a new connection
        print(f"[INFO] Connected by {addr}")

        rospy.init_node("ur3_joint_pub", anonymous=True)
        pub = rospy.Publisher("/joint_command", JointState, queue_size=10)
        joint_state =JointState()
        joint_state.name=["shoulder_pan_joint","shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint","wrist_3_joint", "finger_joint",
                           "right_outer_knuckle_joint","left_outer_finger_joint","right_outer_finger_joint",
                           "left_inner_finger_joint","right_inner_finger_joint","right_inner_finger_pad_joint","left_inner_finger_pad_joint"]
        try:
            while True:
                data = conn.recv(1024)  # Receive data
                if not data:
                    print("[INFO] Connection closed by client.")
                    break  # Client disconnected
                
                # Convert received data back to list of floats
                action_array = list(map(float, data.decode('utf-8').split(',')))  
                print("Received actions:", action_array)
                joint_state.position=action_array
                pub.publish(joint_state)
        except (ConnectionResetError, BrokenPipeError):
            print("[WARNING] Connection lost! Waiting for a new client...")

        finally:
            conn.close()  # Ensure socket is closed before accepting new connections


if __name__ == "__main__":
    main()

    