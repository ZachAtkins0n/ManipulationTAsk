#!/usr/bin/env python3

"""
A socket server that listens and receives incoming joint position data from
the Isaac Lab simulation policy. It then combines with the ur_ros_drivers package
to publish the robot joints to a Universal Robot 3
"""

import rospy
from sensor_msgs.msg import JointState
import socket
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import csv
import time

HOST = "" #Change to device host address  
PORT = 65432   

def main():
    
    #Write time, joint name and pos to csv file
    csv_file = open('read_incoming_joints.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['time', 'name', 'position'])
    
    #Set up time interval
    curr_time =0
    prev_time = time.time()
    
    # Create server socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen()

    print("[INFO] Server is waiting for connections...")
    
    buffer = ""
    connected = True
    while connected:

        conn, addr = sock.accept()  # Accept a new connection
        print(f"[INFO] Connected by {addr}")

        rospy.init_node("ur3_joint_pub", anonymous=True)
        pub = rospy.Publisher("/scaled_pos_joint_traj_controller/command",JointTrajectory, queue_size=True)
        traj = JointTrajectory()
        ur3_joint_names = ["shoulder_pan_joint","shoulder_lift_joint", "elbow_joint", 
                           "wrist_1_joint", "wrist_2_joint","wrist_3_joint"]
        traj.joint_names=ur3_joint_names

        point = JointTrajectoryPoint()
        
        rospy.sleep(1)
        
        try:
            while True:
                data = conn.recv(1024)  #sc Receive data
                if not data:
                    print("[INFO] Connection closed by client.")
                    connected = False
                    break  # Client disconnected
                
                # Convert received data back to list of floats
                buffer += data.decode("utf-8")
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n',1)
                
                try:
                    rcv_joint_array = list(map(float, line.split(',')))
                
                    if len(rcv_joint_array) == 14:
                            
                            #Get time interval for message sent
                            curr_time = time.time()
                            interval = curr_time-prev_time

                            #Send the joint position to the UR3 via ur_ros_drivers package
                            point.time_from_start=rospy.Duration(1.0)
                            point.positions=rcv_joint_array[0:6]
                            traj.points = [point]
                            pub.publish(traj)

                            #Ensure message is published
                            rospy.sleep(0.2)

                except ValueError:
                    pass
                   
        except (ConnectionResetError, BrokenPipeError):
            print("[WARNING] Connection lost! Waiting for a new client...")

        finally:
            conn.close()

        csv_file.close()

if __name__ == "__main__":
    main()

    