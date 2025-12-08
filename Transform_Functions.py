import math
import numpy as np

#Everything is in mm
#The first transform is used to locate the center of the Aqoustic array foot

#World = [0, 0, 0] #GNNSRTK
GNNSRTK = [0, 0, 0] #aka World

#All of the transforms are to and from the X,Y-center of the objects unless otherwise stated
World_To_Aquostic_Array_Foot_Bottum = [-190.5, 0, 0]
Aquostic_Array_Foot_Top = [-190.5, 0, 100]

World_To_Aquostic_Array_Base_Small_Center = [-190.5, 0, 110] #Use for Small base
World_To_Aquostic_Array_Base_Big_Center = [-190.5, 0, 130] #use for Big base

#Distances between mics for each array size
D_M_Small = 213 #It is 213.018 from all 3 in plane mics to the top. all the 3 in the same plane have 213.042 between them... that is an error on my part of 0.024mm
D_M_Big = 1000 #1 meter, says Emil

#Transfroms for the pan/tiltsystem
PanTilt_Bottum = [758.651, 0.341, 0] #The bottom center of the pantilt system sadly has a y ofset of 0.341mm, womp womp, but that is probably going to be fine....

#Dimensions of the pan tilt system parts
Wood_Width = 10 #Thickness of the wood ie form thp to bottum of wood
Wood_Top_To_Alu_Bottum = 150
Alu_Plate_Width = 10 #Thickness of the alu plate ie form thp to bottum of aliu plate
Top_Of_Alu_To_Center_Of_Tilt_Joint = 65
#Dimensions are combined to make come more eledgeble
Bottum_Of_PanTilt_To_LIdar_Frame = Wood_Width + Wood_Top_To_Alu_Bottum + Alu_Plate_Width + Top_Of_Alu_To_Center_Of_Tilt_Joint

World_To_P = [758.651, 0.341] # is the height from ground to the Tilt joint where the rest of the transform depends on the pan and tilt angles
Center_Of_Tilt_Joint_To_LIdar_Frame = 73 #Used to calculate transform to the lidar frame but this part of the transfor depends on the azimuth and elevation angles

###Ant functions for computing the pan/tilt angles with parallax correction
def point_callback(self, msg):
        """
        Calculates Pan/Tilt to aim at 3D point (x, y, z) with Parallax Correction.
        """
        x = msg.x
        y = msg.y
        z = msg.z 

        # 1. PAN (Azimuth)
        pan_rad = math.atan2(y, x)

        # 2. TILT (Elevation)
        # First, calculate the "ideal" angle from the shoulder to the target
        z_relative = z - TILT_AXIS_Z
        xy_distance = math.sqrt(x*x + y*y)
        distance_3d = math.sqrt(z_relative**2 + xy_distance**2) # Hypotenuse D
        
        # Base elevation angle (Center of motor -> Target)
        base_tilt = math.atan2(z_relative, xy_distance)
        
        # 3. PARALLAX CORRECTION
        # We need to tilt slightly DOWN because the sensor is ABOVE the pivot (when arm is vertical).
        # Triangle: Hypotenuse = Distance, Opposite = Lidar Offset
        # correction = asin(Offset / Distance)
        
        # Safety: Ensure we don't divide by zero or asin(>1)
        if distance_3d > LIDAR_OFFSET:
            correction_angle = math.asin(LIDAR_OFFSET / distance_3d)
            # Subtract correction because arm is "above" the look vector
            tilt_rad = base_tilt - correction_angle
        else:
            # Target is inside the robot's head range!
            self.get_logger().warn("Target too close for parallax correction!")
            tilt_rad = base_tilt

        # 4. Send Commands
        self.send_serial_command('P', pan_rad)
        self.send_serial_command('T', tilt_rad)

        # 5. Visualize
        # IMPORTANT: The marker visualizes the RESULT. 
        # Since we corrected the servo angle, the resulting laser line 
        # (calculated from kinematics) should now land EXACTLY on the target.
        self.publish_aiming_marker(pan_rad, tilt_rad, x, y, z)

# --- PHYSICAL GEOMETRY (Meters) ---
# Distances -> wood: 10mm, wood_to_bottom: 150mm , metal_ceiling_width: 10mm, metal_to_tilt_joint: 65mm, tilt_joint_to_lidar: 73mm
HEIGHT_BASE_TO_PAN = 0.16
HEIGHT_PAN_TO_TILT = 0.075
TILT_AXIS_Z = HEIGHT_BASE_TO_PAN + HEIGHT_PAN_TO_TILT
LIDAR_OFFSET = 0.073
###Ant

#Full world to LIdar and aquostic array transforms
World_To_LIdar_Frame = [758.651, 0.341, 298] #298mm is the height from ground to lidar frame center
print("Transform_Functions.py loaded")
print("World_To_LIdar_Frame:", World_To_LIdar_Frame)
print("World_To_Aquostic_Array_Base_Small_Center:", World_To_Aquostic_Array_Base_Small_Center, "World_To_Aquostic_Array_Base_Big_Center:", World_To_Aquostic_Array_Base_Big_Center)

T_World_Small_Array = [
    [1, 0, 0, -190.5],
    [0, 1, 0, 0],
    [0, 0, 1, 110],
    [0, 0, 0, 1]
]

T_World_Big_Array = [
    [1, 0, 0, -190.5],
    [0, 1, 0, 0],
    [0, 0, 1, 130],
    [0, 0, 0, 1]
]

T_World_Pantilt_Base = [
    [1, 0, 0, 758.651],
    [0, 1, 0, 0.341],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

##
T_World_Lidar = [
    [1, 0, 0, 758.651],
    [0, 1, 0, 0.341],
    [0, 0, 1, 298],
    [0, 0, 0, 1]
]
##Use np.linalg.inv() to find inverse transforms:
# To transform from small array frame to world frame:
T_Small_Array_World = np.linalg.inv(T_World_Small_Array)
##
print(T_Small_Array_World)