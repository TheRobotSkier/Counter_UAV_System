import math
import numpy as np
from src.cuav_acoustic.cuav_acoustic.config import regular_tetrahedron_array

#Everything is in mm

# True = print transforms, False = no prints
PRINT=False

#The first transform is used to locate the center of the Aqoustic array foot

BASE = np.array([0, 0, 0]) #aka Base station frame

#------ Acoustics Array Transforms and Dimensions ------#

#All of the transforms/translations are to and from the X,Y-center of the objects unless otherwise stated
BASE_2_ARRAY_FOOT_B = np.array([-190.5, 0, 0]) # From the base station frame to the center of the acoustic array foot bottom

ARRAY_FOOT_B_2_ARRAY_FOOT_T = np.array([0, 0, 100]) #From the bottom center of the acoustic array foot to the top center of the foot

ARRAY_FOOT_T_2_ARRAY_BASE_S_C = np.array([0, 0, 10]) #From the top center of the acoustic array foot to the small array base center
ARRAY_FOOT_T_2_ARRAY_BASE_L_C = np.array([0, 0, 30]) #From the top center of the acoustic array foot to the large array base center

ARRAY_BASE_S_C_2_ARRAY_BASE_S_TOP = np.array([0, 0, 20]) #From small array base center to top of small array base
ARRAY_BASE_L_C_2_ARRAY_BASE_L_TOP = np.array([0, 0, 30]) #From large array base center to top of large array base

MIC_LENGTH_TOTAL = 192 #The total length of the micophones
MIC_BUTTOM = 10 # The buttom part of the micophone that is with in the conector
MIC_LENGTH = MIC_LENGTH_TOTAL - MIC_BUTTOM #The length the sensor of the micophone ir raised above the conector

# From the base of the array to the origin of the mic positions (the center of the tetrahedron)
ARRAY_BASE_2_ARRAY_MIC_ORIGIN = np.array([0,0,MIC_LENGTH]) #From the base of the array to the origin of the mic positions (the center of the tetrahedron)

# Full translation from BASE to each mic array base center:
BASE_2_ARRAY_L_MIC_CENTER = BASE_2_ARRAY_FOOT_B + ARRAY_FOOT_B_2_ARRAY_FOOT_T + ARRAY_FOOT_T_2_ARRAY_BASE_L_C
BASE_2_ARRAY_S_MIC_CENTER = BASE_2_ARRAY_FOOT_B + ARRAY_FOOT_B_2_ARRAY_FOOT_T + ARRAY_FOOT_T_2_ARRAY_BASE_S_C

# Full translation from BASE to each mic array base top:
BASE_2_ARRAY_L_MIC_TOP = BASE_2_ARRAY_L_MIC_CENTER + ARRAY_BASE_L_C_2_ARRAY_BASE_L_TOP
BASE_2_ARRAY_S_MIC_TOP = BASE_2_ARRAY_S_MIC_CENTER + ARRAY_BASE_S_C_2_ARRAY_BASE_S_TOP

# Full translation from BASE to each mic array origin:
BASE_2_ARRAY_L_MIC_ORIGIN = BASE_2_ARRAY_L_MIC_CENTER + ARRAY_BASE_2_ARRAY_MIC_ORIGIN
BASE_2_ARRAY_S_MIC_ORIGIN = BASE_2_ARRAY_S_MIC_CENTER + ARRAY_BASE_2_ARRAY_MIC_ORIGIN


DM_S = 213 #Distance between mics for small array, it t is 213.018 from all 3 in plane mics to the top. all the 3 in the same plane have 213.042 between them... that is an error on my part of 0.024 mm
DM_L = 1000 #Distance between mics for large array

# Mic positions for normal tetrahedron array:
MIC_POSITIONS_NORMAL = regular_tetrahedron_array(DM_L)

# the array is mounted 180 degrees rotated around z axis compared to the BASE frame
R_Z_180 = np.array([
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0, 1]
])

BASE_2_ARRAY_L_MIC_ORIGIN_2_MIC_POSITIONS = MIC_POSITIONS_NORMAL@R_Z_180.T #Rotation to match the real mic positions

# Dimensions for the RTK transform calibration
GPS_TRANSFORM_PLATE_PRINT= 10 #Thickness of the 3D printed plate that holds the GPS receiver for RTK transform calibration
METAL_GROUND_PLATE=1 #Thickness of the metal plate that is in between the GPS receiver and the 3D printed plate
GPS_TRANSFORM_PLATE = GPS_TRANSFORM_PLATE_PRINT + METAL_GROUND_PLATE #Total thickness from base station to GPS receiver
ARRAY_BASE_L_TOP_2_GPS = np.array([0,0,GPS_TRANSFORM_PLATE]) #From mic_base to GPS receiver


BASE_2_GPS_MIC_I=BASE_2_ARRAY_L_MIC_TOP + ARRAY_BASE_L_TOP_2_GPS + BASE_2_ARRAY_L_MIC_ORIGIN_2_MIC_POSITIONS

#------ Pan/Tilt System Transforms and Dimensions ------#
#Transfroms for the pan/tiltsystem
BASE_2_PAN_TILT_B = [758.651, 0.341, 0] # From the base station frame to the bottom center of the pantilt system
#The bottom center of the pantilt system sadly has a y ofset of 0.341mm, womp womp, but that is probably going to be fine....

#Dimensions of the pan tilt system parts
WOOD_WIDTH = 10 #Thickness of the wood ie form thp to np.array([-190.5, 0, 130]) bottum of wood
WOOD_TOP_2_ALU_BOTTUM = 150
ALU_PLATE_WIDTH = 10 #Thickness of the alu plate ie form thp to bottum of aliu plate
TOP_OF_ALU_2_CENTER_OF_TILT_JOINT = 65



#Dimensions are combined to make come more eledgeble
Bottum_Of_PanTilt_To_LIdar_Frame = WOOD_WIDTH + WOOD_TOP_2_ALU_BOTTUM + ALU_PLATE_WIDTH + TOP_OF_ALU_2_CENTER_OF_TILT_JOINT

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

#Full world to LIdar and acoustic array transforms
BASE_2_LIDAR = [758.651, 0.341, 298] #298mm is the height from ground to lidar frame center


# ------ Transformatic matrix ------#

T_BASE_2_ARRAY_S_MIC_CENTER = np.array([
    [1, 0, 0, BASE_2_ARRAY_S_MIC_CENTER[0]],
    [0, 1, 0, BASE_2_ARRAY_S_MIC_CENTER[1]],
    [0, 0, 1, BASE_2_ARRAY_S_MIC_CENTER[2]],
    [0, 0, 0, 1]
])

T_BASE_2_ARRAY_L_MIC_CENTER = np.array([
    [1, 0, 0, BASE_2_ARRAY_L_MIC_CENTER[0]],
    [0, 1, 0, BASE_2_ARRAY_L_MIC_CENTER[1]],
    [0, 0, 1, BASE_2_ARRAY_L_MIC_CENTER[2]],
    [0, 0, 0, 1]
])

T_BASE_2_ARRAY_S_MIC_ORIGIN = np.array([
    [1, 0, 0, BASE_2_ARRAY_S_MIC_ORIGIN[0]],
    [0, 1, 0, BASE_2_ARRAY_S_MIC_ORIGIN[1]],
    [0, 0, 1, BASE_2_ARRAY_S_MIC_ORIGIN[2]],
    [0, 0, 0, 1]
])

T_BASE_2_ARRAY_L_MIC_ORIGIN = np.array([
    [1, 0, 0, BASE_2_ARRAY_L_MIC_ORIGIN[0]],
    [0, 1, 0, BASE_2_ARRAY_L_MIC_ORIGIN[1]],
    [0, 0, 1, BASE_2_ARRAY_L_MIC_ORIGIN[2]],
    [0, 0, 0, 1]
])

T_BASE_2_GPS_MIC_1 = np.array([
    [1, 0, 0, BASE_2_GPS_MIC_I[0][0]],
    [0, 1, 0, BASE_2_GPS_MIC_I[0][1]],
    [0, 0, 1, BASE_2_GPS_MIC_I[0][2]],
    [0, 0, 0, 1]
])

T_BASE_2_GPS_MIC_2 = np.array([
    [1, 0, 0, BASE_2_GPS_MIC_I[1][0]],
    [0, 1, 0, BASE_2_GPS_MIC_I[1][1]],
    [0, 0, 1, BASE_2_GPS_MIC_I[1][2]],
    [0, 0, 0, 1]
])

T_BASE_2_GPS_MIC_3 = np.array([
    [1, 0, 0, BASE_2_GPS_MIC_I[2][0]],
    [0, 1, 0, BASE_2_GPS_MIC_I[2][1]],
    [0, 0, 1, BASE_2_GPS_MIC_I[2][2]],
    [0, 0, 0, 1]
])

T_BASE_2_GPS_MIC_4 = np.array([
    [1, 0, 0, BASE_2_GPS_MIC_I[3][0]],
    [0, 1, 0, BASE_2_GPS_MIC_I[3][1]],
    [0, 0, 1, BASE_2_GPS_MIC_I[3][2]],
    [0, 0, 0, 1]
])

T_BASE_Pantilt_Base = [
    [1, 0, 0, 758.651],
    [0, 1, 0, 0.341],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

##
T_BASE_Lidar = [
    [1, 0, 0, 758.651],
    [0, 1, 0, 0.341],
    [0, 0, 1, 298],
    [0, 0, 0, 1]
]
##Use np.linalg.inv() to find inverse transforms:
# To transform from small array frame to world frame:
T_ARRAY_L_MIC_CENTER_2_BASE = np.linalg.inv(T_BASE_2_ARRAY_L_MIC_CENTER)
##
if PRINT:
    print("Transform_Functions.py loaded")
    print("BASE_2_LIDAR:", BASE_2_LIDAR)
    print("T_ARRAY_L_MIC_CENTER_2_BASE:\n", T_ARRAY_L_MIC_CENTER_2_BASE)