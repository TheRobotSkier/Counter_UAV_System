import numpy as np
import numpy.ma as ma

# input will be on the form of:
# From point pillars:
# 1 Position of UAV bounding box (x,y,z) in world coordinates (ask about format) 
# 2 Orientation of UAV bounding box in world coordinates (ask about format)
# 3 Dimentions of the bounding box (length, width, height)
# 4 Confidence score
# 5 timestamp

# Aquistic DOA vector:
# 6 azimuth angle 
# 7 elevation angle 
# 8 confidence score 
# 9 timestamp
# maybe multiple ones of these per frame
# no previos weight to account for inacurasy

#additionsal information:
# timestamp
# camera frame position and orientation in world coordinates

# Output will be used to controll the pan tilt system to point at the UAV:
# 1 Position of UAV bounding box (x,y,z) in world coordinates (ask about format) 
# 2 Orientation of UAV bounding box in world coordinates (ask about format)
# 3 Timestamp
# maybe add confidence score of the estimated position
# maybe add velocity vector of the UAV
# maybe add acceleration vector of the UAV
# maybe add number of particles used in the estimation
# maybe add resampling information
AqData = np.array([37,50])
PointPillarData = np.array([])
UAV_Estimation = np.array([])

def AquisticDOAData(AqData_Input): 
    DOAData = AqData_Input
    return DOAData

Print_DOA_Text = AquisticDOAData(AqData)
print("DOA Data: ", Print_DOA_Text)


