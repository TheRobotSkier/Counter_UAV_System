import torch 
import numpy as np
from drone_pointpillars.model import PointPillars
from drone_pointpillars.utils import keep_bbox_from_lidar_range

point_cloud = {}

# Parmeters
pcd_limit_range = np.array([0, -40, -40, 70, 40, 40], dtype=np.float32)

# Init model (Choose small or large model)
if True:
    # Ssmall model 
    trained_path = "/path/to/the/Small.pth"
    model = PointPillars(nclasses=1).cuda()
else:
    # Large model
    trained_path = "/path/to/the/Large.pth"
    model = PointPillars(nclasses=1, max_num_points=100, Backbone_layer_strides=[1,2,2],upsample_strides=[1,2,4]).cuda()
    
# Load weights
model.load_state_dict(torch.load(trained_path))
model.eval()

def point_range_filter(pts, point_range=[0, -40, -40, 70, 40, 40]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

with torch.no_grad():
    while True:
        # Remove points out of range 
        pc = point_range_filter(pc, pcd_limit_range)

        # Run the model
        result = model(batched_pts=pc, mode='test')

        # Remove predictions out of range
        result_filter = keep_bbox_from_lidar_range(result, pcd_limit_range)

        # Split up results
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']




