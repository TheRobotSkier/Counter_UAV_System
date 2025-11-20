import argparse
import numpy as np
import os
from tqdm import tqdm
import sys
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

from drone_pointpillars.utils import read_points, read_label, \
    write_pickle, get_points_num_in_bbox \
    
def judge_difficulty(annotation_dict):
    truncated = annotation_dict['truncated']
    occluded = annotation_dict['occluded']
    bbox = annotation_dict['bbox']
    height = bbox[:, 3] - bbox[:, 1]

    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]
    difficultys = []
    for h, o, t in zip(height, occluded, truncated):
        difficulty = -1
        for i in range(2, -1, -1):
            if h > MIN_HEIGHTS[i] and o <= MAX_OCCLUSION[i] and t <= MAX_TRUNCATION[i]:
                difficulty = i
        difficultys.append(difficulty)
    return np.array(difficultys, dtype=np.int32)


def create_data_info_pkl(data_root, data_type, prefix, label=True, ids=None ,db=False):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    # Okay so depending on whether it's train/val/test we read different files. They just have diffeent orders from 0-X. 

    split = 'training' if label else 'testing'

    kitti_infos_dict = {}
  
    for id in tqdm(ids):
    
        cur_info_dict={}
        
        id_str = str(id).zfill(6)
    
        lidar_path = os.path.join(data_root, split,'set2' ,'bin', f'{id_str}.bin')
 
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-4:])
        
        print(lidar_path)

        try:
            lidar_points = read_points(lidar_path)
        except:
            print(f'Error reading points from: {lidar_path}')
            break
    
        if label:
            label_path = os.path.join(data_root, split, 'set2', 'label', f'{id_str}.txt')
            annotation_dict = read_label(label_path)
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=lidar_points,
                r0_rect=np.eye(4, dtype=np.float32), # Identity matrix for drone dataset (Used to be calib['R0_rect'])
                tr_velo_to_cam=np.eye(4, dtype=np.float32), # Identity matrix for drone dataset (Used to be calib['Tr_velo_to_cam'])
                dimensions=annotation_dict['dimensions'],
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict

        kitti_infos_dict[int(id)] = cur_info_dict

        id += 1

    saved_path = os.path.join(data_root,split,'set2', f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    print(f'{data_type} data info pkl file is saved to {saved_path}')
  
    return kitti_infos_dict

def main(args):
    data_root = args.data_root
    prefix = args.prefix

    # The dataset is split into train/val/test with a 70/15/15 ratio
    total = 56301 # Total number of samples in drone dataset
    all_ids = np.arange(total)

    # We shuffle the ids to create train/val/test splits
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    rng.shuffle(all_ids) # We shuffle everything.

    n_train = total * 70 // 100   # 70% Train
    n_val   = total * 15 // 100   # 15% Val
    n_test  = total - n_train - n_val  # remaining 15% Test

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]

    ## 1. train: create data infomation pkl file 
    ##           && create database(points in gt bbox) for data aumentation
    kitti_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, ids=train_ids, db=False)

    ## 2. val: create data infomation pkl file 
    kitti_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix, ids=val_ids)
    
    ## 3. trainval: create data infomation pkl file
    #kitti_trainval_infos_dict = {**kitti_train_infos_dict, **kitti_val_infos_dict}
    #saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    #write_pickle(kitti_trainval_infos_dict, saved_path)

    ## 4. test: create data infomation pkl file
    kitti_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, label=True, ids=test_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='/ceph/project/P7_Gravel_Gun/Datasets/Drone_Data', 
                        help='your data root for drone dataset')
    parser.add_argument('--prefix', default='drone', 
                        help='the prefix name for the saved .pkl file')
    args = parser.parse_args()

    main(args)