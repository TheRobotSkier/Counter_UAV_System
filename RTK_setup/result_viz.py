import os
import argparse
import numpy as np
import pickle

import open3d as o3d

#export XDG_SESSION_TYPE=x11

def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def load_predictions(file_path):
    """
    Returns dict: frame_id -> {'gt': [floats] or None, 'preds': [[floats], ...]}
    Skips a header row if present.
    """
    by_frame = {}
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(',')]
            # skip header or malformed rows
            try:
                frame = int(parts[0])
            except Exception:
                continue
            if len(parts) < 4:
                continue
            label = parts[2].upper()
            try:
                box = [float(x) for x in parts[3:] if x != '']
            except ValueError:
                continue
            entry = by_frame.setdefault(frame, {'gt': None, 'preds': []})
            if label == 'GT':
                entry['gt'] = box
            else:
                entry['preds'].append(box)
    return by_frame

def points_to_pcd(pts):
    """Convert Nx4 array (x,y,z,intensity) to open3d PointCloud."""
    pcd = o3d.geometry.PointCloud()
    xyz = pts[:, :3].astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if pts.shape[1] > 3:
        inten = pts[:, 3]
        # map intensity to gray color
        c = np.repeat(((inten - inten.min()) / (np.ptp(inten) + 1e-8))[:, None], 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

def yaw_to_R(yaw):
    c = np.cos(yaw); s = np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def box_to_lineset(box, color=(1, 0, 0)):
    """
    box: list-like, expects at least 7 values:
        [x, y, z, dx, dy, dz, yaw, ...]
    returns open3d.geometry.LineSet
    """
    x, y, z, dx, dy, dz = box[:6]
    yaw = box[6] if len(box) > 6 else 0.0
    center = np.array([x, y, z], dtype=np.float64)
    extent = np.array([dx, dy, dz], dtype=np.float64)
    R = yaw_to_R(yaw)
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.colors = o3d.utility.Vector3dVector([color for _ in range(len(ls.lines))])
    return ls

def main(args):
    # Load gt boxes and their corresponding predictions
    preds_by_frame = load_predictions(args.predictions)

    # The path to the drone dataset point clouds
    data_infos = read_pickle(os.path.join(args.point_cloud, f'drone_infos_val.pkl'))
    sorted_ids = list(data_infos.keys())

    for i, index in enumerate(sorted_ids):
        # Load point cloud
        data_info = data_infos[index]
        point_cloud = data_info['velodyne_path']

        pts_path = os.path.join(args.point_cloud, point_cloud)
        pts = read_points(pts_path)

        print(f'Point cloud path: {pts_path}')
        # Get GT and preds for frame i (first CSV column must match i)
        entry = preds_by_frame.get(i)
        if entry is None:
            print(f'No predictions for frame {i}')
            continue

        gt_box = entry['gt']
        pred_boxes = entry['preds']

        print(f'Frame {i}: pts {pts.shape}, GT: {gt_box}, #preds: {len(pred_boxes)}')

        # Visualize using Open3D (blocks until window closed)
        geoms = []
        try:
            pcd = points_to_pcd(pts)
            geoms.append(pcd)
            if gt_box is not None:
                geoms.append(box_to_lineset(gt_box, color=(0, 1, 0)))  # green GT
            for pb in pred_boxes:
                # draw predictions in red
                geoms.append(box_to_lineset(pb, color=(1, 0, 0)))
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0.0, 0.0, 0.0])
                geoms.append(axis)
            o3d.visualization.draw_geometries(geoms, window_name=f'Frame {i}', width=1024, height=768)
        except Exception as e:
            print(f'Visualization failed for frame {i}: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--point_cloud', default='/home/gustav/Documents/PP_Viz', 
                        help='your data root for drone dataset')
    parser.add_argument('--predictions', default='/home/gustav/Documents/PP_Viz/Small_20_Gen4_predictions.csv', help='your predictions for drone dataset')

    args = parser.parse_args()

    main(args)
    