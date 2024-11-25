import collections 

import json

import numpy as np
import torch

from collections import defaultdict
from misc.log_utils import log
from pathlib import Path

def load_data(file_pathes, all_data=False, use_gt=False, mot_gt_dir=None):
    data_list = defaultdict(dict)

    mot_gt = dict()
    
    for file_path in file_pathes:
        with open(file_path, 'r') as file:
            for line in file:
                frame_data = json.loads(line)
                mot_set = frame_data["img_path"].split("/")[-3]

                if mot_set not in mot_gt and use_gt:
                    mot_gt[mot_set] = load_mot_gt(Path(mot_gt_dir.format(mot_set))) 
                
                if mot_set[-3:] != "DPM" and not all_data:
                    continue
                
                frame_id = int(frame_data["img_path"].split("/")[-1].split(".")[0])
                
                if frame_id in data_list[mot_set]:
                    log.warning(f"Frame {frame_id} already in {mot_set}")

                if use_gt:
                    frame_data["bboxes"] = [gt_det[2:6] for gt_det in mot_gt[mot_set][frame_id]]
                    frame_data["bboxes"] = [[int(bbox[0]), int(bbox[1]), int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])] for bbox in frame_data["bboxes"]]
                    frame_data["id"] = np.array([int(gt_det[1]) for gt_det in mot_gt[mot_set][frame_id]])

                data_list[mot_set][frame_id] = frame_data

                # print(data_list)
                # exit()

    for mot_set, dset_data in data_list.items():
        data_list[mot_set] = [dset_data[i] for i in sorted(dset_data.keys())]
            
    return data_list

def load_mot_gt(anns_pathes):
    #from mot20 github

    if not anns_pathes.is_file():
        log.warning("NO GT available for this dataset")
        return None
     
    data = []
    with open(anns_pathes, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip().split(",")
        data.append(line)    

    data_dict = defaultdict(list)
    for row in data:
        frame_id = int(row[0])
        data_dict[frame_id].append(row)

    return data_dict

def load_reid_data(file_pathes):
    path_to_map = file_pathes[0]
    path_to_feat = file_pathes[1]

    img_to_fid = load_json(path_to_map)

    with open(path_to_feat, 'rb') as f:
        features = np.load(f)

    return {"img_to_fid": img_to_fid, "features": features}

def append_reid(data_dict, reid_data):
    img_to_fid = reid_data["img_to_fid"]
    # print(img_to_fid.keys())
    # print(data_dict.keys())
    features = reid_data["features"]

    for dset_name in data_dict.keys():
        gt = data_dict[dset_name]

        for i in range(len(gt)):
            img_path = gt[i]["img_path"]
            frame_id = str(int(img_path.split("/")[-1].split(".")[0]))
            feats_id = img_to_fid[dset_name][frame_id]
            gt[i]["reid_features"] = features[feats_id]

    return data_dict

def save_data(data_list, file_path):

    with open(file_path, 'w') as file:
        for _, dset_data in data_list.items():
            for frame_data in dset_data:
                json.dump(frame_data, file)
                file.write("\n")
            
    return data_list

def save_dict_as_json(data_dict, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                                 
    with open(file_path, 'w') as file:
        json.dump(data_dict, file)

    return data_dict

def load_json(file):
    with open(file, 'r') as file:
        data = json.load(file)
        
    return data

def find_nearest_neighbour(frame_id, sorted_index_list):
    #find nearest neighbour in sorted list
    return min(sorted_index_list, key=lambda x:abs(x-frame_id))


def load_calibration(calibration_folder, dset, max_frame_id=None):

    filename = calibration_folder + dset[:8] + "_homography.json"

    with open(filename, 'r') as file:
        data = json.load(file)

    if "-1" in data:
        return np.array(data["-1"]["IPM"])
    else:
        index_s = sorted([int(k) for k in data.keys()])

        if max_frame_id is None:
            max_frame_id = max(index_s)

        # find missing index in range 0 to len(data)
        missing_index = list(set(range(1,max_frame_id+1)) - set(index_s))

        if len(missing_index) > 0:
            log.warning(f"Missing {len(missing_index)} / {max_frame_id} homographies in {dset}: {missing_index}, replacing by nearest neighbour")
            for i in missing_index:
                nearest_frame_index = find_nearest_neighbour(i, index_s)
                log.spam(f"Using homography from frame {nearest_frame_index} for frame {i}")
                index_s.append(nearest_frame_index)
            
            index_s.sort()

        return [np.array(data[str(k)]["IPM"]) for k in index_s]

    return data


def project_points(points, homography):

    poing_hom = np.ones((points.shape[0], points.shape[1] + 1))
    poing_hom[:, :2] = points
    poing_hom = poing_hom.T

    projected_points = homography @ poing_hom
    projected_points = (projected_points[:-1] / projected_points[-1]).T

    return  projected_points
    
def get_ground_point_from_bbox(bboxes, homography):
    
    bbox_bottom = np.vstack([(bboxes[:,0] + bboxes[:,2]) / 2, bboxes[:, 3]]).T
    ground = project_points(bbox_bottom, homography)

    return ground

def get_bbox_center(bboxes):
    bbox_center = np.vstack([(bboxes[:,0] + bboxes[:,2]) / 2, (bboxes[:,1] + bboxes[:,3]) / 2]).T
    
    return bbox_center

def project_motion_to_image(motion_ground, start_motion, homography_image_to_ground):
    motion_ground = np.array(motion_ground)
    start_motion = np.array(start_motion)
    
    end_motion = motion_ground + start_motion

    #project start and end position to image
    start_image = project_points(start_motion, np.linalg.inv(homography_image_to_ground))
    end_image = project_points(end_motion, np.linalg.inv(homography_image_to_ground))

    motion_image = end_image - start_image
        
    return motion_image

def append_homography(data_dict, calib_dir):
    
    for dset_name in data_dict.keys():
        
        gt = data_dict[dset_name]
        max_frame_id = max([int(frame["img_path"].split("/")[-1].split(".")[0]) for frame in gt])
        homographies = load_calibration(calib_dir, dset_name, max_frame_id)
            
        if type(homographies) != list:
            log.debug(f"Static camera in {dset_name}")
            for i in range(len(gt)):
                gt[i]["homography"] = homographies
                gt[i]["bboxes"] = np.array(gt[i]["bboxes"])
        else:
            log.debug(f"Dynamic camera in {dset_name}")
            # homographies = homographies[-len(gt):]
            for i in range(len(gt)):
                frame_id = int(gt[i]["img_path"].split("/")[-1].split(".")[0]) - 1
                gt[i]["homography"] = homographies[frame_id]
                gt[i]["bboxes"] = np.array(gt[i]["bboxes"])
            
    return data_dict


def gausian_center_heatmap(size, points, radius, reid=None):
    
    if reid is not None:
        return gausian_center_heatmap_with_reid(size, points, radius, reid)
    
    points = np.array(points)
    hm = np.zeros(size)

    for point in points:
        if is_in_frame(point, size):
            draw_umich_gaussian(hm, point, radius, k=1)

    hm = torch.from_numpy(hm).to(torch.float32).unsqueeze(0)

    return hm

def gausian_center_heatmap_with_reid(size, points, radius, reid_features):

    points = np.array(points)
    hm = np.zeros([reid_features.shape[1]+1] + list(size))

    for point, reid in zip(points, reid_features):
        draw_umich_gaussian(hm[0,:,:], point, radius, k=1)
        if is_in_frame(point, size):
            hm[1:, int(point[1]), int(point[0])] = reid

    hm = torch.from_numpy(hm).to(torch.float32)

    return hm

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
        
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # import pdb; pdb.set_trace()
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def reject_outliers(data, m = 2., axis=0):
        d = np.abs(data - np.median(data, axis=0))
        mdev = np.median(d, axis=0)
        s = d/mdev if (mdev != 0).all() else np.zeros(len(d))
        
        return data[(s<m).all(axis=1)]

    
def is_in_frame(point, frame_size):
    is_in_top_left = point[0] > 0 and point[1] > 0
    is_in_bottom_right = point[0] < frame_size[0] and point[1] < frame_size[1]
    
    return is_in_top_left and is_in_bottom_right

