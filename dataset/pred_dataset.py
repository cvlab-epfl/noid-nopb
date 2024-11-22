
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils import load_data, append_homography, get_ground_point_from_bbox, get_bbox_center, reject_outliers, gausian_center_heatmap, load_reid_data, append_reid, is_in_frame
from misc.utils import listdict_to_dictlist, stack_tensors, PinnableDict

from torch.nn.utils.rnn import pad_sequence

class PredDataset:
    def __init__(self, pred_files, calibration_dir, data_dir, hm_size, nb_frames=2, frame_interval=(1,2), hm_radius=3, ground=True, reid=False, reid_files=None, use_gt=False, mot_gt_dir=None):
        self.img_root = data_dir
        
        data_pred = load_data(pred_files, use_gt=use_gt, mot_gt_dir=mot_gt_dir)

        if ground:
            self.data_pred = append_homography(data_pred, calibration_dir)        
        else:
            self.data_pred = data_pred
            for dset_name in self.data_pred.keys():
                gt = self.data_pred[dset_name]
                for i in range(len(gt)):
                    gt[i]["bboxes"] = np.array(gt[i]["bboxes"])

        if reid:
            reid_data = load_reid_data(reid_files)
            self.data_pred = append_reid(self.data_pred, reid_data)
        
        self.hm_builder = gausian_center_heatmap
        self.hm_size_yx = np.array(hm_size)
        self.hm_size_xy = np.array([hm_size[1], hm_size[0]])
        
        self.nb_frames = nb_frames
        self.frame_interval = frame_interval
        self.hm_radius = hm_radius
        self.ground = ground

        self.reid = reid

        self.use_gt = use_gt
        
        # For each sequence we omit a number of last frame to be guarantee to always be able to sample nb_frames.
        self.end_sequence_omit = (self.nb_frames-1)*(max(self.frame_interval)-1)
        
        self.idx_to_set = self.build_idx_to_set()
        
        self.min_val, self.max_val = self.get_normalization_factor()
        
    def build_idx_to_set(self):
        idx_to_set = dict()
        
        idx = 0
        for dset_name in self.data_pred.keys():
            dset_frames = self.data_pred[dset_name][:-self.end_sequence_omit]
            
            for i in range(len(dset_frames)):
                idx_to_set[idx] = (dset_name, i)
                idx += 1
                
        return idx_to_set
    
        
    def get_normalization_factor(self):
        
        min_val = dict()
        max_val = dict()
        
        for set_name, dset in self.data_pred.items():
            all_dets = []
            
            for frame_pred in dset:
                if self.ground:
                    frame_points = get_ground_point_from_bbox(frame_pred["bboxes"], frame_pred["homography"]) 
                else:
                    frame_points = get_bbox_center(frame_pred["bboxes"])

                all_dets.append(frame_points)

            all_dets = np.concatenate(all_dets, axis=0)
            all_dets = all_dets.astype(np.float32)
            
            all_dets = reject_outliers(all_dets)

            min_val[set_name] = np.min(all_dets, axis=0)
            max_val[set_name] = np.max(all_dets, axis=0)
            
            
        return min_val, max_val
    
    def normalize(self, points, set_name, hm_occupancy=0.95):
        
        # put point in the [0,1] range
        norm_points = (points - self.min_val[set_name]) / (self.max_val[set_name]-self.min_val[set_name])
        
        # Rescale the point to fill hm_size*hm_occupancy (leaving a padding of (1-hm_occupancy) / 2 around the points) 
        norm_points = norm_points * hm_occupancy * self.hm_size_xy + (((1-hm_occupancy) / 2) * self.hm_size_xy)
        
        return norm_points
    
    def unnormalize_motion(self, norm_points, set_name, hm_occupancy=0.95):
            
            # Rescale the point to fill hm_size*hm_occupancy (leaving a padding of (1-hm_occupancy) / 2 around the points) 
            points = norm_points / self.hm_size_xy
            
            # put point in the [0,1] range
            points = points * (self.max_val[set_name]-self.min_val[set_name])
            
            return points
    
    def unnormalize_points(self, norm_points, set_name, hm_occupancy=0.95):
        
        points = (norm_points - (((1-hm_occupancy) / 2) * self.hm_size_xy)) / (hm_occupancy * self.hm_size_xy)
        points = (points * (self.max_val[set_name]-self.min_val[set_name])) + self.min_val[set_name]
        
        return points
    

    def set_value_for_square(self, array, center_point, value, radius, frame_size):
        x, y = center_point[0], center_point[1]
        height, width = frame_size

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        array[:, y - top:y + bottom, x - left:x + right] = value[..., np.newaxis, np.newaxis]
    #     tracking_mask[:, y - top:y + bottom, x - left:x + right] = 1
        
        return array

    def build_tracking_gt(self, person_id, pre_person_id, gt_points, pre_gt_points):


        intersect_id, ind, pre_ind = np.intersect1d(person_id, pre_person_id, return_indices=True)

        #Initialize map and mask
        tracking_map = np.zeros((2, self.hm_size_yx[0], self.hm_size_yx[1]))

        for i in range(intersect_id.shape[0]):
            frame_id = ind[i]
            pre_id = pre_ind[i]
            
            point = pre_gt_points[pre_id].int()

            if is_in_frame(point, self.hm_size_xy):
                tracking_map = self.set_value_for_square(tracking_map, point, gt_points[frame_id] - pre_gt_points[pre_id], self.hm_radius, self.hm_size_yx)

        tracking_map = torch.from_numpy(tracking_map).to(torch.float32)

        return tracking_map
    

    def __getitem__(self, idx):
        multi_frame_data = dict()
        
        set_name, idx = self.idx_to_set[idx]
        frame_interval = random.randrange(*self.frame_interval)
        
        for frame_id in range(self.nb_frames):
            true_frame_id = idx+frame_id*frame_interval
            bboxes = self.data_pred[set_name][true_frame_id]["bboxes"]

            if self.ground:
                homography = self.data_pred[set_name][true_frame_id]["homography"]
                multi_frame_data[f"homography_{frame_id}"] = homography
                points = get_ground_point_from_bbox(bboxes, homography)
            else:
                points = get_bbox_center(bboxes)

            if self.reid:
                reid = self.data_pred[set_name][true_frame_id]["reid_features"]
            else:
                reid = None

            hm_points = self.normalize(points, set_name)
            # hm_points = hm_points.round().astype(int)
            hm = self.hm_builder(self.hm_size_yx, hm_points, self.hm_radius, reid=reid)

            img_path = self.data_pred[set_name][true_frame_id]["img_path"]

            multi_frame_data[f"img_path_{frame_id}"] = img_path
            multi_frame_data[f"points_{frame_id}"] = torch.from_numpy(hm_points)
            multi_frame_data[f"hm_{frame_id}"] = hm

            if self.use_gt:
                person_id = self.data_pred[set_name][true_frame_id]["id"]
                multi_frame_data[f"person_id_{frame_id}"] = torch.from_numpy(person_id)

        if self.use_gt:
            for i in range(self.nb_frames-1):
                multi_frame_data[f"offset_{i}_{i+1}f"] = self.build_tracking_gt(multi_frame_data[f"person_id_{i+1}"], multi_frame_data[f"person_id_{i}"], multi_frame_data[f"points_{i+1}"], multi_frame_data[f"points_{i}"])
                multi_frame_data[f"offset_{i+1}_{i}b"] = self.build_tracking_gt(multi_frame_data[f"person_id_{i}"], multi_frame_data[f"person_id_{i+1}"], multi_frame_data[f"points_{i}"], multi_frame_data[f"points_{i+1}"])
            
        return multi_frame_data
            
            
    def __len__(self):
        total_len = 0
        for k, v in self.data_pred.items():
            total_len += len(v) - self.end_sequence_omit
            
        return total_len
    
    @staticmethod
    def collate_fn(batch):
        #pad gt_arr and keep original size
        batch = listdict_to_dictlist(batch)

        dict_len = dict()                          
        for k, v in batch.items():
            if k.startswith("points_"):
                lengths = [arr.shape[0] for arr in v]
                dict_len[k.replace("points_", "points_len_")] = lengths
                batch[k] = pad_sequence(v, batch_first=True)

            if k.startswith("person_id_"):
                lengths = [arr.shape[0] for arr in v]
                dict_len[k.replace("person_id_", "person_id_len_")] = lengths
                batch[k] = pad_sequence(v, batch_first=True)

        batch.update(dict_len)
        batch = stack_tensors(batch)
        collate_dict = PinnableDict(batch)

        return collate_dict
