import numpy as np
import torch

import shutil
from pathlib import Path

from tracker.byte_tracker import ByteTrackerMotion
from tracker.utils import save_sequence

from tracker.utils import save_sequence

from dataset.utils import append_homography
from dataset.utils import load_calibration, load_data, load_mot_gt

from misc.log_utils import log

def run_tracker(root_path, model_name, epoch, det_with_motion, track_on_ground, motion_ground, tracker_interval, frame_interval, calibration_dir, gt_mot_dir, use_motion=False, use_kalman_filter=False):
    
    mot_name = gt_mot_dir.split("/")[-5]
    assert mot_name.startswith("MOT")

    log.debug(f"Running tracker with the following parameters:")
    log.debug(f"Tracker frame interval: {tracker_interval}")
    log.debug(f"Prediction frame interval: {frame_interval}")
    log.debug(f"Use motion: {use_motion}")
    log.debug(f"Use kalman filter: {use_kalman_filter}")
    log.debug(f"Track on ground: {track_on_ground}")
    log.debug(f"Motion ground: {motion_ground}")
    log.debug(f"MOT name: {mot_name}")
    
    assert frame_interval[1] == (frame_interval[0] + 1)

    if track_on_ground and not motion_ground:
        det_with_motion = append_homography(det_with_motion, calibration_dir)

    if track_on_ground:
        path_ext = "_ground"
    else:
        path_ext = "_image"

    root_track_eval_path = root_path / Path("TrackEvalDataTemp") / f"{model_name}_epoch_{epoch}_tracker{path_ext}"
    root_track_eval_path.parents[0].mkdir(exist_ok=True)
    root_track_eval_path.mkdir(exist_ok=True)
    #empty the folder of any file or folder
    for item in root_track_eval_path.glob('*'):
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(str(item))


    seqmap = []
    for dset_name in det_with_motion.keys(): # ["MOT17-02-DPM"]:#: 
            
        pred = det_with_motion[dset_name]
        log.info(f"Running tracker on {dset_name}")
        seqmap.append(dset_name)
        track_as_list = []
        track_as_list_pred = []  
        nb_det_per_frame_gt = []

        try:
            gt = load_mot_gt(Path(gt_mot_dir.format(mot_set=dset_name)))
        except Exception as e:
            log.error(f"Error loading gt for {dset_name}")
            log.error(e)
            gt = None


        tracker_pred = []
        tracker =  ByteTrackerMotion(ground_assign=track_on_ground, use_motion=use_motion,  use_kalman_filter=use_kalman_filter)

        tracker_step_dicts = []
        frame_id = 0
        for i, frame_pred in enumerate(pred):
            if i % tracker_interval != 0:
                continue

            if gt is not None:
                gt_index = int(frame_pred["img_path"].split("/")[-1].split(".")[0])
                frame_gt = gt[gt_index]

            step_dict = {}

            step_dict[f"frame_id"] = frame_id

            bboxes = np.array(frame_pred["bboxes"])
            step_dict["labels"] = torch.from_numpy(np.array(frame_pred["labels"]))
            step_dict["scores"] = torch.from_numpy(np.array(frame_pred["scores"]))
            step_dict["bboxes"] = torch.from_numpy(bboxes)
    #         step_dict["hm_0_gt_bboxes"] = np.array(frame_gt["bboxes"])

            step_dict["homography"] = frame_pred["homography"] if track_on_ground else None

            if use_motion:
                step_dict["motion"] =  torch.from_numpy(np.array(frame_pred["motion"], dtype=float))
                # normalize the motion according to interval between frame in model and tracker
                step_dict["motion"] = step_dict["motion"] * (tracker_interval/frame_interval[0])
            else:
                step_dict["motion"] = None

            tracker_step_dicts.append(step_dict)

            if gt is not None:
                for gt_det in frame_gt:
                    track_as_list.append((int(frame_id+1),gt_det))
                   
            track_out = tracker.track(step_dict)
   
            tracker_pred.append(track_out)

            for bbox, person_id, score in zip(track_out["bboxes"], track_out["instances_id"], track_out["scores"]):
                track_as_list_pred.append({'FrameId':frame_id+1, 'Id':int(person_id), 'X':int(bbox[0]), 'Y':int(bbox[1]), 'Width':int(bbox[2]-bbox[0]), 'Height':int(bbox[3]-bbox[1]), 'Score':score})

            frame_id += 1


        #save pred        
        path_pred = root_track_eval_path / f"pred/{mot_name}-seqmaps/ByteTrackerMotion/data/{dset_name}.txt"
        save_sequence(track_as_list_pred, path_pred)

        #save gt
        path_gt = root_track_eval_path / f"gt/{mot_name}-seqmaps/{dset_name}/gt/gt.txt"
        save_sequence(track_as_list, path_gt, gt=True)

        #save sequinfo
        
        path_org_seqinfo = Path(gt_mot_dir.replace("gt/gt.txt", "seqinfo.ini").format(mot_set=dset_name))
        path_seqinfo = root_track_eval_path / f"gt/{mot_name}-seqmaps/{dset_name}/seqinfo.ini"

        path_seqinfo.parent.mkdir(parents=True, exist_ok=True)
        #load original seqinfo
        with open(path_org_seqinfo, 'r') as file:
            data = file.readlines()

        #change seqlength
        new_seq_length = len(pred)
        new_data = []
        for line in data:
            if "seqLength" in line:
                new_data.append(f"seqLength={new_seq_length}\n")
            else:
                new_data.append(line)

        with open(path_seqinfo, 'w') as file:
            file.writelines(new_data)   

    #save seqmap
    path_seqmap = root_track_eval_path / f"gt/seqmaps/{mot_name}-seqmaps.txt"
    path_seqmap.parent.mkdir(parents=True, exist_ok=True)

    with open(path_seqmap, 'w') as f:
        f.write("name\n")
        for item in seqmap:
            f.write("%s\n" % item)

    return root_track_eval_path