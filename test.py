import argparse
import sys
import time

import torch

from dataset import factory as data_factory
from dataset.utils import project_motion_to_image, save_dict_as_json, is_in_frame
from loss import factory as loss_factory
from misc.log_utils import DictMeter, batch_logging, log, dict_to_string

from model import factory as model_factory
from tracker.tracker import run_tracker
from misc.metrics import compute_tracking_metric

def extract_motion(points, motion_flow, radius=2):
    motion = torch.zeros_like(points)

    for point_idx, point in enumerate(points):
        x, y = int(point[0]), int(point[1])

        if is_in_frame(point, motion_flow.shape[-2:]):
            height, width = motion_flow.shape[-2:]
                
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)
            # import pdb; pdb.set_trace()
            offset_det  = motion_flow[:,y - top:y + bottom, x - left:x + right]

            #minimize std deviation of x and y offset
            mean_motion = torch.mean(offset_det, dim=(1,2))

            if torch.isnan(mean_motion).any():
                mean_motion = torch.zeros_like(mean_motion)

            motion[point_idx] = mean_motion
        else:
            log.spam(f"point {point} not in frame")
            motion[point_idx] = torch.zeros_like(motion[point_idx])

    return motion

def evaluation(train_loader, model, criterion, epoch, conf):
    stats_meter  = DictMeter()
    model.eval()
    criterion.eval()
    
    total_nb_frames = len(train_loader)

    all_pred_motion = dict()
    end = time.time()
    if "disable_motion" in conf["data_conf"] and conf["data_conf"]["disable_motion"]:
        det_with_motion = train_loader.dataset.data_pred
    else:
        for i, input_data in enumerate(train_loader):
            input_data = input_data.to(conf["device"])
            data_time = time.time() - end
            
            with torch.no_grad():
                output_data = model(input_data)
                model_time = time.time() - end - data_time
                
                end2 = time.time()
                if conf["data_conf"]["gt_motion"]:
                    criterion_output = {"stats":{}}
                else:
                    criterion_output = {"stats":{}}

                criterion_time = time.time() - end2 

                motion_pred = extract_motion(input_data["points_1"][0], output_data["offset_1_0b"][0])

                dset_name = input_data["img_path_1"][0].split("/")[-3]
                motion_pred = train_loader.dataset.unnormalize_motion(motion_pred.cpu().numpy(), dset_name).tolist()

                if conf["data_conf"]["ground"]:
                    point_unnorm = train_loader.dataset.unnormalize_points(input_data["points_1"][0].cpu().numpy(), dset_name)
                    motion_pred = project_motion_to_image(motion_pred, point_unnorm, input_data["homography_1"][0])

                all_pred_motion[input_data["img_path_1"][0]] = motion_pred

            batch_time = time.time() - end

            epoch_stats_dict = {**criterion_output["stats"], **output_data["time_stats"], "batch_time":batch_time, "data_time":data_time, "model_time":model_time, "criterion_time":criterion_time, "optim_time":0}
            stats_meter.update(epoch_stats_dict)

            if i % conf["main"]["print_frequency"] == 0 or i == (total_nb_frames - 1):
                batch_logging(epoch, i, total_nb_frames, stats_meter, loss_to_print=conf["training"]["loss_to_print"], metric_to_print=conf["training"]["metric_to_print"])

            del input_data
            del output_data
            
            end = time.time()

        det_with_motion = merge_motion_with_det(all_pred_motion, train_loader.dataset.data_pred) 

    use_kalman_filter = conf["data_conf"]["kalman_filter"] if "kalman_filter" in conf["data_conf"] else False
    use_motion = not(conf["data_conf"]["disable_motion"]) if "disable_motion" in conf["data_conf"] else True

    mot_name = conf["data_conf"]["gt_mot_dir"].split("/")[-5]
    assert mot_name.startswith("MOT")

    path_tracker_pred = run_tracker(conf["training"]["ROOT_PATH"], conf["main"]["name"], epoch, det_with_motion, conf["model_conf"]["tracker_ground"], conf["data_conf"]["ground"], conf["model_conf"]["tracker_interval"], conf["data_conf"]["frame_interval_eval"], conf["data_conf"]["calibration_dir"], conf["data_conf"]["gt_mot_dir"], use_motion=use_motion, use_kalman_filter=use_kalman_filter)
    metric_res, metric_msg  = compute_tracking_metric(path_tracker_pred, mot_name)

    metric = {
        "HOTA":metric_res['MotChallenge2DBox']['ByteTrackerMotion']['COMBINED_SEQ']["pedestrian"]["HOTA"]["HOTA"].mean(),
        "MOTA":metric_res['MotChallenge2DBox']['ByteTrackerMotion']['COMBINED_SEQ']["pedestrian"]["CLEAR"]["MOTA"],
        "MOTP":metric_res['MotChallenge2DBox']['ByteTrackerMotion']['COMBINED_SEQ']["pedestrian"]["CLEAR"]["MOTP"],
        "IDF1":metric_res['MotChallenge2DBox']['ByteTrackerMotion']['COMBINED_SEQ']["pedestrian"]["Identity"]["IDF1"]
    }

    return {"stats": {**stats_meter.avg(), **metric} }


def merge_motion_with_det(motions, detection_data):

    #iterate over detection data if img_path match add motion
    for dset_name, dset in detection_data.items():
        for frame in dset:
            frame_path = frame["img_path"]
            if frame_path in motions:
                assert len(frame["bboxes"]) == len(motions[frame_path])
                frame["motion"] = motions[frame_path]                
            else:
                log.debug(f"no motion for {frame['img_path']}")
                frame["motion"] = [[0,0] for _ in range(len(frame["bboxes"]))]

    return detection_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    ####### Configuration #######
    parser.add_argument("checkpoint_path", help='path to the checkpoint to evaluate')
    parser.add_argument("-dev", "--device", dest="device", help="select device to use either cpu or cuda", default="cuda")
    parser.add_argument("-bs", '--batch_size', dest="batch_size", type=int, default=1,  help="The size of the batches")
    parser.add_argument("-tra", '--train_split', dest="train_split", default=False, action="store_true",  help="use training split to evaluate")
    parser.add_argument("-dval", '--disable_val_split', dest="val_split", default=True, action="store_false",  help="use validation split to evaluate")
    parser.add_argument("-tes", '--test_split', dest="test_split", default=False, action="store_true",  help="use test split to evaluate")
    #add arg for use motion and use kalman filter
    parser.add_argument("-dmo", '--disable_motion', dest="disable_motion", default=False, action="store_true",  help="disable motion")
    parser.add_argument("-kf", '--kalman_filter', dest="kalman_filter", default=False, action="store_true",  help="use kalman filter to evaluate")
    parser.add_argument("-mfile", '--motion_file', dest="motion_file", type=str, default="./motion_file.json")
    parser.add_argument("-tground", '--track_on_ground', dest="track_on_ground", default=False, action="store_true",  help="track on ground")
    parser.add_argument("-fs", '--frame_skip', dest="frame_skip", type=int, nargs='+', default=[1],  help="The size of the batches")


    args, unknown = parser.parse_known_args()

    checkpoint_dict = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
    config = checkpoint_dict["conf"]
    #remove checkpint path from arg list
    del sys.argv[1]

    config["data_conf"]["batch_size"] = args.batch_size
    config["data_conf"]["shuffle_train"] = False
    config["data_conf"]["disable_motion"] = args.disable_motion
    config["data_conf"]["kalman_filter"] = args.kalman_filter
    config["model_conf"]["tracker_ground"] = args.track_on_ground

    if "gt_motion" not in config["data_conf"]:
        config["data_conf"]["gt_motion"] = False

    if "gt_mot_dir" not in config["data_conf"]:
        config["data_conf"]["gt_mot_dir"] = "/cvlabscratch/cvlab/home/engilber/datasets/MOT17/train/{mot_set}/gt/gt.txt"


    config["main"]["print_frequency"] = 100
    ##################
    ### Initialization
    ##################
    config["device"] = torch.device('cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu') 
    log.info(f"Device: {config['device']}")

    end = time.time()
    log.info("Initializing model ...")
    
    model = model_factory.get_model(config["model_conf"], config["data_conf"])
    # log.debug(model.state_dict())
    model.load_state_dict(checkpoint_dict["state_dict"])
    # log.debug(model.state_dict())
    model.to(config["device"])

    # for param in join_emb.cap_emb.parameters():
    #     param.requires_grad = False

    log.info(f"Model initialized in {time.time() - end} s")

    
    end = time.time()
    log.info("Loading Data ...")
    
    metric_per_skip = dict()
    for frame_skip in args.frame_skip:
        config["data_conf"]["frame_interval_eval"] = [frame_skip, frame_skip+1]
        config["model_conf"]["tracker_interval"] = frame_skip

        log.info(f"Loading Data, train: {args.train_split}, val: {args.val_split}, test: {args.test_split}")
        dataloader = val_dataloader = data_factory.get_dataloader(config["data_conf"], train=args.train_split, val=args.val_split, test=args.test_split, eval=True)

        log.info(f"Data loaded in {time.time() - end} s")

        criterion = loss_factory.get_loss(config["model_conf"], config["data_conf"], config["loss_conf"])

        ##############
        ### Evaluation
        ##############

        end = time.time()
        log.info(f"Beginning validation")
        eval_results = evaluation(dataloader, model, criterion, checkpoint_dict["epoch"], config)
        # merge_motiont_with_det(preds_motion, config["data_conf"]["pred_file"], args.motion_file)

        metric_per_skip[frame_skip] = eval_results["stats"]

        log.info(f"Validation completed in {time.time() - end}s")


    log.info(f"Metric per frame skip: \n {dict_to_string(metric_per_skip)}")
    # save output
    if args.track_on_ground:
        save_path = f"{config['training']['ROOT_PATH']}/logs/metrics/{config['main']['name']}_epoch_{checkpoint_dict['epoch']}_tground_eval.json"
    else:
        save_path = f"{config['training']['ROOT_PATH']}/logs/metrics/{config['main']['name']}_epoch_{checkpoint_dict['epoch']}_eval.json"

    save_dict_as_json(metric_per_skip, save_path)
#python evaluation.py weights/model_425/model_425_epoch_30.pth.tar -dset mot20train1 mot20train2  -motrf -vis
#  python test.py weights/model_24/model_24_epoch_106.pth.tar -mfile motion_file_model_24_epoch106.json