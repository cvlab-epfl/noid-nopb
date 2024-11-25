import sys
import argparse

from configs.pathes import conf_path, data_path, model_path
from configs.utils import read_yaml_file, convert_yaml_dict_to_arg_list, fill_dict_with_missing_value, aug_tuple
from configs.utils import args_to_dict
from misc.log_utils import log, set_log_level, dict_to_string


parser = argparse.ArgumentParser(
    description='Multi-view object tracking configuration',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Create argument groups for better organization
parser_train = parser.add_argument_group("training")
parser_data = parser.add_argument_group("dataset")
parser_model = parser.add_argument_group("model")
parser_loss = parser.add_argument_group("loss")


####### General Configuration #######
parser.add_argument("-n", '--name',
    default="model",
    help='Name of the model'
)
parser.add_argument("-pf", "--print_frequency",
    dest="print_frequency",
    type=int,
    default=2500,
    help="Number of elements processed between prints"
)
parser.add_argument("-lf", "--log_frequency",
    dest="log_frequency", 
    type=int,
    default=3,
    help="Number of images to log per epoch"
)
parser.add_argument("-dev", "--device",
    dest="device",
    default="cuda",
    help="Select device to use: cpu or cuda"
)
parser.add_argument("-ch", "--nb_checkpoint",
    dest="nb_checkpoint",
    type=int, 
    default=10,
    help="Maximum number of checkpoints to save before overwriting old ones"
)
parser.add_argument("-l", "--log_lvl",
    dest="log_lvl",
    default="info",
    choices=["debug", "spam", "verbose", "info", "warning", "error"],
    help='Set logging verbosity level'
)
parser.add_argument("-mtl", "--max_tracklet_lenght",
    dest="max_tracklet_lenght",
    type=int,
    default=50,
    help="Maximum length of tracklets"
)
parser.add_argument("-cfg", "--config_file",
    dest="config_file",
    type=str,
    default="./configs/files/config_continuous_test_time_mot17.yaml",
    help="Path to YAML config file with default values (command line args override these)"
)

####### Training Configuration #######
parser_train.add_argument("-lr", "--learning_rate",
    dest="lr",
    type=float,
    default=0.001,
    help="Initial learning rate"
)
parser_train.add_argument("-lrd", "--learning_rate_decrease",
    dest="lrd",
    nargs='+',
    type=float,
    default=[0.5, 20, 40, 60, 80, 100],
    help="List of epochs where learning rate is decreased (multiplied by first arg)"
)
parser_train.add_argument("-dec", "--decay",
    dest="decay",
    type=float,
    default=5e-4,
    help="Adam weight decay"
)
parser_train.add_argument("-vis", '--eval_visual',
    dest="eval_visual",
    action="store_true",
    default=True,
    help="Create video visualizations from evaluation outputs"
)
parser_train.add_argument("-ksp", '--eval_ksp',
    dest="eval_ksp",
    action="store_true",
    default=False,
    help="Post-process tracking using KSP (slower) and generate visualizations"
)
parser_train.add_argument("-fepoch", '--fepoch',
    dest="fepoch",
    type=int,
    default=-1,
    help="Epoch to start finetuning"
)
parser_train.add_argument("-mepoch", '--max_epoch',
    dest="max_epoch",
    type=int,
    default=200,
    help="Maximum number of epochs"
)
parser_train.add_argument("-sstep", '--substep',
    dest="substep",
    type=int,
    default=1,
    help="Number of substeps per epoch"
)
parser_train.add_argument("-dtrack", '--disable_tracker',
    dest="disable_tracker",
    action="store_true",
    default=True,
    help="Disable tracker during evaluation"
)
parser_train.add_argument("-gclip", '--gradient_clip_value',
    dest="gradient_clip_value",
    type=float,
    default=10000000000,
    help="Gradient clipping threshold for training stability"
)
parser_train.add_argument("-dtv", "--detection_to_evaluate",
    dest="detection_to_evaluate",
    nargs='+',
    type=str,
    default=["hm_0"],
    help="List of detections to evaluate at inference"
)
parser_train.add_argument("-mtv", "--motion_to_evaluate",
    dest="motion_to_evaluate",
    nargs='+',
    type=str,
    default=[],
    help="List of motion predictions to evaluate at inference"
)
parser_train.add_argument("-metp", "--metric_to_print",
    dest="metric_to_print",
    nargs='+',
    type=str,
    default=[],
    help="Metrics to log during validation"
)
parser_train.add_argument("-lott", "--loss_to_print",
    dest="loss_to_print",
    nargs='+',
    type=str,
    default=["hm_loss","reg_loss"],
    help="Losses to log during training"
)

####### Dataset Configuration #######
parser_data.add_argument("-nbf", "--nb_frames",
    dest="nb_frames",
    type=int,
    default=2,
    help="Number of frames to process simultaneously from same scene"
)
parser_data.add_argument("-vid", "--view_ids",
    dest="view_ids",
    type=int,
    default=[0],
    nargs='*',
    help="View IDs to process (starting from 0)"
)
parser_data.add_argument("-hms", "--hm_size",
    dest="hm_size",
    nargs="+",
    type=int,
    default=[128, 128],
    help="Size to resize images before processing (width, height)"
)
parser_data.add_argument("-hmr", "--hm_radius",
    dest="hm_radius",
    type=int,
    default=3,
    help="Radius of gaussian filter for heatmap generation"
)
parser_data.add_argument("-shft", "--shuffle_train",
    dest="shuffle_train",
    action="store_false",
    default=True,
    help="Disable training set shuffling"
)
parser_data.add_argument("-bs", "--batch_size",
    dest="batch_size",
    type=int,
    default=4,
    help="Batch size"
)
parser_data.add_argument("-nw", "--num_workers",
    dest="num_workers",
    type=int,
    default=10,
    help="Number of dataloader workers"
)
parser_data.add_argument("-aug", "--aug_train",
    dest="aug_train",
    action="store_true",
    default=False,
    help="Enable training data augmentation"
)
parser_data.add_argument("-fi", "--frame_interval",
    dest="frame_interval",
    type=int,
    nargs='+',
    default=[1, 2],
    help="Frame interval range for sampling video triplets"
)
parser_data.add_argument("-fieval", "--frame_interval_eval",
    dest="frame_interval_eval",
    type=int,
    nargs='+',
    default=[1, 2],
    help="Frame interval range for evaluation triplets"
)
parser_data.add_argument("-dtest", "--data_test",
    dest="data_test",
    action="store_true",
    default=False,
    help="Use test dataset mode (smaller dataset)"
)
parser_data.add_argument("-vaug", "--views_based_aug_list",
    dest="views_based_aug_list",
    type=aug_tuple,
    nargs='+',
    default=[(None, 1)],
    help="View-based augmentations as (type,prob) pairs"
)
parser_data.add_argument("-saug", "--scene_based_aug_list",
    dest="scene_based_aug_list", 
    type=aug_tuple,
    nargs='+',
    default=[(None, 1)],
    help="Scene-based augmentations as (type,prob) pairs"
)
parser_data.add_argument("-pftr", "--pred_file_train",
    dest="pred_file_train",
    type=str,
    default=None,
    help="Path to training predictions file"
)
parser_data.add_argument("-pfva", "--pred_file_val",
    dest="pred_file_val",
    type=str,
    default=None,
    help="Path to validation predictions file"
)
parser_data.add_argument("-pfte", "--pred_file_test",
    dest="pred_file_test",
    type=str,
    default=None,
    help="Path to test predictions file"
)
parser_data.add_argument("-preidfeat", "--path_reid_features",
    dest="path_reid_features",
    type=str,
    default=None,
    help="Path to ReID features file"
)
parser_data.add_argument("-preidmap", "--path_reid_map",
    dest="path_reid_map",
    type=str,
    default=None,
    help="Path to ReID mapping file"
)
parser_data.add_argument("-cdir", "--calibration_dir",
    dest="calibration_dir",
    type=str,
    default="./data/MOT17/calibration_data/",
    help="Directory containing scene calibration files"
)
parser_data.add_argument("-ddir", "--data_dir",
    dest="data_dir",
    type=str,
    default="./data/MOT17/",
    help="Base data directory"
)
parser_data.add_argument("-gt_mot_dir", "--gt_mot_dir",
    dest="gt_mot_dir",
    type=str,
    default="./data/MOT17/train/{mot_set}/gt/gt.txt",
    help="Ground truth detection directory format"
)
parser_data.add_argument("-gr", "--ground",
    dest="ground",
    action="store_true",
    default=False,
    help="Predict motion in ground plane"
)
parser_data.add_argument("-re", "--reid_feature",
    dest="reid_feature",
    action="store_true",
    default=False,
    help="Use ReID features as network input"
)
parser_data.add_argument("-tra", '--train_split',
    dest="train_split",
    action="store_true",
    default=False,
    help="Use training split"
)
parser_data.add_argument("-val", '--val_split',
    dest="val_split",
    action="store_true",
    default=False,
    help="Use validation split"
)
parser_data.add_argument("-tes", '--test_split',
    dest="test_split",
    action="store_true",
    default=False,
    help="Use test split"
)
parser_data.add_argument("-etra", '--eval_train_split',
    dest="eval_train_split",
    action="store_true",
    default=False,
    help="Evaluate on training split"
)
parser_data.add_argument("-eval", '--eval_val_split',
    dest="eval_val_split",
    action="store_true",
    default=False,
    help="Evaluate on validation split"
)
parser_data.add_argument("-etes", '--eval_test_split',
    dest="eval_test_split",
    action="store_true",
    default=False,
    help="Evaluate on test split"
)
parser_data.add_argument("-gt_motion", '--gt_motion',
    dest="gt_motion",
    action="store_true",
    default=False,
    help="Use ground truth motion for evaluation"
)

####### Model Configuration #######
parser_model.add_argument("-bb", "--backbone",
    dest="backbone",
    type=str,
    default="r50",
    choices=["vgg", "r50", "r101", "r152", "r18", "r34"],
    help="Backbone architecture for feature extraction"
)
parser_model.add_argument("-trgr", "--tracker_ground",
    dest="tracker_ground",
    action="store_true",
    default=False,
    help="Use ground truth detections for tracking"
)
parser_model.add_argument("-trint", "--tracker_interval",
    dest="tracker_interval",
    type=int,
    default=1,
    help="Frame interval for tracking"
)

####### Loss Configuration #######
parser_loss.add_argument("-flt", "--flow_loss_type",
    dest="flow_loss_type",
    type=str,
    default="flow",
    choices=["flow", "cont"],
    help="Flow estimation loss type"
)
parser_loss.add_argument("-flc", "--flow_crit",
    dest="flow_crit",
    type=str,
    default="mse",
    choices=["mse", "focal", "bce", "bcemse"],
    help="Flow loss criterion"
)
parser_loss.add_argument("-tlt", "--track_loss_type",
    dest="track_loss_type",
    type=str,
    default="focal",
    choices=["mse", "focal", "flow"],
    help="Tracking loss type"
)
parser_loss.add_argument("-clt", "--consistency_type",
    dest="consistency_type",
    type=str,
    default="mse",
    choices=["mse", "bce", "kl"],
    help="Flow consistency loss type"
)
parser_loss.add_argument("-rf", '--reweigthing_factor',
    dest="reweigthing_factor",
    type=float,
    default=0,
    help="Reweighting factor for non-static predictions"
)
parser_loss.add_argument("-regt", "--regularization_type",
    dest="regularization_type",
    type=str,
    default="all",
    choices=["not_center_only", "not_center_only_rw", "not_gt_hm", "not_gt_hm_rw", "not_gt_hm_rw_smooth", "all"],
    help="Offset regularization type"
)
parser_loss.add_argument("-lbdr", "--lambda_reg",
    dest="lambda_reg",
    type=float,
    default="1e-1",
    help="Offset regularization weight"
)
parser_loss.add_argument("-lbdrd", "--lambda_reg_det",
    dest="lambda_reg_det",
    type=float,
    default="1e-4",
    help="Detection offset regularization weight"
)
parser_loss.add_argument("-lbdrs", "--lambda_smoothing",
    dest="lambda_smoothing",
    type=float,
    default="10",
    help="Offset smoothing weight"
)
parser_loss.add_argument("-lbdcs", "--lambda_consistency",
    dest="lambda_consistency",
    type=float,
    default="0",
    help="Temporal consistency weight"
)
parser_loss.add_argument("-soff", '--supervised_offset',
    dest="supervised_offset",
    action="store_true",
    default=False,
    help="Use ground truth offsets for training"
)
parser_loss.add_argument("-rks", "--rec_ksize",
    dest="rec_ksize",
    type=int,
    default=39,
    help="Reconstruction window size for heatmap generation"
)
parser_loss.add_argument("-recw", "--rec_expe_weight",
    dest="rec_expe_weight",
    type=float,
    default=0.8,
    help="Reconstruction exponential weight"
)
parser_loss.add_argument("-recwi", "--rec_weight_increment",
    dest="rec_weight_increment",
    type=float,
    default=0.08,
    help="Reconstruction weight increment"
)
parser_loss.add_argument("-recwm", "--rec_weight_max",
    dest="rec_weight_max",
    type=float,
    default=5,
    help="Maximum reconstruction weight"
)
parser_loss.add_argument("-lt", "--loss_type",
    dest="loss_type",
    type=str,
    default="bce",
    choices=["focal", "bce", "l1", "l2"],
    help="Heatmap loss criterion"
)
parser_loss.add_argument("-lrt", "--loss_reg_type",
    dest="loss_reg_type",
    type=str,
    default="l2",
    choices=["l1", "l2"],
    help="Regression loss criterion"
)


def get_config_dict(existing_conf_dict=None):
    """Generate configuration dictionary from command line arguments and optional config file.
    
    Args:
        existing_conf_dict: Optional existing config to update with new values
        
    Returns:
        Dictionary containing all configuration parameters
    """
    log.debug(f'Original command {" ".join(sys.argv)}')
    args = parser.parse_args()

    if args.config_file is not None:
        yaml_dict = read_yaml_file(args.config_file)
        arg_list = convert_yaml_dict_to_arg_list(yaml_dict)
        args = parser.parse_args(arg_list + sys.argv[1:])

    args_dict = args_to_dict(parser, args)

    config = {
        "data_conf": {**args_dict["dataset"], **data_path},
        "model_conf": {**args_dict["model"], **model_path},
        "loss_conf": args_dict["loss"],
        "training": {**args_dict["training"], **conf_path},
        "main": vars(args)
    }
    
    set_log_level(config["main"]["log_lvl"])

    if existing_conf_dict is not None:
        config = fill_dict_with_missing_value(existing_conf_dict, config)

    return config


if __name__ == '__main__':
    conf_dict = get_config_dict()
