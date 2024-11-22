from torch.utils.data import DataLoader, Subset, ConcatDataset


from configs.pathes import conf_path
from misc.log_utils import log, dict_to_string
from dataset.pred_dataset import PredDataset

def get_dataset(data_arg, train, val, test, eval):

    pred_files = []

    if train:
        pred_files.append(data_arg["pred_file_train"])
    if val:
        pred_files.append(data_arg["pred_file_val"])
    if test:
        pred_files.append(data_arg["pred_file_test"])

    if len(pred_files) == 0:
        log.error("No pred_files found")
        return None

    if eval:
        frame_interval = data_arg["frame_interval_eval"]
    else:
        frame_interval = data_arg["frame_interval"]

    dataset = PredDataset(
        pred_files, 
        data_arg["calibration_dir"], 
        data_arg["data_dir"],
        data_arg["hm_size"], 
        nb_frames=data_arg["nb_frames"], 
        frame_interval=frame_interval, 
        hm_radius=data_arg["hm_radius"], 
        ground=data_arg["ground"], 
        reid=data_arg["reid_feature"],
        reid_files=(data_arg["path_reid_map"], data_arg["path_reid_features"]), 
        use_gt=data_arg["gt_motion"] and not eval,
        mot_gt_dir=data_arg["gt_mot_dir"]
        )
    
    log.debug(f"Dataset: {dataset}, len: {len(dataset)}")

    return dataset


def get_dataloader(data_arg, train=True, val=False, test=False, eval=False):
    log.info(f"Building Datasets")
    log.debug(f"Data spec: {dict_to_string(data_arg)}")

    dataset = get_dataset(data_arg, train, val, test, eval)

    batch_size = data_arg["batch_size"] if not(eval) else 1
    shuffle = data_arg["shuffle_train"] if not(eval) else False

    train_dataloaders = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=data_arg["num_workers"]
        )
      
    return train_dataloaders