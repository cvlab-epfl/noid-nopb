# Data Organization

This directory should contain the elements listed below. The paths to the files can be specified in the [config file](../configs/files/config_continuous_test_time_mot17.yaml).

## Pre-computed Detections
The YOLOx detection files computed using mmdetection:
- `bytrack_mot17_result_halftrain.json` (41MB) - Detection for training split
- `bytrack_mot17_result_halfval.json` (40MB) - Detection for validation split  
- `bytrack_mot17_result_test.json` (149MB) - Detection for test split

For each detection we also provide appearance features extracted using the [osnet](https://github.com/KaiyangZhou/deep-person-reid) model:
- `appearance/crop_features.npy` - Appearance features for each detection
- `appearance/crop_id_annotation_file.json` - Mapping between detections and features

Download link: [YOLOx detections and appearance features](https://drive.google.com/file/d/1hrIM0j8GqzsyuF1xB3VNJgEHSQQyISKF/view?usp=sharing)


## MOT17 Dataset
The MOT17 dataset should be symlinked or copied to the `data/MOT17/` directory containing:
- `train/` - Training sequences
- `test/` - Test sequences
- `calibration_data/` - Camera calibration files

Download from: [MOT17 website](https://motchallenge.net/data/MOT17/)

## Calibration Files
The `MOT17/calibration_data/` directory should contain per-sequence calibration files:
- `MOT17-XX-full_calib.json` - Full calibration parameters
- `MOT17-XX_homography.json` - Homography matrices
- `MOT17-XX_intrinsics.txt` - Camera intrinsic parameters

Where XX ranges from 01-14 for each sequence.

Download from: [QuoVadis calibration](https://github.com/dendorferpatrick/QuoVadis/tree/main/downloads/MOT17)
