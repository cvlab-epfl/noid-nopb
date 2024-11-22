from loss.loss import ContinuousFlowLoss
from loss.gt_loss import GtMotionLoss

def get_loss(model_conf, data_conf, loss_conf):
    if data_conf["gt_motion"]:
        return GtMotionLoss(model_conf, data_conf, loss_conf)
    else:
        return ContinuousFlowLoss(model_conf, data_conf, loss_conf)