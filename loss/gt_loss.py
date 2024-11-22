import torch 
import torch.nn.functional as F

from misc.log_utils import log

class MSEwithROILoss(torch.nn.Module):
    def __init__(self, reweighting_factor):
        super(MSEwithROILoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="none")
        self.reweighting_factor = reweighting_factor

    def forward(self, pred, target, reweighting=None):

        loss = self.mse(pred, target)

        if reweighting is not None and self.reweighting_factor > 0:
            loss = loss + self.reweighting_factor*loss*reweighting

        return loss.sum()
    

class GtMotionLoss(torch.nn.Module):
    def __init__(self, model_spec, data_spec, loss_spec, stats_name="cont"):
        super(GtMotionLoss, self).__init__()

        self.criterion = MSEwithROILoss(loss_spec["reweigthing_factor"])

       
        self.stats_name = stats_name
        
        self.rec_ksize = loss_spec["rec_ksize"]
        self.expe_weight = loss_spec["rec_expe_weight"]


    def forward(self, input_data, output_flow):
        loss_offset_0_1_f = self.criterion(output_flow["offset_0_1f"], input_data["offset_0_1f"]) 
        loss_offset_1_0_b = self.criterion(output_flow["offset_1_0b"], input_data["offset_1_0b"]) 

        total_loss =  loss_offset_0_1_f + loss_offset_1_0_b

        stats = {
            "loss_"+self.stats_name : total_loss.item(),
            "loss_offset_0_1_f" : loss_offset_0_1_f.item(),
            "loss_offset_1_0_b" : loss_offset_1_0_b.item(),
            "loss" : total_loss.item(),
            }

        return {"loss":total_loss, "stats":stats}