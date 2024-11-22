import torch 
import torch.nn.functional as F

from misc.log_utils import log

def get_coord_grid(x_size, y_size, device=None):
    xs = torch.arange(0, x_size, device=device)
    ys = torch.arange(0, y_size, device=device)
    x, y = torch.meshgrid(xs, ys)
    
    coord_grid = torch.stack([x, y]).permute(2,1,0)
    
    return coord_grid.float()


def reconstruct_from_offset_unfold(hm, offset, ksize, expe_weight=0.5, shift=-10, slope=4):
    
    assert ksize % 2 == 1, "reconstruction windows must be of uneven dimension" 
    
    B, C, H, W = hm.size()

    if offset is not None:
        B_o, H_o, W_o, C_o = offset.size()
    
        assert B == B_o
        assert C_o == 2
        assert H == H_o
        assert W == W_o
    
    #generate base coordinate grid flatten height and widht dimension
    coord_grid = get_coord_grid(W, H, hm.device)#reshape(-1, 2)
    
    coord_grid = coord_grid.repeat(B, 1, 1, 1)

    new_coord = coord_grid.clone()
    
    #compute future coord after applying offset
    if offset is not None:
        updated_coord = coord_grid + offset
    else:
        updated_coord = coord_grid
    
    
    kernel_h, kernel_w = ksize, ksize
    stride = 1
    p2d = (kernel_w//2, kernel_w//2, kernel_h//2, kernel_h//2)

    new_coord_u = new_coord.permute(0,3,1,2).unsqueeze(4).unsqueeze(5)
    updated_coord_u = torch.nn.functional.pad(updated_coord.permute(0,3,1,2), p2d).unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
    
    distance =  -(torch.sqrt(torch.clamp(((new_coord_u - updated_coord_u)**2).sum(dim=1, keepdim=True), min=1e-8))*slope*expe_weight+shift)
    distance = torch.exp(distance)
    distance = distance / (distance + 1)

    hm_u = torch.nn.functional.pad(hm, p2d).unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
  
    rec = (hm_u * distance).sum(dim=(4,5))

    return rec

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
    
class ConsistencywithROILoss(torch.nn.Module):
    def __init__(self, reweighting_factor):
        super(ConsistencywithROILoss, self).__init__()

        self.reweighting_factor = reweighting_factor
        self.loss = torch.nn.MSELoss(reduction="none")

    def forward(self, pred1, pred2, reweighting=None):

        loss = self.loss(torch.clamp(pred1, 1e-10, 1-1e-10), torch.clamp(pred2, 1e-10, 1-1e-10))

        if reweighting is not None:
            loss = loss + self.reweighting_factor*loss*reweighting

        return loss.sum()

class ContinuousFlowLoss(torch.nn.Module):
    def __init__(self, model_spec, data_spec, loss_spec, stats_name="cont"):
        super(ContinuousFlowLoss, self).__init__()

        self.criterion = MSEwithROILoss(loss_spec["reweigthing_factor"])

        self.consitency_criterion = ConsistencywithROILoss(loss_spec["reweigthing_factor"])

        self.stats_name = stats_name
        self.lambda_consistency = loss_spec["lambda_consistency"]

        self.lambda_reg = loss_spec["lambda_reg"] #1e-1
        self.lambda_reg_det = loss_spec["lambda_reg_det"] #1e-4
        self.lambda_reg_smoothing = loss_spec["lambda_smoothing"] #1
        self.gt_radius = data_spec["hm_radius"] // 2
        
        self.rec_ksize = loss_spec["rec_ksize"]
        self.expe_weight = loss_spec["rec_expe_weight"]


    def compute_flow_consistency_loss(self, offset_forw, offset_back, gt_start, gt_end, reweighting=None):
        
        #We compute consistency loss both direction
        time_consistency_loss = self.compute_oneway_vec_consistency_loss(offset_forw, offset_back, gt_start)
        time_consistency_loss += self.compute_oneway_vec_consistency_loss(offset_back, offset_forw, gt_end)

        return time_consistency_loss / 2


    def compute_oneway_vec_consistency_loss(self, offset_forw, offset_back, gt_start, reweighting=None):

        B, C, H, W = offset_forw.size()
        B_o, C_o, H_o, W_o = offset_back.size()

        assert B == B_o
        assert C_o == 2 == C
        assert H == H_o
        assert W == W_o

        grid = get_coord_grid(W, H, offset_forw.device).permute(2,0,1).long() #+ offset
        grid = (grid.repeat(B, 1, 1, 1) + offset_forw) #* (offset > 0.1)
        grid = torch.round(grid).long()
        
        grid[:,0] = torch.clamp(grid[:,0], min=0, max=W-1)
        grid[:,1] = torch.clamp(grid[:,1], min=0, max=H-1)

        #Flip coordinate dimension from (x,y) to (y,x) to be able to index tensor
        grid = torch.flip(grid, [1])
        
        offset_back_real = torch.zeros_like(offset_back)
        
        for b in range(B):
            offset_back_real[b,0] = offset_back[b,0,grid[b,0],grid[b,1]]
            offset_back_real[b,1] = offset_back[b,1,grid[b,0],grid[b,1]]

        mask = (offset_forw.abs() > 0.5).int().detach()
        offset_back_real = offset_back_real * mask

        reg_loss = self.criterion(offset_forw, -offset_back_real)

        return reg_loss


    def compute_offset_smoothing(self, offset, gt_det, gt_det_len):
        B, C, H, W = offset.size()

        smoothness_term = torch.tensor([0.], device=offset.device, dtype=offset.dtype)
        nb_det = sum(gt_det_len)
        for b in range(B):
            for point in gt_det[b][:gt_det_len[b]]:
                    x, y = int(point[0]), int(point[1])

                    height, width = offset.shape[-2:]
                        
                    left, right = min(x, self.gt_radius), min(width - x, self.gt_radius + 1)
                    top, bottom = min(y, self.gt_radius), min(height - y, self.gt_radius + 1)
                    # import pdb; pdb.set_trace()
                    offset_det  = offset[b,:,y - top:y + bottom, x - left:x + right]

                    #minimize std deviation of x and y offset
                    std = torch.std(offset_det, dim=(1,2)).abs().sum()
                    # print(std)
                    if not torch.isnan(std):
                        smoothness_term += std / nb_det         
            
        #divide accumulated smoothnest term by total number of gt
        return smoothness_term * self.lambda_reg_smoothing
    

    def compute_offset_reg(self, offset, hm_gt):
        """
        Three type of offset regularization, where regularization is applied (1) everywhere, (2) everywhere except at detection coordinate (3) everywhere except at detection hm guassian.
        """

        hm_mask = (hm_gt == 0).int()

        offset_reg = (offset * hm_mask).abs().sum()
        

        return offset_reg *  self.lambda_reg


    def forward(self, input_data, output_flow):

        hm_0 = input_data["hm_0"][:,0:1]
        hm_1 = input_data["hm_1"][:,0:1]

        pre_reweighting = torch.abs(hm_0 - hm_1)
        
        stats = {}
        
        rec_1f = reconstruct_from_offset_unfold(hm_0, output_flow["offset_0_1f"].permute(0,2,3,1), self.rec_ksize, expe_weight=self.expe_weight)
        rec_0b = reconstruct_from_offset_unfold(hm_1, output_flow["offset_1_0b"].permute(0,2,3,1), self.rec_ksize, expe_weight=self.expe_weight)

        
        with torch.no_grad():
            hm_0 = reconstruct_from_offset_unfold(hm_0, None, self.rec_ksize , expe_weight=self.expe_weight) 
            hm_1 = reconstruct_from_offset_unfold(hm_1, None, self.rec_ksize , expe_weight=self.expe_weight) 
        
        rec_weight_norm = torch.clamp(hm_0.max(), min=1)

        loss_rec_1f = self.criterion(rec_1f/rec_weight_norm, hm_1/rec_weight_norm, reweighting=torch.abs(hm_0 - hm_1)/rec_weight_norm) 
        loss_rec_0b = self.criterion(rec_0b/rec_weight_norm, hm_0/rec_weight_norm, reweighting=torch.abs(hm_0 - hm_1)/rec_weight_norm) 
        
        loss_rec = (loss_rec_1f + loss_rec_0b) / 2

        stats = {**stats,
            "loss_" + self.stats_name + "_rec_1f" : loss_rec_1f.item() / 2,
            "loss_" + self.stats_name + "_rec_0b" : loss_rec_0b.item() / 2,
            "loss_" + self.stats_name + "_rec" : loss_rec.item(),
            }
        

        if self.lambda_consistency > 0:
            loss_prev_flow_consistency = self.lambda_consistency * self.compute_flow_consistency_loss(output_flow["offset_0_1f"], output_flow["offset_1_0b"], input_data["hm_0"][:,0:1], input_data["hm_1"][:,0:1], reweighting=pre_reweighting)

            time_consistency_loss = loss_prev_flow_consistency

            stats = {**stats,
                "loss_" + self.stats_name + "_prev_time_consistency" : loss_prev_flow_consistency.item() / 2,
                "loss_" + self.stats_name + "_time_consistency" : time_consistency_loss.item(),
                }
        else:
            time_consistency_loss = 0

            stats = {**stats,
                "loss_" + self.stats_name + "_prev_time_consistency" : 0,
                "loss_" + self.stats_name + "_time_consistency" : 0
                }

        reg_offset = self.compute_offset_reg(output_flow["offset_0_1f"], input_data["hm_0"][:,0:1]) \
            + self.compute_offset_reg(output_flow["offset_1_0b"], input_data["hm_1"][:,0:1]) \
            
        stats = {**stats,
            "loss_"+self.stats_name + "_reg_offset" : reg_offset.item()
            }

        if self.lambda_reg_smoothing == 0:
            smoothing_offset = 0

            stats = {**stats,
                "loss_"+self.stats_name + "_smoothing_offset" : 0
            }
        else:
            smoothing_offset = self.compute_offset_smoothing(output_flow["offset_0_1f"], input_data["points_0"], input_data["points_len_0"])  \
                + self.compute_offset_smoothing(output_flow["offset_1_0b"], input_data["points_1"], input_data["points_len_1"])  \
                
            stats = {**stats,
                "loss_"+self.stats_name + "_smoothing_offset" : smoothing_offset.item()
            }

        total_loss =  loss_rec + time_consistency_loss + reg_offset + smoothing_offset

        stats = {**stats,
            "loss_"+self.stats_name : total_loss.item(),
            "loss" : total_loss.item(),
            }

        return {"loss":total_loss, "stats":stats}