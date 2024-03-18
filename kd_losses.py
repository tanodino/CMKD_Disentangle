import torch
import torch.nn as nn
import torch.nn.functional as F

######### STANDARD KD #########
######### Temperature parameter value from "Multi-level Logit Distillation" CVPR 2023 #########
def kd_loss(logits_student, logits_teacher, temperature=4):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd
######################################################################################################

######### DKD "Decoupled knowledge distillation" CVRP 2022 #########
######### Alpha, Beta and Temperature parameter values from "Multi-level Logit Distillation" CVPR 2023 #########
def dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=4.0):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

######################################################################################################

##### MLKD "Multi-level Logit Distillation" CVPR 2023 #########
######## I USED THE PSEUDO CODE ALGORITHM PROPOSED IN THE PAPER ########
def mlkd_loss(logits_student, logits_teacher):
    temperatures = [2.0, 3.0, 4.0, 5.0, 6.0]
    tot_loss = None
    B, C = logits_student.shape
    for t in temperatures:
        p_stu = F.softmax(logits_student / t) # B x C
        p_tea = F.softmax(logits_teacher / t) # B x C
        l_ins = F.kl_div(p_tea, p_stu)
        G_stu = torch.mm(p_stu, p_stu.t()) # B x B
        G_tea = torch.mm(p_tea, p_tea.t()) # B x B
        l_batch = ((G_stu - G_tea) ** 2).sum() / B
        M_stu = torch.mm(p_stu.t(), p_stu) # C x C
        M_tea = torch.mm(p_tea.t(), p_tea) # C x C
        l_class = ((M_stu - M_tea) ** 2).sum() / C
        if tot_loss is None:
            tot_loss = (l_ins + l_batch + l_class)
        else:
            tot_loss += (l_ins + l_batch + l_class)
    return tot_loss