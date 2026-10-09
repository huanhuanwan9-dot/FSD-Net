import os
import cv2
import numpy as np
import argparse
import random
import math
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Fusion loss components
# ==========================================
class SegAWBFocalLoss(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-6, gamma=2.0):
        super(SegAWBFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, logits, targets):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            probs_concat = torch.cat([1 - probs, probs], dim=1)
            if targets.dim() == 4:
                targets = targets.squeeze(1)
            targets = targets.long()
            targets_one_hot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
            curr_classes = 2
            probs_flat = probs_concat.permute(0, 2, 3, 1).reshape(-1, 2)
            targets_flat = targets_one_hot.permute(0, 2, 3, 1).reshape(-1, 2)
        else:
            probs_concat = F.softmax(logits, dim=1)
            if targets.dim() == 4:
                targets = targets.squeeze(1)
            targets = targets.long()
            targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            curr_classes = self.num_classes
            probs_flat = probs_concat.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            targets_flat = targets_one_hot.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        N_k = targets_flat.sum(dim=0)
        valid_cls_mask = N_k > 0
        max_N = N_k.max()
        omega_k = torch.log((max_N / (N_k + self.epsilon)) + 1) * valid_cls_mask.float()
        pixel_weights = torch.matmul(targets_flat, omega_k.unsqueeze(1)).squeeze()
        pt = (probs_flat * targets_flat).sum(dim=1)
        focal_factor = (1.0 - pt + self.epsilon) ** self.gamma
        log_probs_flat = torch.log(probs_flat + self.epsilon)
        ce_per_pixel = -torch.sum(targets_flat * log_probs_flat, dim=1)
        focal_ce_per_pixel = focal_factor * ce_per_pixel
        loss_1 = (pixel_weights * focal_ce_per_pixel).sum() / (pixel_weights.sum() + self.epsilon)

        loss_2 = 0.0
        for k in range(curr_classes):
            if N_k[k] > 1:
                mask = targets_flat[:, k] == 1
                class_probs = probs_flat[mask, k]
                mean_p = class_probs.mean()
                std_p = class_probs.std()
                term = omega_k[k] * (std_p / (mean_p + self.epsilon))
                loss_2 += term
        loss_2 = loss_2 / (valid_cls_mask.sum() + self.epsilon)
        return loss_1 + loss_2


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()
            if targets.dim() == 3 and probs.dim() == 4:
                targets = targets.unsqueeze(1)
            intersection = (probs * targets).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()
        else:
            return torch.tensor(0.0).to(logits.device)


class USLossComponent(nn.Module):
    def __init__(self, lambda0=0.3, beta=0.02, eps=1e-6):
        super(USLossComponent, self).__init__()
        self.lambda0 = lambda0
        self.beta = beta
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, logits, targets, epoch):
        if logits.shape[1] == 1:
            p = torch.sigmoid(logits)
            p = torch.clamp(p, self.eps, 1.0 - self.eps)
            if targets.dim() == 3 and p.dim() == 4:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            H_p = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
            delta = torch.abs(p - targets)
            UE_p = delta * H_p
            L_U = self.mse(H_p, UE_p)
            current_lambda = self.lambda0 * math.exp(-self.beta * epoch)
            return current_lambda * L_U
        else:
            return torch.tensor(0.0).to(logits.device)


class EntropyAwareFusionLoss(nn.Module):
    def __init__(self, num_classes=1, lambda0_us=0.3, beta_us=0.02, focal_gamma=2.0, max_epochs=50):
        super(EntropyAwareFusionLoss, self).__init__()
        self.awb_focal_loss = SegAWBFocalLoss(num_classes=num_classes, gamma=focal_gamma)
        self.dice_loss = SoftDiceLoss()
        self.us_loss_comp = USLossComponent(lambda0=lambda0_us, beta=beta_us)
        self.max_epochs = max_epochs
        self.current_weights = {}

    def _get_cosine_schedule(self, current_epoch, start_val, end_val):
        ep = min(current_epoch, self.max_epochs)
        cosine_factor = 0.5 * (1 + math.cos((ep / self.max_epochs) * math.pi))
        return end_val + (start_val - end_val) * cosine_factor

    def forward(self, logits, targets, epoch=1):
        with torch.no_grad():
            if logits.shape[1] == 1:
                p_safe = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
                entropy_map = -(p_safe * torch.log(p_safe) + (1 - p_safe) * torch.log(1 - p_safe))
                mean_entropy = entropy_map.mean().item()
                normalized_entropy = mean_entropy / 0.6931
            else:
                normalized_entropy = 0.0
        w_focal_awb = self._get_cosine_schedule(epoch, start_val=1.0, end_val=0.6)
        w_dice_base = self._get_cosine_schedule(epoch, start_val=0.5, end_val=1.2)
        w_dice = w_dice_base * (1.0 + 0.3 * (1 - normalized_entropy))
        l_awb_focal = self.awb_focal_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        l_us = self.us_loss_comp(logits, targets, epoch)
        self.current_weights = {"w_fawb": w_focal_awb, "w_dice": w_dice, "ent": normalized_entropy}
        return (w_focal_awb * l_awb_focal) + (w_dice * l_dice) + l_us

# ==========================================
# 2. Metrics & Visualization Helpers
# ==========================================
def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_t.detach().cpu().numpy()
    img = img * std + mean
    img = img.clip(0, 1)
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 3 and m.shape[0] > 1:
        m = torch.argmax(m, dim=0).float()
        return (m * 255.0 / m.max()).byte().numpy() if m.max() > 0 else m.byte().numpy()
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img


def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_gray = mask_to_gray(logits[i], thr)
        gt_gray = mask_to_gray(targets[i], thr)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png", img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png", gt_gray)


@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr)
    gt_gray = mask_to_gray(targets, thr)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png", img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png", gt_gray)


def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice


def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)


def sensitivity_specificity_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    TP = (pred * target).sum(dim=(2, 3))
    TN = ((1 - pred) * (1 - target)).sum(dim=(2, 3))
    FP = (pred * (1 - target)).sum(dim=(2, 3))
    FN = ((1 - pred) * target).sum(dim=(2, 3))
    sensitivity = (TP + eps) / (TP + FN + eps)
    specificity = (TN + eps) / (TN + FP + eps)
    return sensitivity.view(-1), specificity.view(-1)

# ==========================================
# 3. Augmentation
# ==========================================
def apply_paper_augmentations(inputs, targets):
    if random.random() > 0.5:
        inputs = torch.flip(inputs, dims=[3])
        targets = torch.flip(targets, dims=[3])
    if random.random() > 0.5:
        inputs = torch.flip(inputs, dims=[2])
        targets = torch.flip(targets, dims=[2])
    if random.random() > 0.5:
        k = random.randint(1, 3)
        inputs = torch.rot90(inputs, k, [2, 3])
        targets = torch.rot90(targets, k, [2, 3])
    if random.random() > 0.8:
        gray = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:3, :, :]
        inputs = gray.repeat(1, 3, 1, 1)
    if random.random() > 0.5:
        shift = torch.empty((inputs.size(0), 3, 1, 1), device=inputs.device).uniform_(-0.05, 0.05)
        inputs = inputs + shift
    return inputs, targets

# ==========================================
# 4. RAGE‑E TTA Wrapper
# ==========================================
class RAGE_TTA_Wrapper(nn.Module):
    def __init__(self, model, feature_dim=128, momentum=0.9):
        super(RAGE_TTA_Wrapper, self).__init__()
        self.model = model
        self.momentum = momentum
        self.register_buffer('proto_fg', torch.zeros(1, feature_dim, 1, 1))
        self.register_buffer('proto_bg', torch.zeros(1, feature_dim, 1, 1))
        self.register_buffer('is_initialized', torch.tensor(0, dtype=torch.bool))
        self.current_features = None
        self._register_hook()

    def _register_hook(self):
        def forward_hook(module, input, output):
            self.current_features = input[0]
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'scratch'):
            self.model.head.scratch.output_conv[0].register_forward_hook(forward_hook)

    @torch.no_grad()
    def forward(self, inputs):
        self.model.eval()
        transforms = [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
            (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
            (lambda x: torch.rot90(x, k=1, dims=[2, 3]), lambda x: torch.rot90(x, k=-1, dims=[2, 3])),
            (lambda x: torch.rot90(x, k=2, dims=[2, 3]), lambda x: torch.rot90(x, k=-2, dims=[2, 3])),
            (lambda x: torch.rot90(x, k=3, dims=[2, 3]), lambda x: torch.rot90(x, k=-3, dims=[2, 3]))
        ]
        probs_list = []
        weights_list = []
        deep_features = None
        for i, (forward_t, inverse_t) in enumerate(transforms):
            aug_input = forward_t(inputs)
            logits = self.model(aug_input)
            if i == 0 and self.current_features is not None:
                deep_features = self.current_features.clone()
            restored_logits = inverse_t(logits)
            p = torch.sigmoid(restored_logits)
            p_safe = torch.clamp(p, 1e-6, 1.0 - 1e-6)
            entropy = - (p_safe * torch.log(p_safe) + (1 - p_safe) * torch.log(1 - p_safe))
            weight = torch.exp(-entropy)
            probs_list.append(p)
            weights_list.append(weight)
        stacked_probs = torch.stack(probs_list, dim=0)
        stacked_weights = torch.stack(weights_list, dim=0)
        P_geo = torch.sum(stacked_probs * stacked_weights, dim=0) / torch.sum(stacked_weights, dim=0)
        P_geo = torch.clamp(P_geo, 1e-5, 1.0 - 1e-5)
        E_spatial = - (P_geo * torch.log(P_geo) + (1 - P_geo) * torch.log(1 - P_geo))
        E_norm = E_spatial / math.log(2.0)
        P_final = P_geo
        if deep_features is not None:
            B, C, H_f, W_f = deep_features.shape
            P_geo_low = F.interpolate(P_geo, size=(H_f, W_f), mode='bilinear', align_corners=False)
            mask_fg = (P_geo_low > 0.95).float()
            mask_bg = (P_geo_low < 0.05).float()
            curr_fg_feat = (deep_features * mask_fg).sum(dim=(2, 3), keepdim=True) / (
                    mask_fg.sum(dim=(2, 3), keepdim=True) + 1e-6)
            curr_bg_feat = (deep_features * mask_bg).sum(dim=(2, 3), keepdim=True) / (
                    mask_bg.sum(dim=(2, 3), keepdim=True) + 1e-6)
            if not self.is_initialized:
                self.proto_fg.data = curr_fg_feat.data
                self.proto_bg.data = curr_bg_feat.data
                self.is_initialized.data = torch.tensor(1, dtype=torch.bool)
            else:
                norm_feat = F.normalize(deep_features, p=2, dim=1)
                norm_proto_fg = F.normalize(self.proto_fg, p=2, dim=1)
                norm_proto_bg = F.normalize(self.proto_bg, p=2, dim=1)
                sim_fg = (norm_feat * norm_proto_fg).sum(dim=1, keepdim=True)
                sim_bg = (norm_feat * norm_proto_bg).sum(dim=1, keepdim=True)
                P_prior_low = torch.exp(sim_fg) / (torch.exp(sim_fg) + torch.exp(sim_bg) + 1e-6)
                P_prior = F.interpolate(P_prior_low, size=P_geo.shape[-2:], mode='bilinear', align_corners=False)
                alpha = E_norm * 0.5
                P_final = (1 - alpha) * P_geo + alpha * P_prior
                if mask_fg.sum() > 10:
                    self.proto_fg.data = self.momentum * self.proto_fg + (1 - self.momentum) * curr_fg_feat
                if mask_bg.sum() > 10:
                    self.proto_bg.data = self.momentum * self.proto_bg + (1 - self.momentum) * curr_bg_feat
        P_final = torch.clamp(P_final, 1e-5, 1.0 - 1e-5)
        final_logits = torch.log(P_final / (1.0 - P_final))
        return final_logits

# ==========================================
# 5. Train & Evaluation Loop
# ==========================================
def train_one_epoch(model, train_loader, optimizer, device, criterion, scheduler=None, dice_thr=0.5, vis_dir=None,
                    epoch=1):
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    sens_scores = []
    spec_scores = []
    first_batch_logged = False
    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs, targets = apply_paper_augmentations(inputs, targets)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets, epoch)
        if not loss.requires_grad:
            raise RuntimeError("Loss does not have grad, check model parameter freeze status.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            sens, spec = sensitivity_specificity_binary_torch(logits, targets, thresh=dice_thr)
            dice_scores.append(dice)
            iou_scores.append(iou)
            sens_scores.append(sens.mean().item())
            spec_scores.append(spec.mean().item())
        cw = criterion.current_weights
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", dice=f"{dice:.4f}",
            sen=f"{sens.mean().item():.4f}", spe=f"{spec.mean().item():.4f}",
            wF=f"{cw.get('w_fawb', 0):.2f}", wD=f"{cw.get('w_dice', 0):.2f}"
        )
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            first_batch_logged = True
    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0
    avg_sens = float(np.mean(sens_scores)) if len(sens_scores) > 0 else 0.0
    avg_spec = float(np.mean(spec_scores)) if len(spec_scores) > 0 else 0.0
    print(f"[Train Epoch {epoch}] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}  sens={avg_sens:.4f}  spec={avg_spec:.4f}")
    return avg_loss, avg_dice, avg_iou, avg_sens, avg_spec


@torch.no_grad()
def evaluate(model, tta_wrapper, val_loader, device, criterion, dice_thr=0.5, vis_dir=None, epoch=1, use_tta=True):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    sens_scores = []
    spec_scores = []
    idx_global = 0
    pbar = tqdm(val_loader, desc=f"[Eval e{epoch}]")
    for (inputs, targets, _) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if use_tta:
            logits = tta_wrapper(inputs)
        else:
            logits = model(inputs)
        eval_epoch = criterion.max_epochs if hasattr(criterion, 'max_epochs') else epoch
        loss = criterion(logits, targets, eval_epoch)
        total_loss += loss.item()
        dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        iou = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        sens, spec = sensitivity_specificity_binary_torch(logits, targets, thresh=dice_thr)
        dice_scores.append(dice)
        iou_scores.append(iou)
        sens_scores.append(sens.mean().item())
        spec_scores.append(spec.mean().item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}",
                         sen=f"{sens.mean().item():.4f}", spe=f"{spec.mean().item():.4f}")
        if vis_dir is not None:
            if idx_global < 20:
                save_eval_visuals(idx_global, inputs[0], logits[0], targets[0], out_dir=vis_dir, thr=dice_thr,
                                  fname_prefix=f"val_ep{epoch:03d}")
            idx_global += 1
    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0
    avg_sens = float(np.mean(sens_scores)) if len(sens_scores) > 0 else 0.0
    avg_spec = float(np.mean(spec_scores)) if len(spec_scores) > 0 else 0.0
    print(f"[Eval Epoch {epoch}] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}  sens={avg_sens:.4f}  spec={avg_spec:.4f}")
    return avg_loss, avg_dice, avg_iou, avg_sens, avg_spec

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=None)
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"])
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_tta", action="store_true", default=True, help="Use RAG‑E TTA")
    args = parser.parse_args()
    if args.img_size is not None:
        args.input_h = args.img_size
        args.input_w = args.img_size
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    save_root = f"./runs/segdino_RAGETTA_{args.dino_size}_{args.input_h}_{args.dataset}"
    os.makedirs(save_root, exist_ok=True)
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir = os.path.join(save_root, "val_vis")
    ckpt_dir = os.path.join(save_root, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Loading DINOv3 backbone ({args.dino_size}) from {args.repo_dir}...")
    try:
        if args.dino_size == "b":
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
        else:
            backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
        from fsdi import FSD_Net
        model = FSD_Net(nclass=args.num_classes, backbone=backbone)
    except Exception as e:
        print("Model loading warning, please ensure DINO and fsdi.py exist:", e)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    has_trainable_params = any(p.requires_grad for p in model.parameters())
    if not has_trainable_params:
        print("Detect all params frozen, set requires_grad=True.")
        for p in model.parameters():
            p.requires_grad = True
    torch.set_grad_enabled(True)
    tta_wrapper = RAGE_TTA_Wrapper(model, feature_dim=128, momentum=0.9).to(device)
    backbone_params_ids = list(map(id, backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params_ids and p.requires_grad, model.parameters())
    backbone_params = filter(lambda p: id(p) in backbone_params_ids and p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    criterion = EntropyAwareFusionLoss(
        num_classes=args.num_classes,
        lambda0_us=0.3,
        beta_us=0.02,
        focal_gamma=args.focal_gamma,
        max_epochs=args.epochs
    ).to(device)
    print("Initializing Data Loaders...")
    try:
        from dataset import FolderDataset, ResizeAndNormalize
    except ImportError:
        print("Dataset script not found, returning.")
        return
    root = os.path.join(args.data_dir, args.dataset)
    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    val_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    train_dataset = FolderDataset(
        root=root, split="train", img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name, mask_ext=args.mask_ext, transform=train_transform,
    )
    val_dataset = FolderDataset(
        root=root, split="test", img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name, mask_ext=args.mask_ext, transform=val_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )
    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou = -1.0
    best_val_sens = -1.0
    best_val_spec = -1.0
    print(f"Start Training with RAG‑E TTA. Epochs: {args.epochs}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou, train_sens, train_spec = train_one_epoch(
            model, train_loader, optimizer, device, criterion,
            scheduler=scheduler, dice_thr=0.5,
            vis_dir=train_vis_dir, epoch=epoch
        )
        val_loss, val_dice, val_iou, val_sens, val_spec = evaluate(
            model, tta_wrapper, val_loader, device, criterion,
            dice_thr=0.5, vis_dir=val_vis_dir, epoch=epoch, use_tta=args.use_tta
        )
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            latest_path
        )
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_dice_epoch = epoch
            best_val_iou = val_iou
            best_val_sens = val_sens
            best_val_spec = val_spec
            best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice:.4f}_{val_iou:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best ckpt saved: {best_path}")
    print("=" * 60)
    print(f"Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"Best Val IoU  = {best_val_iou:.4f}")
    print(f"Best Val Sens = {best_val_sens:.4f}")
    print(f"Best Val Spec = {best_val_spec:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
