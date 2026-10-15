# test.py
import os
import csv
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# -------------------- Utils (consistent with train visualization) --------------------
def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img
@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="test"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr)
    gt_gray = mask_to_gray(targets, thr)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png", img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png", gt_gray)
# -------------------- Metrics: Dice / IoU / Sens / Spec / HD95 --------------------
def iou_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    union = np.logical_or(pred_bin, tgt_bin).sum(dtype=np.float64)
    return float((inter + eps) / (union + eps))
def dice_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    s = pred_bin.sum(dtype=np.float64) + tgt_bin.sum(dtype=np.float64)
    return float((2.0 * inter + eps) / (s + eps))
def sens_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    tp = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    fn = np.logical_and(~pred_bin, tgt_bin).sum(dtype=np.float64)
    return float(tp / (tp + fn + eps))
def spec_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    tn = np.logical_and(~pred_bin, ~tgt_bin).sum(dtype=np.float64)
    fp = np.logical_and(pred_bin, ~tgt_bin).sum(dtype=np.float64)
    return float(tn / (tn + fp + eps))
def _binary_boundary(mask_bin: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask_u8)
    for cnt in contours:
        cv2.drawContours(boundary, [cnt], -1, color=1, thickness=1)
    return boundary.astype(np.uint8)
def hd95_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    pred_has = pred_bin.any()
    tgt_has = tgt_bin.any()
    if (not pred_has) and (not tgt_has):
        return 0.0
    if (pred_has and (not tgt_has)) or ((not pred_has) and tgt_has):
        return float("inf")
    Pb = _binary_boundary(pred_bin)
    Tb = _binary_boundary(tgt_bin)
    if Pb.max() == 0:
        Pb = pred_bin.astype(np.uint8)
    if Tb.max() == 0:
        Tb = tgt_bin.astype(np.uint8)
    def dist_to_border(border01: np.ndarray) -> np.ndarray:
        inv = np.where(border01 > 0, 0, 1).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        return dist
    dist_to_T = dist_to_border(Tb)
    dist_to_P = dist_to_border(Pb)
    Py, Px = np.where(Pb > 0)
    Ty, Tx = np.where(Tb > 0)
    if len(Py) == 0 or len(Ty) == 0:
        return float("inf")
    d_PT = dist_to_T[Py, Px]
    d_TP = dist_to_P[Ty, Tx]
    d_all = np.concatenate([d_PT, d_TP], axis=0)
    if d_all.size == 0:
        return float("inf")
    return float(np.percentile(d_all, 95))
# -------------------- Main Test --------------------
@torch.no_grad()
def run_test(model, loader, device, dice_thr=0.5, vis_dir=None, csv_path=None):
    model.eval()
    os.makedirs(vis_dir, exist_ok=True) if vis_dir else None
    rows = []
    dices, ious, hd95s, senss, specs = [], [], [], [], []
    idx_global = 0
    pbar = tqdm(loader, desc="[Test]")
    for batch in pbar:
        if len(batch) == 3:
            inputs, targets, ids = batch
            case_ids = list(ids)
        else:
            inputs, targets = batch
            case_ids = [f"case_{idx_global + i:05d}" for i in range(inputs.size(0))]
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)

        probs = torch.sigmoid(logits)
        preds = (probs > dice_thr).float()
        B = inputs.size(0)
        for b in range(B):
            gt = targets[b, 0].detach().cpu().numpy() > 0.5
            pr = preds[b, 0].detach().cpu().numpy() > 0.5
            dsc = dice_binary_numpy(pr, gt)
            iou = iou_binary_numpy(pr, gt)
            sens = sens_binary_numpy(pr, gt)
            spec = spec_binary_numpy(pr, gt)
            hd95 = hd95_binary_numpy(pr, gt)
            dices.append(dsc)
            ious.append(iou)
            hd95s.append(hd95)
            senss.append(sens)
            specs.append(spec)
            rows.append({"id": case_ids[b], "dice": dsc, "iou": iou, "sens": sens, "spec": spec, "hd95": hd95})
            if vis_dir is not None:
                save_eval_visuals(idx_global, inputs[b], logits[b], targets[b], vis_dir, thr=dice_thr,
                                  fname_prefix="test")
            idx_global += 1
        pbar.set_postfix(dice=np.mean(dices) if dices else 0.0,
                         iou=np.mean(ious) if ious else 0.0,
                         sens=np.mean(senss) if senss else 0.0,
                         spec=np.mean(specs) if specs else 0.0)
    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_sens = float(np.mean(senss)) if senss else 0.0
    mean_spec = float(np.mean(specs)) if specs else 0.0
    finite_hd = [x for x in hd95s if np.isfinite(x)]
    mean_hd95 = float(np.mean(finite_hd)) if len(finite_hd) > 0 else float("inf")
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "dice", "iou", "sens", "spec", "hd95"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            writer.writerow({"id": "MEAN", "dice": mean_dice, "iou": mean_iou, "sens": mean_sens, "spec": mean_spec,
                             "hd95": mean_hd95})
    print("\n" + "=" * 60)
    print("🏆 [Test Summary]")
    print(f"  * Dice        : {mean_dice:.4f}")
    print(f"  * IoU         : {mean_iou:.4f}")
    print(f"  * Sensitivity : {mean_sens:.4f}")
    print(f"  * Specificity : {mean_spec:.4f}")
    print(f"  * HD95        : {mean_hd95 if math.isfinite(mean_hd95) else 'inf':.4f}")
    print("=" * 60 + "\n")
    return mean_dice, mean_iou, mean_sens, mean_spec, mean_hd95
def load_ckpt_flex(model, ckpt_path, map_location="cpu"):
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)
def main():
    import argparse
    import os
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="isic")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--dice_thr", type=float, default=0.5)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="./runs")
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    args = parser.parse_args()
    save_root = os.path.join(args.save_root, f"test_segdino_{args.dino_size}_{args.input_h}_{args.dataset}")
    vis_dir = os.path.join(save_root, "test_vis")
    csv_path = os.path.join(save_root, "test_metrics.csv")
    os.makedirs(save_root, exist_ok=True)
    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
    from dpt import DPT
    model = DPT(nclass=1, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # 1. 加载权重
    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt_flex(model, args.ckpt, map_location=device)
    from dataset import FolderDataset, ResizeAndNormalize
    test_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    test_dataset = FolderDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    # 3. 运行测试
    run_test(model, loader=test_loader, device=device, dice_thr=args.dice_thr, vis_dir=vis_dir, csv_path=csv_path)
if __name__ == "__main__":
    main()
