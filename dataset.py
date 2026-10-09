# #PH2
#
# import os
# from typing import List, Tuple, Optional
# import cv2
# import torch
# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms.functional as TF
# from torchvision.transforms import InterpolationMode
#
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)
#
# class ResizeAndNormalize:
#     def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
#         self.size = size  # (H, W)
#         self.mean = mean
#         self.std = std
#         self.thr = thr
#     def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(img_rgb)
#         img_resized = TF.resize(
#             img_pil, self.size,
#             interpolation=InterpolationMode.BICUBIC, antialias=True
#         )
#         img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
#         img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
#         H, W = self.size
#         mask_ts = []
#         for c in range(mask_hwc.shape[2]):
#             m = Image.fromarray(mask_hwc[..., c])
#             m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
#             mt = TF.to_tensor(m)[0]        # Take the single channel
#             mask_ts.append(mt)
#         mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
#         mask_t = (mask_t > self.thr).float()           # Binarize mask by threshold
#         return img_t, mask_t
#
# SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#
# def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
#     out = []
#     for root, _, files in os.walk(dir_path):
#         for f in files:
#             if os.path.splitext(f)[1].lower() in exts:
#                 out.append(os.path.join(root, f))
#     out.sort()
#     return out
#
# def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
#     parts = path.replace("\\", "/").split("/")
#     try:
#         i = parts.index(src_dir_name)
#     except ValueError:
#         raise RuntimeError(f"Directory name '{src_dir_name}' not found in path: {path}")
#     parts[i] = dst_dir_name
#     dst_path = "/".join(parts)
#     if dst_ext is not None:
#         base, _ = os.path.splitext(dst_path)
#         dst_path = f"{base}_segmentation{dst_ext}"
#     else:
#         base, ext = os.path.splitext(dst_path)
#         dst_path = f"{base}_segmentation{ext}"
#     return dst_path
#
# # ========== FolderDataset（已内嵌PH2自动识别逻辑，eval脚本无需改动） ==========
# class FolderDataset(data.Dataset):
#     """
#     Directory structure for ISIC/BUSI:
#       root/
#         train/
#           img/
#           label/
#         test/
#           img/
#           label/
#     For PH2: root直接指向PH2根目录，内部自动读取IMDxxx子文件夹
#     """
#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_dir_name: str = "img",
#         label_dir_name: str = "label",
#         img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
#         mask_ext: Optional[str] = None,
#         transform: Optional[ResizeAndNormalize] = None,
#         strict_pair: bool = True,
#     ):
#         super().__init__()
#         self.root = root
#         self.split = split
#         self.img_dir_name = img_dir_name
#         self.label_dir_name = label_dir_name
#         self.img_exts = img_exts
#         self.mask_ext = mask_ext
#         self.transform = transform
#         self.strict_pair = strict_pair
#         self.pairs: List[Tuple[str, str]] = []
#
#         # ===== PH2自动劫持分支，遇到PH2目录直接走PH2加载逻辑 =====
#         if os.path.basename(root).upper() == "PH2":
#             subdirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)) and d.startswith("IMD")]
#             subdirs.sort()
#             for imd_id in subdirs:
#                 imd_folder = os.path.join(self.root, imd_id)
#                 img_subfolder = os.path.join(imd_folder, f"{imd_id}_Dermoscopic_Image")
#                 img_path = os.path.join(img_subfolder, f"{imd_id}.bmp")
#                 mask_subfolder = os.path.join(imd_folder, f"{imd_id}_lesion")
#                 mask_path = os.path.join(mask_subfolder, f"{imd_id}_lesion.bmp")
#                 if os.path.isfile(img_path) and os.path.isfile(mask_path):
#                     self.pairs.append((img_path, mask_path))
#                 else:
#                     if self.strict_pair:
#                         raise FileNotFoundError(f"Missing pair for PH2 {imd_id}: img={img_path}, mask={mask_path}")
#             if len(self.pairs) == 0:
#                 raise RuntimeError(f"No valid PH2 pairs found under {self.root}")
#             return  # PH2加载完成，直接return，跳过后面普通folder逻辑
#         # =========================================================
#
#         # 原有逻辑：ISIC / BUSI 等普通文件夹数据集
#         self.img_dir = os.path.join(root, split, img_dir_name)
#         self.label_dir = os.path.join(root, split, label_dir_name)
#         if not os.path.isdir(self.img_dir):
#             raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
#         if not os.path.isdir(self.label_dir):
#             raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
#         self.img_paths = _list_files(self.img_dir, img_exts)
#         if len(self.img_paths) == 0:
#             raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")
#         self.pairs: List[Tuple[str, str]] = []
#         for ip in self.img_paths:
#             mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
#             if not os.path.isfile(mp) and self.mask_ext is None:
#                 base, _ = os.path.splitext(mp)
#                 found = False
#                 for ext in SUPPORTED_EXTS:
#                     cand = base + ext
#                     if os.path.isfile(cand):
#                         mp = cand
#                         found = True
#                         break
#                 if not found and self.strict_pair:
#                     raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
#                 elif not found:
#                     continue
#             elif not os.path.isfile(mp) and self.strict_pair:
#                 raise FileNotFoundError(f"Label file not found: {mp}")
#             elif not os.path.isfile(mp):
#                 continue
#             self.pairs.append((ip, mp))
#         if len(self.pairs) == 0:
#             raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")
#
#     def __len__(self):
#         return len(self.pairs)
#
#     def __getitem__(self, idx: int):
#         img_path, mask_path = self.pairs[idx]
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
#         if img is None:
#             raise RuntimeError(f"Failed to read image: {img_path}")
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
#         if mask is None:
#             raise RuntimeError(f"Failed to read label: {mask_path}")
#         # (H,W) -> (H,W,1)
#         mask_hwc = mask[..., None]
#         if self.transform is not None:
#             img_t, mask_t = self.transform(img, mask_hwc)
#         else:
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
#             m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
#             mask_t = (m > 0.5).float()
#         if mask_t.max().item() < 1.0:
#             mask_t = (mask_t > 0).float()
#         meta = {
#             "img_path": img_path,
#             "mask_path": mask_path,
#             "id": os.path.splitext(os.path.basename(img_path))[0],
#         }
#         return img_t, mask_t, meta
#
# # PH2Dataset 类保留在这里备用（不会被eval调用，留着以后其他脚本用）
# class PH2Dataset(data.Dataset):
#     """
#     PH2 Dataset for skin lesion segmentation
#     root: /mnt/sda/data/PH2/
#     Folder structure:
#         PH2/
#             IMD437/
#                 IMD437_Dermoscopic_Image/
#                     IMD437.bmp        # input image
#                 IMD437_lesion/
#                     IMD437_lesion.bmp # lesion segmentation mask
#                 IMD437_roi/
#             IMD436/
#                 ...
#     """
#     def __init__(
#         self,
#         root: str,
#         transform: Optional[ResizeAndNormalize] = None,
#         strict_pair: bool = True,
#     ):
#         super().__init__()
#         self.root = root
#         self.transform = transform
#         self.strict_pair = strict_pair
#
#         self.pairs: List[Tuple[str, str]] = []
#         # 遍历root下所有IMDxxx文件夹
#         subdirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)) and d.startswith("IMD")]
#         subdirs.sort()
#
#         for imd_id in subdirs:
#             imd_folder = os.path.join(self.root, imd_id)
#             # 原图路径
#             img_subfolder = os.path.join(imd_folder, f"{imd_id}_Dermoscopic_Image")
#             img_path = os.path.join(img_subfolder, f"{imd_id}.bmp")
#             # mask路径
#             mask_subfolder = os.path.join(imd_folder, f"{imd_id}_lesion")
#             mask_path = os.path.join(mask_subfolder, f"{imd_id}_lesion.bmp")
#
#             if os.path.isfile(img_path) and os.path.isfile(mask_path):
#                 self.pairs.append((img_path, mask_path))
#             else:
#                 if self.strict_pair:
#                     raise FileNotFoundError(f"Missing pair for {imd_id}: img={img_path}, mask={mask_path}")
#
#         if len(self.pairs) == 0:
#             raise RuntimeError(f"No valid (img, label) pairs found under PH2 root: {self.root}")
#
#     def __len__(self):
#         return len(self.pairs)
#
#     def __getitem__(self, idx: int):
#         img_path, mask_path = self.pairs[idx]
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
#         if img is None:
#             raise RuntimeError(f"Failed to read image: {img_path}")
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
#         if mask is None:
#             raise RuntimeError(f"Failed to read label: {mask_path}")
#         # (H,W) -> (H,W,1)
#         mask_hwc = mask[..., None]
#         if self.transform is not None:
#             img_t, mask_t = self.transform(img, mask_hwc)
#         else:
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
#             m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
#             mask_t = (m > 0.5).float()
#         if mask_t.max().item() < 1.0:
#             mask_t = (mask_t > 0).float()
#         meta = {
#             "img_path": img_path,
#             "mask_path": mask_path,
#             "id": os.path.splitext(os.path.basename(img_path))[0],
#         }
#         return img_t, mask_t, meta






# ISIC
import os
from typing import List, Tuple, Optional
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ResizeAndNormalize:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.thr = thr

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
        img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
        H, W = self.size
        mask_ts = []
        for c in range(mask_hwc.shape[2]):
            m = Image.fromarray(mask_hwc[..., c])
            m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
            mt = TF.to_tensor(m)[0]        # Take the single channel
            mask_ts.append(mt)
        mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
        mask_t = (mask_t > self.thr).float()           # Binarize mask by threshold
        return img_t, mask_t


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
    parts = path.replace("\\", "/").split("/")
    try:
        i = parts.index(src_dir_name)
    except ValueError:
        raise RuntimeError(f"Directory name '{src_dir_name}' not found in path: {path}")
    parts[i] = dst_dir_name
    dst_path = "/".join(parts)
    if dst_ext is not None:
        base, _ = os.path.splitext(dst_path)
        #dst_path = base + dst_ext ############################
        dst_path = f"{base}_segmentation{dst_ext}"
    else:
        # 如果没有指定扩展名，仅在文件名后加 _segmentation
        base, ext = os.path.splitext(dst_path)
        dst_path = f"{base}_segmentation{ext}"
    return dst_path

class FolderDataset(data.Dataset):
    """
    Directory structure:
      root/
        train/
          img/
          label/
        test/
          img/
          label/
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "img",
        label_dir_name: str = "label",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,
        transform: Optional[ResizeAndNormalize] = None,
        strict_pair: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.img_dir = os.path.join(root, split, img_dir_name)
        self.label_dir = os.path.join(root, split, label_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_paths = _list_files(self.img_dir, img_exts)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
            if not os.path.isfile(mp) and self.mask_ext is None:
                base, _ = os.path.splitext(mp)
                found = False
                for ext in SUPPORTED_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        mp = cand
                        found = True
                        break
                if not found and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                elif not found:
                    continue
            elif not os.path.isfile(mp) and self.strict_pair:
                raise FileNotFoundError(f"Label file not found: {mp}")
            elif not os.path.isfile(mp):
                continue

            self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")
        # (H,W) -> (H,W,1)
        mask_hwc = mask[..., None]

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
            mask_t = (m > 0.5).float()

        if mask_t.max().item() < 1.0:
            mask_t = (mask_t > 0).float()

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return img_t, mask_t, meta

































# #kvasir tn3k
# #tn3k no .segmentation
# import os
# from typing import List, Tuple, Optional
# import cv2
# import torch
# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms.functional as TF
# from torchvision.transforms import InterpolationMode
#
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)
#
# class ResizeAndNormalize:
#     def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5):
#         self.size = size  # (H, W)
#         self.mean = mean
#         self.std = std
#         self.thr = thr
#
#     def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         img_pil = Image.fromarray(img_rgb)
#         img_resized = TF.resize(
#             img_pil, self.size,
#             interpolation=InterpolationMode.BICUBIC, antialias=True
#         )
#         img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
#         img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
#         H, W = self.size
#         mask_ts = []
#         for c in range(mask_hwc.shape[2]):
#             m = Image.fromarray(mask_hwc[..., c])
#             m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
#             mt = TF.to_tensor(m)[0]        # Take the single channel
#             mask_ts.append(mt)
#         mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
#         mask_t = (mask_t > self.thr).float()           # Binarize mask by threshold
#         return img_t, mask_t
#
# # 支持的图像/掩码扩展名
# SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#
# def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
#     out = []
#     for root, _, files in os.walk(dir_path):
#         for f in files:
#             if os.path.splitext(f)[1].lower() in exts:
#                 out.append(os.path.join(root, f))
#     out.sort()
#     return out
#
# def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
#     """
#     替换目录名，保持文件名不变；若指定dst_ext则替换扩展名
#     示例：img/xxx.png → mask/xxx.jpg（若dst_ext=".jpg"）
#     """
#     parts = path.replace("\\", "/").split("/")
#     try:
#         i = parts.index(src_dir_name)
#     except ValueError:
#         raise RuntimeError(f"Directory name '{src_dir_name}' not found in path: {path}")
#     parts[i] = dst_dir_name
#     dst_path = "/".join(parts)
#     # 替换扩展名（如果指定了dst_ext）
#     if dst_ext is not None:
#         base, _ = os.path.splitext(dst_path)
#         dst_path = base + dst_ext  # 直接替换为指定扩展名
#     return dst_path
#
# class FolderDataset(data.Dataset):
#     """
#     Directory structure：
#       root/
#         train/
#           image/          # 图像文件：xxx.png/xxx.jpg/...
#           mask/           # 掩码文件：xxx.png/xxx.jpg/...（与图像同名）
#         test/
#           image/
#           mask/
#     """
#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_dir_name: str = "image",
#         label_dir_name: str = "mask",
#         img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
#         mask_ext: Optional[str] = None,  # 恢复默认值为None，由用户指定或自动匹配
#         transform: Optional[ResizeAndNormalize] = None,
#         strict_pair: bool = True,
#     ):
#         super().__init__()
#         self.root = root
#         self.split = split
#         self.img_dir_name = img_dir_name
#         self.label_dir_name = label_dir_name
#         self.img_dir = os.path.join(root, split, img_dir_name)
#         self.label_dir = os.path.join(root, split, label_dir_name)
#         self.mask_ext = mask_ext
#         self.transform = transform
#         self.strict_pair = strict_pair
#
#         if not os.path.isdir(self.img_dir):
#             raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
#         if not os.path.isdir(self.label_dir):
#             raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
#
#         self.img_paths = _list_files(self.img_dir, img_exts)
#         if len(self.img_paths) == 0:
#             raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")
#
#         self.pairs: List[Tuple[str, str]] = []
#         for ip in self.img_paths:
#             # 生成掩码路径：替换目录名 + 可选替换扩展名
#             mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
#             # 处理掩码文件不存在的情况：遍历支持的扩展名寻找同名掩码
#             if not os.path.isfile(mp):
#                 base, _ = os.path.splitext(mp)
#                 found = False
#                 # 遍历支持的扩展名，寻找同名掩码
#                 for ext in SUPPORTED_EXTS:
#                     cand = base + ext
#                     if os.path.isfile(cand):
#                         mp = cand
#                         found = True
#                         break
#                 if not found:
#                     if self.strict_pair:
#                         # 修正错误提示：不再硬编码.jpg，显示实际寻找的路径
#                         raise FileNotFoundError(f"Mask file not found for {ip} (tried: {mp} and {base}.*)")
#                     else:
#                         continue
#             self.pairs.append((ip, mp))
#
#         if len(self.pairs) == 0:
#             raise RuntimeError(f"No valid (img, mask) pairs in {self.split} set!")
#
#     def __len__(self):
#         return len(self.pairs)
#
#     def __getitem__(self, idx: int):
#         img_path, mask_path = self.pairs[idx]
#
#         # 读取图像（BGR格式）
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
#         if img is None:
#             raise RuntimeError(f"Failed to read image: {img_path}")
#
#         # 读取掩码：灰度图（单通道）
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
#         if mask is None:
#             raise RuntimeError(f"Failed to read mask: {mask_path}")
#         # (H,W) -> (H,W,1)（适配后续transform的通道处理）
#         mask_hwc = mask[..., None]
#
#         if self.transform is not None:
#             img_t, mask_t = self.transform(img, mask_hwc)
#         else:
#             # 无transform时的默认处理：转RGB+归一化
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
#             m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
#             mask_t = (m > 0.5).float()
#
#         # 兜底：确保掩码是二值的（0/1）
#         if mask_t.max().item() < 1.0:
#             mask_t = (mask_t > 0).float()
#
#         meta = {
#             "img_path": img_path,
#             "mask_path": mask_path,
#             "id": os.path.splitext(os.path.basename(img_path))[0],
#         }
#         return img_t, mask_t, meta