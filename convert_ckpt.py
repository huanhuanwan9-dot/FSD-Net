import torch

def convert_old_dpt_to_fsdnet(old_ckpt_path: str, new_ckpt_path: str):
    """
    Convert old checkpoint from dpt.py (DPT) to new fsdi.py (FSD_Net) compatible weights.
    :param old_ckpt_path: path of source old checkpoint
    :param new_ckpt_path: output path of converted checkpoint
    """
    ckpt = torch.load(old_ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        src_sd = ckpt["state_dict"]
        is_full_ckpt = True
    else:
        src_sd = ckpt
        is_full_ckpt = False

    name_mapping = {
        "neck.": "fsd_neck.",
        "neck.fcm.": "fsd_neck.ccm.",
        "neck.fuseFeature.": "fsd_neck.fdb.",
        "neck.MDAF_L.": "fsd_neck.lk_cda_low.cdf.",
        "neck.MDAF_H.": "fsd_neck.lk_cda_high.cdf.",
    }

    new_state_dict = {}
    for old_key, tensor in src_sd.items():
        new_key = old_key
        for old_prefix, new_prefix in name_mapping.items():
            new_key = new_key.replace(old_prefix, new_prefix)
        new_state_dict[new_key] = tensor

    if is_full_ckpt:
        ckpt["state_dict"] = new_state_dict
        torch.save(ckpt, new_ckpt_path)
    else:
        torch.save(new_state_dict, new_ckpt_path)

    print("Checkpoint convert finished.")
    print(f"Source checkpoint: {old_ckpt_path}")
    print(f"Output checkpoint: {new_ckpt_path}")


if __name__ == "__main__":
    OLD_CKPT = "./runs/xxx/ckpts/best_old.pth"
    NEW_CKPT = "./best_converted_fsdnet.pth"
    convert_old_dpt_to_fsdnet(OLD_CKPT, NEW_CKPT)
