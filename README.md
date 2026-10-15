# FSD-Net
## Testing
### Pre-trained Weights
The checkpoint `best_model_all.pth` for ISIC2017 segmentation can be downloaded via Baidu Netdisk:
> Link: https://pan.baidu.com/s/1L74TSNDOlw-bFj8OxjDE8Q?pwd=wmec
> Extraction code: `wmec`

>  Note: The complete source code will be released upon acceptance of this paper.

### Run Test Command
```bash
python test_segdinov3.py \
--data_dir /mnt/sda/data/ \
--dataset isic \
--input_h 256 \
--input_w 256 \
--dino_size s \
--dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
--repo_dir ./dinov3 \
--ckpt /home/huanhuan/code/segdino2/runs/best_model_all.pth
