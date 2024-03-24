# LightweightGAN
Unofficial custmized LightweightGAN.

## python LightweightGAN.py --help
```
usage: LightweightGAN.py [-h] [--image_dir IMAGE_DIR] [--result_dir RESULT_DIR] [--weight_dir WEIGHT_DIR]
                         [--image_size IMAGE_SIZE] [--lr LR] [--mul_lr_dis MUL_LR_DIS] [--batch_size BATCH_SIZE]
                         [--num_train NUM_TRAIN] [--cpu] [--generate GENERATE] [--noresume]

options:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
  --result_dir RESULT_DIR
  --weight_dir WEIGHT_DIR
  --image_size IMAGE_SIZE
  --lr LR
  --mul_lr_dis MUL_LR_DIS
  --batch_size BATCH_SIZE
  --num_train NUM_TRAIN
  --cpu
  --generate GENERATE
  --noresume
```

### for example
```
python LightweightGAN.py --image_dir "/usr/share/datasets/image_dir"
```
and
```
python LightweightGAN.py --generate 10
```

**Note:**
- resume.pkl is a file that saves learning checkpoints for resume and includes models, weight data, etc.
- If a weight.pth file exists in the current directory, the network weights will be automatically read.

**Default use only WGAN-gp. Other methods(LSGAN,APA,ADA,ICR) is comment-out.**
