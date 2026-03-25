# VDMamba
VDMamba: Vector Decomposition in Vision Mamba for Image Deraining and Beyond (TMM'2026)

[Kui Jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl)

**Paper**: [VDMamba: Vector Decomposition in Vision Mamba for Image Deraining and Beyond](https://ieeexplore.ieee.org/abstract/document/11359071)

## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train_VDMamba.py
```


## Evaluation

1. Download the model and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800) from here and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
```
python test_VDMamba.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
evaluate_PSNR_SSIM.m 
```

#### Reference:

If you find our code or ideas useful, please cite:

    @ARTICLE{11359071,
       author={Jiang, Kui and Jiang, Junjun and Wang, Shiqi and Ren, Wenqi and Lin, Chia-Wen and Li, Zhengguo},
       journal={IEEE Transactions on Multimedia}, 
       title={VDMamba: Vector Decomposition in Vision Mamba for Image Deraining and Beyond}, 
       year={2026},
       volume={},
       number={},
       pages={1-13}
    }


