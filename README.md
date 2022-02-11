# dacl-demo

Repo to demonstrate how to use baselines from bikit and dacl.ai.


***Examples of images representing detectable damage with available dacl-models.** Crack (Top left); Spalling, Effloresence, BarsExposed, Rust (Top right); Crack, Efflorescence (Bottom left); Spalling, Effloresence, BarsExposed, Rust (Bottom right)*

## Available old Models from bikit-paper:

| Modelname             | Dataset           | EMR   | F1   | Tag          | Checkpoint                |
|-----------------------|-------------------|-------|------|--------------|---------------------------|
| Code_res_dacl         | codebrim_balanced | 73.73 | 0.85 | ResNet       | Code_res_dacl.pth         |
| Code_mobilev2_dacl    | codebrim_balanced |70.41  | 0.84 | MobileNetV2  | Code_mobilev2_dacl.pth    |
| Code_mobile_dacl      | codebrim_balanced | 69.46 | 0.83 | MobileNet    | Code_mobile_dacl.pth      |
| Code_eff_dacl         | codebrim_balanced | 68.67 | 0.84 | EfficientNet | Code_eff_dacl.pth         |
| McdsBikit_mobile_dacl | mcds_Bikit        | 54.44 | 0.66 | MobileNet    | McdsBikit_mobile_dacl.pth |
| McdsBikit_eff_dacl    | mcds_Bikit        | 51.85 | 0.65 | EfficientNet | McdsBikit_eff_dacl.pth    |
| McdsBikit_res_dacl    | mcds_Bikit        | 48.15 | 0.62 | ResNet       | McdsBikit_res_dacl.pth    |


## Structure

```
dacl_demo
├── assets
	└── *.jpg #example images
├── demo.ipynb # Main code
├── cat_to_name.json
└── models
	└── *.pth #checkpoints
```

## Meta4

These are the temorary results on test data of meta4 dataset:

| new_cp | origin_cp | Dataset | Split | Approach | Base      | ExactMatchRatio | F1    | Precision | Recall | Accuracy | AUROC | Recall-NoDamage | Recall-Crack | Recall-Efflorescence | Recall-Spalling | Recall-BarsExposed | Recall-Rust |
|--------------------------------|---------------------|---------|-------|----------|-----------|-----------------|-------|-----------|--------|----------|-------|-----------------|--------------|----------------------|-----------------|--------------------|-------------|
| META4_MobileNetV3Large_ho.pth  | comic-bee-1         | meta4   | test  | HO       | mobilenet | 65.34           | 69.96 | 82.19     | 66.23  | 88.96    | 94.45 | 98.68           | 36.41        | 54.19                | 76.67           | 60.95              | 70.48       |
| META4_MobileNetV3Large_hta.pth | hearty-elevator-18  | meta4   | test  | HTA      | mobilenet | 77.84           | 79.40 | 87.68     | 74.85  | 93.01    | 97.69 | 99.17           | 60.82        | 67.60                | 81.48           | 69.52              | 70.48       |
| META4_ResNet50_ho.pth          | frosty-wood-6       | meta4   | test  | HO       | resnet    | 63.71           | 61.31 | 78.51     | 55.55  | 88.36    | 93.26 | 97.32           | 36.01        | 31.28                | 59.63           | 51.90              | 57.14       |
| META4_MobileNetV3Large_dhb.pth | daily-dream-16      | meta4   | test  | DHB      | mobilenet | 79.28           | 79.96 | 86.13     | 76.78  | 93.46    | 97.63 | 98.83           | 64.79        | 69.27                | 81.11           | 66.19              | 80.48       |
| META4_ResNet50_hta.pth         | golden-spaceship-32 | meta4   | test  | HTA      | resnet    | 79.72           | 79.22 | 85.65     | 75.55  | 93.57    | 97.67 | 98.68           | 65.28        | 70.39                | 77.04           | 66.67              | 75.24       |
| META4_ResNet50_dhb.pth         | olive-snow-29       | meta4   | test  | DHB      | resnet    | 80.23           | 79.55 | 85.78     | 76.09  | 93.81    | 97.48 | 98.77           | 66.61        | 67.04                | 85.56           | 73.33              | 65.24       |