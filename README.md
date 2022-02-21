# bikit-models


Please visit [Building Inspection Toolkit](https://github.com/phiyodr/building-inspection-toolkit/) for all details on how to use these models 

```python
from bikit.utils import load_model, load_img_from_url
from bikit.models import make_prediction

img = load_img_from_url("https://github.com/phiyodr/building-inspection-toolkit/raw/master/bikit/data/11_001990.jpg")
model, metadata = load_model("MCDSbikit_ResNet50_dhb", add_metadata=True)
prob, pred = make_prediction(model, img, metadata, print_predictions=True, preprocess_image=True)
```

## `codebrim_balanced` and `mcds_bikit`



| Modelname                                     | Dataset           | EMR   | F1   | Tag          | Checkpoint                |
|-----------------------------------------------|-------------------|-------|------|--------------|---------------------------|
| CODEBRIMbalanced_ResNet50_hta                 | codebrim_balanced | 73.73 | 0.85 | ResNet       | CODEBRIMbalanced_ResNet50_hta.pth         |
| CODEBRIMbalanced_MobileNetV2                  | codebrim_balanced | 70.41 | 0.84 | MobileNetV2  | CODEBRIMbalanced_MobileNetV2.pth    |
| CODEBRIMbalanced_MobileNetV3Large_hta         | codebrim_balanced | 69.46 | 0.83 | MobileNet    | CODEBRIMbalanced_MobileNetV3Large_hta.pth      |
| CODEBRIMbalanced_EfficientNetV1B0_dhb         | codebrim_balanced | 68.67 | 0.84 | EfficientNet | CODEBRIMbalanced_EfficientNetV1B0_dhb.pth         |
| MCDSbikit_MobileNetV3Large_hta                | mcds_bikit        | 54.44 | 0.66 | MobileNet    | MCDSbikit_MobileNetV3Large_hta.pth |
| MCDSbikit_EfficientNetV1B0_dhb                | mcds_bikit        | 51.85 | 0.65 | EfficientNet | MCDSbikit_EfficientNetV1B0_dhb.pth    |
| MCDSbikit_ResNet50_dhb                        | mcds_bikit        | 48.15 | 0.62 | ResNet       | MCDSbikit_ResNet50_dhb.pth    |



## `meta4`

These are the results on test data of meta4 dataset:

| new_cp | origin_cp | Dataset | Split | Approach | Base      | ExactMatchRatio | F1    | Precision | Recall | Accuracy | AUROC | Recall-NoDamage | Recall-Crack | Recall-Efflorescence | Recall-Spalling | Recall-BarsExposed | Recall-Rust |
|--------------------------------|---------------------|---------|-------|----------|-----------|-----------------|-------|-----------|--------|----------|-------|-----------------|--------------|----------------------|-----------------|--------------------|-------------|
| META4_MobileNetV3Large_ho.pth  | comic-bee-1         | meta4   | test  | HO       | mobilenet | 65.34           | 69.96 | 82.19     | 66.23  | 88.96    | 94.45 | 98.68           | 36.41        | 54.19                | 76.67           | 60.95              | 70.48       |
| META4_MobileNetV3Large_hta.pth | hearty-elevator-18  | meta4   | test  | HTA      | mobilenet | 77.84           | 79.40 | 87.68     | 74.85  | 93.01    | 97.69 | 99.17           | 60.82        | 67.60                | 81.48           | 69.52              | 70.48       |
| META4_ResNet50_ho.pth          | frosty-wood-6       | meta4   | test  | HO       | resnet    | 63.71           | 61.31 | 78.51     | 55.55  | 88.36    | 93.26 | 97.32           | 36.01        | 31.28                | 59.63           | 51.90              | 57.14       |
| META4_MobileNetV3Large_dhb.pth | daily-dream-16      | meta4   | test  | DHB      | mobilenet | 79.28           | 79.96 | 86.13     | 76.78  | 93.46    | 97.63 | 98.83           | 64.79        | 69.27                | 81.11           | 66.19              | 80.48       |
| META4_ResNet50_hta.pth         | golden-spaceship-32 | meta4   | test  | HTA      | resnet    | 79.72           | 79.22 | 85.65     | 75.55  | 93.57    | 97.67 | 98.68           | 65.28        | 70.39                | 77.04           | 66.67              | 75.24       |
| META4_ResNet50_dhb.pth         | olive-snow-29       | meta4   | test  | DHB      | resnet    | 80.23           | 79.55 | 85.78     | 76.09  | 93.81    | 97.48 | 98.77           | 66.61        | 67.04                | 85.56           | 73.33              | 65.24       |
