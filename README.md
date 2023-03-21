# bikit-models

Please visit [Building Inspection Toolkit](https://github.com/phiyodr/building-inspection-toolkit/) for further details on how to use these models. 


## New Usage daclnets

```python 
from bikit_models.daclnet import build_dacl
from bikit_models.utils import process_img_daclnet, make_prediciton

img_path = "assets/image_0000468_crop_0000001.png"
cp_path = "models/checkpoints/codebrim-classif-balancedcodebrim-classif-balanced_ResNet50_hta.pth"
model, cat_to_name = build_dacl(cp_path=cp_path)
img = process_img_daclnet(img_path)
make_prediciton(model, img, cat_to_name)
```

## New Usage vistranet_0
```python 
from bikit_models.vistranet_0 import build_vistra_0
from bikit_models.utils import process_img_vistranet, make_prediciton

img_path = "assets/image_0000468_crop_0000001.png"
cp_path = "models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_0.pth"
model, cat_to_name = build_vistra_0(cp_path=cp_path)
img = process_img_vistranet(img_path)
make_prediciton(model, img, cat_to_name)
```

## New Usage vistranet_1
```python 
from bikit_models.vistranet_1 import build_vistra_1
from bikit_models.utils import process_img_vistranet, make_prediciton

img_path = "assets/image_0000468_crop_0000001.png"
cp_path = "models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_1.pth"
model, cat_to_name = build_vistra_1(cp_path=cp_path)
img = process_img_vistranet(img_path)
make_prediciton(model, img, cat_to_name, activation=False) # activation is already performed 
```

## Old Usage
```python
from bikit.utils import load_model, load_img_from_url
from bikit.models import make_prediction

img = load_img_from_url("https://github.com/phiyodr/building-inspection-toolkit/raw/master/bikit/data/11_001990.jpg")
model, metadata = load_model("codebrim-classif-balanced_MobileNetV3-Large_hta", add_metadata=True)
prob, pred = make_prediction(model, img, metadata, print_predictions=True, preprocess_image=True)
```

## Performances of available models

| Cp-name                                         | TrainedOn                 | TrainCropSize | MetricsReportedOn | ExactMatchRatio | F1    | Precision | Recall | Accuracy | AUROC | Recall-NoDamage | Recall-Crack | Recall-Efflorescence | Recall-Spalling | Recall-BarsExposed | Recall-Rust | Scaling | Other |
|-------------------------------------------------|---------------------------|---------------|-------------------|-----------------|-------|-----------|--------|----------|-------|-----------------|--------------|----------------------|-----------------|--------------------|-------------|---------|-------|
| codebrim-classif_MobileNetV3-Large_hta          | codebrim-classif          | 224           | itself            | 70.57           | 83.04 | 86.27     | 81.07  | 92.25    | 96.67 | 94              | 84           | 82.67                | 65.1            | 84.67              | 76          |         |       |
| codebrim-classif-balanced_ViT_s8_1              | codebrim-classif-balanced | 224           | itself            | 79.59           | 89.82 | 91.3      | 88.54  | 86.65    |       | 94              | 89.33        | 85.91                | 84              | 90                 | 88          |         |       |
| codebrim-classif-balanced_ViT_s8_0              | codebrim-classif-balanced | 224           | itself            | 77.53           | 89.62 | 89.65     | 89.76  | 86.05    |       | 95.33           | 92           | 84.56                | 90.67           | 90.67              | 85.33       |         |       |
| codebrim-classif-balanced_ResNet50_hta          | codebrim-classif-balanced | 224           | itself            | 71.36           | 84.13 | 85.33     | 83.2   | 92.59    | 96.99 | 93.33           | 85.33        | 77.18                | 84.67           | 85.33              | 73.33       |         |       |
| codebrim-classif-balanced_MobileNetV3-Large_hta | codebrim-classif-balanced | 224           | itself            | 68.99           | 82.77 | 84.36     | 81.75  | 91.98    | 96.45 | 94              | 80           | 72.48                | 84.67           | 86.67              | 72.67       |         |       |
| codebrim-classif-balanced_EfficientNetV1-B0_hta | codebrim-classif-balanced | 224           | itself            | 65.66           | 81    | 80.33     | 82.52  | 90.88    | 96.06 | 90              | 77.33        | 67.79                | 88.67           | 92.67              | 78.67       |         |       |
| dacl1k_MobileNetV3-Large_dhb                    | dacl1k                    | 512           | itself            | 31.51           | 66.94 | 76.24     | 60.49  | 78.54    | 83.41 | 73.91           | 42.50        | 52.27                | 68.89           | 56.60              | 68.75       |         |       |
| dacl1k_MobileNetV3-Large_hta                    | dacl1k                    | 224           | itself            | 23.29           | 56.94 | 75.72     | 46.95  | 76.18    | 82.58 | 65.22           | 22.5         | 43.18                | 44.44           | 35.85              | 70.54       |         |       |
| mcds_bikit_MobileNetV3-Large_hta                | mcds_bikit                | 224           | itself            | 54.44           | 65.52 | 79.48     | 59.44  | 90.65    | 93.67 | 70              | 76.67        | 90                   | 58.89           | 21.67              | 68.33       | 43.33   | 46.67 |
| mcds_bikit_EfficientNetV1-B0_dha                | mcds_bikit                | 224           | itself            | 51.85           | 64.55 | 77.72     | 58.06  | 90.23    | 91.91 | 46.67           | 73.33        | 80                   | 61.11           | 38.33              | 75          | 43.33   | 46.67 |
| mcds_bikit_ResNet50_dha                         | mcds_bikit                | 224           | itself            | 48.15           | 62.33 | 80.88     | 54.93  | 89.81    | 93.07 | 66.67           | 73.33        | 86.67                | 44.44           | 23.33              | 65          | 36.67   | 43.33 |
| meta2_MobileNetV3-Large_hta                     | meta2                     | 224           | itself            | 70.41           | 82.99 | 87.43     | 80.1   | 92.39    | 96.5  | 94.44           | 88.33        | 70.39                | 82.22           | 68.57              | 76.67       |         |       |
| meta2+dacl1k_MobileNetV3-Large_dhb              | meta2+dacl1k              | 512           | dacl1k            | 31.51           | 67.08 | 82.09     | 57.88  | 80.44    | 85.77 | 65.22           | 31.25        | 54.55                | 65.56           | 56.60              | 74.11       |         |       |
| meta2+dacl1k_MobileNetV3-Large_hta              | meta2+dacl1k              | 224           | itself            | 49.22           | 66.48 | 72.17     | 66.48  | 85.45    | 89.27 | 32.35           | 83.13        | 61.8                 | 78.33           | 65.02              | 78.26       |         |       |
| meta3_MobileNetV3-Large_hta                     | meta3                     | 224           | itself            | 81.52           | 85.28 | 90.84     | 81.2   | 95.31    | 98.03 | 97.29           | 93.75        | 65.92                | 82.59           | 71.43              | 76.19       |         |       |
| meta3+dacl1k_MobileNetV3-Large_dhb              | meta3+dacl1k              | 512           | dacl1k            | 32.42           | 68.94 | 83.03     | 60.24  | 81.13    | 84.93 | 67.39           | 36.25        | 50.00                | 73.33           | 60.38              | 74.11       |         |       |
| meta3+dacl1k_MobileNetV3-Large_hta              | meta3+dacl1k              | 224           | itself            | 75.22           | 82.14 | 90.41     | 76.18  | 93.4     | 96.85 | 93.54           | 85.54        | 60.67                | 78.33           | 65.4               | 73.6        |         |       |
| meta4_MobileNetV3-Large_hta                     | meta4                     | 224           | itself            | 77.84           | 79.4  | 87.68     | 74.85  | 93.01    | 97.69 | 99.17           | 60.82        | 67.6                 | 81.48           | 69.52              | 70.48       |         |       |
| meta4+dacl1k_MobileNetV3-Large_hta              | meta4+dacl1k              | 224           | itself            | 76.81           | 76.44 | 86.19     | 71.11  | 92.76    | 97.4  | 98.42           | 61.85        | 59.18                | 73.61           | 59.7               | 73.91       |         |       |

** The performance of the models trained on codebrim-classif-balanced dataset in the bikit-models repo differ from the original bikit paper due to sanity changes in bikit. The original models from the paper can be found at [dacl-demo](https://github.com/jfltzngr/dacl-demo) repo.