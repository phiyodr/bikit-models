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

These are the temorary resulkts on test data of meta4 dataset:

|   | cp(REDO!)                      | Dataset | Split | Approach | Base      | ExactMatchRatio   | F1                | Precision         | Recall            | Accuracy          | AUROC             | Recall-NoDamage   | Recall-Crack      | Recall-Efflorescence | Recall-Spalling   | Recall-BarsExposed | Recall-Rust       |
|---|--------------------------------|---------|-------|----------|-----------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|----------------------|-------------------|--------------------|-------------------|
| 0 | META4_ResNet50_dhb.pth         | meta4   | test  | dha      | resnet    | 0.802309007981756 | 0.795473635196686 | 0.857846617698669 | 0.760902106761932 | 0.938070178031921 | 0.974837720394134 | 0.987699866294861 | 0.666051685810089 | 0.670391082763672    | 0.855555534362793 | 0.733333349227905  | 0.65238094329834  |
| 0 | META4_MobileNetV3Large_dhb.pth | meta4   | test  | dha      | mobilenet | 0.792759407069555 | 0.799630284309387 | 0.861323773860931 | 0.767789840698242 | 0.934601902961731 | 0.976343631744385 | 0.988314867019653 | 0.647908985614777 | 0.692737400531769    | 0.811111092567444 | 0.661904752254486  | 0.80476188659668  |
| 0 | META4_MobileNetV3Large_ho.pth  | meta4   | test  | ho       | mobilenet | 0.653363740022805 | 0.69956374168396  | 0.821889638900757 | 0.662285447120667 | 0.889609456062317 | 0.944478988647461 | 0.986777365207672 | 0.364083647727966 | 0.54189944267273     | 0.766666650772095 | 0.609523832798004  | 0.704761922359467 |
| 0 | META4_ResNet50_ho.pth          | meta4   | test  | ho       | resnet    | 0.637115165336374 | 0.613096415996552 | 0.785095512866974 | 0.555492520332336 | 0.883575677871704 | 0.932595193386078 | 0.973247230052948 | 0.360086113214493 | 0.312849164009094    | 0.596296310424805 | 0.519047617912293  | 0.571428596973419 |
