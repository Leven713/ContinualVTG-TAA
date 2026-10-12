## 📚 Datasets
<b>QVHighlights</b> : The data is set as followed, you need to replace the feat_root path in the bash file with your own. You can download the preprocessed QVHighlight dataset from [
qvhighlight.zip](https://drive.google.com/file/d/1LXsZZBsv6Xbg_MmNQOezw0QYKccjcOkP/view?usp=sharing). 

```txt
QVHighlight
└──── features
    ├── slowfast_features
    ├── clip_text_features
    ├── clip_features
    └── clip_sub_features

```

## 🛠️ Environment Setup 
```txt
conda create -n taa python=3.7
conda activate taa

# Install pytorch 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
pip install -r requirements.txt 

```

## 🚀 Train

`cd ContinualVTG-TAA-main`

- QVHighlights

  - **TAA (Ours)**

    `bash moment_detr_CL/scripts/qvhighlight/TAA.sh`

  - Fine-tuning (FT)

    `bash moment_detr_CL/scripts/qvhighlight/FT.sh`

  - Feature Knowledge Distillation (FKD)

    `bash moment_detr_CL/scripts/qvhighlight/FKD.sh`

  - Elastic Weights Consolidation (EWC)

    `bash moment_detr_CL/scripts/qvhighlight/EWC.sh`

  - Experience Replay (ER)

    `bash moment_detr_CL/scripts/qvhighlight/ER.sh`

  - Joint Training (JT)

    `bash moment_detr_CL/scripts/qvhighlight/JT.sh`


  You need to modify the relevant path to your own.  
