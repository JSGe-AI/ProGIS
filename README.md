

# Progis: Prototype-guided interactive segmentation for pathological images

This is the code repository for the paper: **Progis: Prototype-guided interactive segmentation for pathological images**.



## Project Overview

This project consists of two main components: **Pixel-level Segmentation** and **Slide-level Segmentation**.

---

## 0. Installation & Environment Setup

Before running the scripts, ensure you have [Conda](https://docs.conda.io/en/latest/) installed. You can set up the environment using the provided `environment.yml` file:

```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate [your_env_name]

```

---

## 1. Pixel-level Segmentation

### Data Preprocessing

Before starting the training process, complete the following data preparation steps:

* **Superpixel Segmentation Maps**: Run the **SLIC algorithm** to generate superpixel maps for each image and save them in `.npy` format.
* **RoI Interaction Signals**: Generate interaction signals for the largest connected component (LCC) of the target class using the **morphological skeleton** method (Nuclick style) and save as `.npy`.
* **LCC Centroids**: Extract the center points of the largest connected components for the target segmentation class and save as `.npy`.

### Model Training

The pixel-level pipeline involves training the **P-RoIseg** and the **Backbone** network.

#### 1.1 P-RoIseg Training

```bash
python ROI_Segmentor/EfficientUnet-PyTorch-master/train_P-ROIseg.py

```

#### 1.2 Backbone Training (Using Contrastive Loss)

Select the architecture based on your requirements:

* **ResNet-18**: `python X_ISF\models\train_resunet_DDP.py`
* **EfficientNet**: `python ROI_Segmentor/EfficientUnet-PyTorch-master/train_backbone_efficient.py`
* **ViT**: `python TransUNet/train_ViT_B_singleGPU.py`

### Inference

* **ResNet-18**: `python ROI_Segmentor/EfficientUnet-PyTorch-master/inference_backbone_ResNet.py`
* **EfficientNet**: `python ROI_Segmentor/EfficientUnet-PyTorch-master/inference_backbone_efficient.py`
* **ViT**: `python TransUNet/inference_VIT_B.py`

---

## 2. Slide-level Segmentation (WSI)

### Data Preprocessing

1. **Feature Extraction**: Use a foundation model to extract patch-level image features and store them in `.h5` format.
2. **Interaction Signal Generation**: Generate interaction signals using the morphological skeleton method (stores the coordinates of each patch within the interaction signal).

### Model Training

* **C16 Dataset**: `python WSI_ISF/WSI_model_ROI_C16.py`
* **Lung Dataset**: `python WSI_ISF/WSI_model_ROI_Lung.py`

### Inference

* **C16 Inference**: `python WSI_ISF/WSI_model_inference_C16.py`
* **Lung Inference**: `python WSI_ISF/WSI_model_inference_Lung.py`



## Citation

If you find this project useful for your research, please cite our work:

```bibtex
@ARTICLE{11168941,
  author={Ge, Jiusong and Zhang, Di and Zhan, Yingkang and Liu, Jiashuai and Gong, Tieliang and Wu, Jialun and Crispin-Ortuzar, Mireia and Li, Chen and Gao, Zeyu},
  journal={IEEE Transactions on Medical Imaging}, 
  title={ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images}, 
  year={2026},
  volume={45},
  number={3},
  pages={881-892},
  keywords={Image segmentation;Pathology;Prototypes;Accuracy;Training;Navigation;Germanium;Feature extraction;Tumors;Predictive models;Interactive segmentation;tissue segmentation;prototype learning;computational pathology},
  doi={10.1109/TMI.2025.3611123}}

