# 👤 Deep Learning Face Verification: Architectures, Metric Learning & Benchmarking

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNet--B0-blue.svg)](https://arxiv.org/abs/1905.11946)
[![MobileNetV3](https://img.shields.io/badge/Mobile-MobileNetV3--Large-green.svg)](https://arxiv.org/abs/1905.02244)
[![ArcFace & Triplet](https://img.shields.io/badge/Loss-ArcFace%20%26%20Batch%20Hard%20Triplet-red.svg)](https://arxiv.org/abs/1801.07698)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Bilingual README: [Français](#-version-française) | [English](#-english-version)*

---

## 🇫🇷 Version Française

### 🎯 Objectif
Le projet **Face Verification 101** est un cadre d'expérimentation et d'évaluation comparative approfondie de modèles de Deep Learning pour l'authentification et la vérification faciale biométrique (1:1 Verification). L'objectif est d'explorer et de confronter différentes familles de backbones convolutifs (légers et optimisés) ainsi que plusieurs fonctions de perte métrique avancées (**ArcFace Margin Loss** et **Batch Hard Triplet Loss**) afin d'identifier le meilleur compromis entre précision biométrique, compacité du modèle et vitesse d'inférence.

### 🛠️ Stack Technologique
- **Deep Learning Framework** : PyTorch, Torchvision, CUDA.
- **Architectures de Réseaux** :
  - **EfficientNet-B0** couplé aux modules d'attention spatiale et de canaux **CBAM** (*Convolutional Block Attention Module*).
  - **MobileNetV3-Large** optimisé pour l'inférence mobile et les architectures embarquées.
  - **ResNet** comme référence comparative standard (*baseline*).
- **Metric Learning & Fonctions de Perte** :
  - **ArcFace** (*Additive Angular Margin Loss*) avec projection sur hypersphère.
  - **Batch Hard Triplet Loss** pour maximiser la distance euclidienne relative entre paires positives et paires négatives les plus dures (*hard negatives mining*).
- **Vision par Ordinateur & Prétraitement** : OpenCV, PIL, détection et recadrage automatique de visages (`Pretraitement/01_Cleaning_Crop.ipynb`).
- **Évaluation Biométrique** : Courbes ROC, courbes DET (Detection Error Tradeoff), distributions des scores de similarité, et projections t-SNE des embeddings.

### 👩‍💻 Mon Rôle & Contributions
- **Pipeline de Détection & Recadrage de Visages (`Pretraitement/`)** :
  - Conception de la chaîne de détection faciale, alignement des yeux et recadrage centré (112x112 px) pour éliminer les artefacts d'arrière-plan.
  - Préparation d'un protocole d'évaluation scrupuleux (Split Train/Val/Test 80/10/10) sans fuite d'identité (*unseen identities testing*).
- **Implémentation & Entraînement des Architectures Métriques** :
  - **Branche EfficientNet-B0 + CBAM + ArcFace** : implémentation de la couche de marge angulaire additive combinée à l'attention CBAM pour focaliser les convolutions sur les traits discriminants du visage (yeux, nez, bouche).
  - **Branche MobileNetV3 + Batch Hard Triplet Loss** : conception de l'échantillonneur de triplets difficiles au sein de chaque batch d'entraînement.
  - Étude comparative de l'impact des canaux de couleur (RGB vs Nuances de gris / GRAYSCALE).
- **Évaluation et Benchmarking Biométrique** :
  - Calcul de l'aire sous la courbe (AUC-ROC), analyse des taux d'erreur biométriques FAR (*False Acceptance Rate*) et FRR (*False Rejection Rate*), et détermination du seuil de décision optimal.

### 📊 Résultats & Métriques Clés
- **Performance EfficientNet-B0 + CBAM + ArcFace** :
  - **Score AUC** : **0.8872** (AUC ~ 89%).
  - **Taux de Fausse Acceptation (FAR)** : **11.19%** au seuil optimal de `0.2111`, garantissant un niveau élevé de sécurité anti-usurpation.
  - **Taux de Faux Rejet (FRR)** : **18.18%**.
- **Séparabilité de l'Espace Latent** : Les visualisations t-SNE démontrent une forte concentration des représentations d'une même identité et un espacement net entre identités distinctes.
- **Portabilité Embarquée** : MobileNetV3-Large atteint une vitesse d'inférence en temps réel (>30 FPS) adaptée aux smartphones et systèmes IoT.

---

## 🇬🇧 English Version

### 🎯 Objective
**Face Verification 101** is a deep metric learning benchmark and experimental framework for automated 1:1 facial biometric authentication. It provides a systematic comparative study analyzing state-of-the-art compact convolutional backbones evaluated across distinct metric learning objectives (**ArcFace Additive Angular Margin Loss** and **Batch Hard Triplet Loss**) to identify optimal trade-offs between identity discrimination accuracy, computational complexity, and inference latency.

### 🛠️ Tech Stack
- **Deep Learning Framework**: PyTorch, Torchvision, CUDA acceleration.
- **Neural Architectures**:
  - **EfficientNet-B0** enriched with **CBAM** (Convolutional Block Attention Module: spatial + channel attention).
  - **MobileNetV3-Large** engineered for resource-constrained edge/mobile inference.
  - **ResNet** established as a baseline comparative benchmark.
- **Loss Functions & Metric Learning**:
  - **ArcFace**: Additive Angular Margin penalty projecting facial features onto a normalized hypersphere.
  - **Batch Hard Triplet Loss**: Online mining of the hardest positive and negative pairs per mini-batch.
- **Computer Vision & Preprocessing**: OpenCV, PIL, facial bounding-box localization, landmark alignment, and 112x112 standardization.
- **Biometric Profiling**: ROC Curves, DET analysis, score distributions, and t-SNE latent embedding clustering.

### 👩‍💻 My Role & Key Contributions
- **Data Ingestion & Face Preprocessing (`Pretraitement/`)**:
  - Built an automated face extraction and cropping pipeline enforcing uniform 112x112 resolution.
  - Enforced an identity-disjoint Train/Val/Test (80/10/10) split to benchmark generalization on unseen identities.
- **Deep Learning Model Engineering**:
  - Implemented the custom ArcFace layer with tunable margin and scale parameters atop EfficientNet-B0 + CBAM.
  - Developed the online Batch Hard Triplet Loss mining routine for MobileNetV3-Large.
  - Conducted comparative ablation studies assessing the impact of Grayscale versus RGB input modalities.
- **Biometric Evaluation & Decision Calibration**:
  - Calculated ROC curves, Equal Error Rate (EER) operating regions, optimal verification thresholds, FAR (False Acceptance Rate), and FRR (False Rejection Rate).

### 📊 Key Results & Impact
- **EfficientNet-B0 + CBAM + ArcFace Performance**:
  - **AUC-ROC Score**: **0.8872** (~89% discriminative capability).
  - **False Acceptance Rate (FAR)**: **11.19%** at calibrated optimal threshold `0.2111`.
  - **False Rejection Rate (FRR)**: **18.18%**.
- **Embedding Space Compactness**: t-SNE projections validate high intra-identity cohesion and wide inter-identity boundaries.
- **Edge Deployment Feasibility**: MobileNetV3-Large yields sub-30ms verification speeds suitable for real-time edge devices.

---

### 📂 Repository Structure / Structure du Projet
```text
Face-Verification-101-Project/
├── Pretraitement/
│   └── 01_Cleaning_Crop.ipynb              # Face detection, landmark alignment & crop
├── EfficientNet/
│   ├── EfficientNet-B0 + CBAM avec ArcFace.ipynb
│   └── results/                            # ROC, t-SNE, model_metrics.json (AUC: 0.8872)
├── MobileNet/
│   ├── MobileNetV3-Large + Batch Hard Triplet Loss.../
│   └── MobileNetV3-Large + Batch Hard Triplet Loss + GRAYSCALE/
├── ResNet/
│   └── Resnet.ipynb                        # Baseline residual network comparison
└── README.md                               # Project documentation
```