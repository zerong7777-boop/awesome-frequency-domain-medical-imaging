# Awesome Frequency-Domain Methods for Medical Imaging

**Last updated:** 2025-12-18  


---

## Table of Contents
- [0. Scope & How to Use](#0-scope--how-to-use)
- [0.1 Tag Legend](#01-tag-legend)
  - [Frequency Transform Tags](#frequency-transform-tags)
  - [Injection / Usage Tags](#injection--usage-tags)
  - [Modality / Dim Tags](#modality--dim-tags)
  - [Entry Template](#entry-template)
- [1. Segmentation](#1-segmentation)
  - [1.1 CNN-based](#11-cnn-based)
  - [1.2 ViT-based](#12-vit-based)
  - [1.3 Mamba / SSM-based](#13-mamba--ssm-based)
  - [1.4 Hybrid](#14-hybrid)
  - [1.5 Other Backbones](#15-other-backbones)
  - [1.6 Backbone-agnostic / Plug-in Modules](#16-backbone-agnostic--plug-in-modules)
- [2. Reconstruction & Super-Resolution](#2-reconstruction--super-resolution)
  - [2.1 CNN-based](#21-cnn-based)
  - [2.2 ViT-based](#22-vit-based)
  - [2.3 Mamba / SSM-based](#23-mamba--ssm-based)
  - [2.4 Hybrid](#24-hybrid)
  - [2.5 Other Backbones](#25-other-backbones)
  - [2.6 Backbone-agnostic / Plug-in Modules](#26-backbone-agnostic--plug-in-modules)
  - [2.7 K-space / Complex-specific](#27-k-space--complex-specific)
- [3. Denoising / Enhancement / Artifact Reduction](#3-denoising--enhancement--artifact-reduction)
  - [3.1 CNN-based](#31-cnn-based)
  - [3.2 ViT-based](#32-vit-based)
  - [3.3 Mamba / SSM-based](#33-mamba--ssm-based)
  - [3.4 Hybrid](#34-hybrid)
  - [3.5 Other Backbones](#35-other-backbones)
  - [3.6 Backbone-agnostic / Plug-in Modules](#36-backbone-agnostic--plug-in-modules)
- [4. Registration / Motion / Deformation](#4-registration--motion--deformation)
  - [4.1 CNN-based](#41-cnn-based)
  - [4.2 ViT-based](#42-vit-based)
  - [4.3 Mamba / SSM-based](#43-mamba--ssm-based)
  - [4.4 Hybrid](#44-hybrid)
  - [4.5 Other Backbones](#45-other-backbones)
- [5. Classification / Detection / Diagnosis](#5-classification--detection--diagnosis)
  - [5.1 CNN-based](#51-cnn-based)
  - [5.2 ViT-based](#52-vit-based)
  - [5.3 Mamba / SSM-based](#53-mamba--ssm-based)
  - [5.4 Hybrid](#54-hybrid)
  - [5.5 Other Backbones](#55-other-backbones)
- [6. Self-supervised / Pretraining / Domain Generalization](#6-self-supervised--pretraining--domain-generalization)
  - [6.1 Frequency-based SSL objectives](#61-frequency-based-ssl-objectives)
  - [6.2 Frequency augmentation & invariance](#62-frequency-augmentation--invariance)
  - [6.3 Robustness under domain shift (DG / DA / Federated)](#63-robustness-under-domain-shift-dg--da--federated)
- [7. Foundation Models](#7-foundation-models)
  - [7.1 Segmentation FM + Frequency Adapters](#71-segmentation-fm--frequency-adapters)
  - [7.2 Reconstruction / Diffusion FM + Frequency Constraints](#72-reconstruction--diffusion-fm--frequency-constraints)
  - [7.3 VLM / Prompting + Frequency Priors](#73-vlm--prompting--frequency-priors)
- [8. Datasets & Benchmarks](#8-datasets--benchmarks)
- [9. Implementation Notes](#9-implementation-notes)
- [10. Index](#10-index)
  - [10.1 Index by Frequency Transform](#101-index-by-frequency-transform)
  - [10.2 Index by Backbone](#102-index-by-backbone)
  - [10.3 Index by Injection / Usage](#103-index-by-injection--usage)
- [11. Contributing](#11-contributing)
- [12. Citation](#12-citation)


---

## 0. Scope How to Use
This repository curates **frequency-domain (spectral) methods for medical imaging**, including **transforms, plug-in modules, losses, and augmentations** used across tasks such as segmentation, reconstruction/SR, denoising/enhancement, registration, and classification.

**How to use**
- Browse by **task** (main sections), then by **backbone** (CNN / ViT / Mamba-SSM / Hybrid / Other).
- Use the tags in each entry to filter by:
  - **transform** (e.g., `[FFT] [DWT] [DCT]`),
  - **usage/injection** (e.g., `[Input] [Feature] [Loss] [Selection] [Fusion]`),
  - optional **modality/dim** (e.g., `[MRI] [CT] [US]`, `[2D] [3D]`).
- For transform- or module-centric reading, jump to the **Index** (by Transform / Backbone / Injection).

---

## Included / Excluded

**Included**
- Frequency transforms: FFT / DWT / DCT / STFT / Shearlet / Curvelet, plus **learnable spectral operators**.
- Usage patterns: input-/feature-level branches, frequency-aware attention/mixers, spectral losses/regularizers, frequency augmentation, **adaptive selection/routing**, and **multi-transform fusion**.
- Medical imaging tasks: segmentation, reconstruction/SR (incl. MRI/k-space when applicable), denoising/enhancement, registration, classification/detection, SSL/domain generalization, and foundation-model adapters/constraints.

**Excluded (by default)**
- Pure CV frequency papers **without medical imaging experiments** (except a few “Start Here” references).
- Untraceable engineering notes (no paper/tech report, unclear method).
- Works where “frequency” is only a vague description without an explicit spectral operator.

**Start Here (TODO)**
- Add 5–10 essential reads with one-line rationale + tags.

---

## 0.1 Tag Legend

### Frequency Transform Tags
- `[FFT]` Fourier transform / spectrum
- `[DWT]` Discrete Wavelet Transform
- `[DCT]` Discrete Cosine Transform
- `[STFT]` Short-time Fourier Transform
- `[Shearlet]` Shearlet transform
- `[Curvelet]` Curvelet transform
- `[FourierFeatures]` Fourier feature mapping / positional encoding
- `[SpectralConv]` Spectral convolution (FFT-based conv, Fourier layers)
- `[LearnableFreq]` Learnable frequency filters / adaptive spectral gating
- `[HybridFreq]` Hybrid transforms / mixed spectral operators
- `[DTCWT]` Dual-Tree Complex Wavelet Transform
- `[LaplacianPyr]` Laplacian pyramid / frequency decomposition (pyramid-based)

  
### Injection / Usage Tags
- `[Input]` input-level transform / pre-processing
- `[Feature]` feature-level spectral branch / frequency-path
- `[Attention]` spectral attention / frequency-aware attention
- `[TokenMix]` spectral token mixing / mixer in frequency domain
- `[Loss]` spectral loss / frequency regularization
- `[Aug]` frequency-domain augmentation (masking, mixing, jitter)
- `[Complex]` complex-valued processing
- `[K-space]` k-space operations (MRI)
- `[MultiScale]` multi-scale pyramid / wavelet decomposition
- `[Selection]` adaptive transform selection / routing (MoE, gating, Gumbel, etc.)
- `[Fusion]` multi-transform fusion / collaboration (e.g., FFT + DCT + DWT)
- `[Explain]` interpretability / spectral attribution / frequency response analysis
- `[Consistency]` frequency-domain consistency regularization (SSL/UDA/DG)
- `[Prompt]` frequency-domain prompting (e.g., Fourier-space prompts)


### Modality / Dim Tags
- Modalities: `[MRI] [CT] [US] [Xray] [PET] [Microscopy]`
- Dimensions: `[2D] [3D]`
- Data regime (optional): `[LowDose] [FastMRI] [FewShot] [CrossSite]`

### Entry Template
- **Paper Title** (Venue, Year) — one-line takeaway.  
  `Tags:` [FFT][Feature][Fusion][MRI][3D] | `Task:` Seg | `Backbone:` ViT  
  `Code:` (link) | `Data:` (datasets) | `Notes:` (ablation/highlight)

---

## 1. Segmentation

### 1.1 CNN-based
- **Wavelet U-Net for Medical Image Segmentation** (MICCAI, 2020) — Replaces pooling/upsampling with DWT/IWT to reduce information loss.  
  `Tags:` [DWT][Feature][MultiScale] | `Task:` Seg | `Backbone:` CNN/U-Net  
  [[MICCAI 2020](https://dl.acm.org/doi/10.1007/978-3-030-61609-0_63)]

- **GFUNet: A Global-Frequency-Domain Network for Medical Image Segmentation** (Computers in Biology and Medicine, 2023) — Uses Fourier-domain/global filtering to improve efficiency and accuracy in UNet-style segmentation.  
  `Tags:` [FFT][Feature] | `Task:` Seg | `Backbone:` CNN/U-Net  
  [[Comput Biol Med 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007552)] [[PubMed](https://pubmed.ncbi.nlm.nih.gov/37579584/)]
  
- **SASAN: Spectrum-Axial Spatial Approach Networks for Medical Image Segmentation** (IEEE TMI, 2024) — Introduces a spectrum branch to capture frequency information alongside axial spatial modeling.  
  `Tags:` [FFT][Feature][MultiScale] | `Task:` Seg | `Backbone:` CNN  
  [[IEEE TMI 2024](https://pure.bit.edu.cn/en/publications/spectrum-axial-spatial-approach-networks-for-medical-image-segmentation)] [[Code](https://github.com/IMOP-lab/SASAN-Pytorch)]

- **PFESA: FFT-based Parameter-Free Edge and Structure Attention** (MICCAI, 2025) — Parameter-free FFT decoupling for edge (high-freq) vs structure (low-freq) to improve skip connections.  
  `Tags:` [FFT][Feature][Explain] | `Task:` Seg | `Backbone:` CNN/U-Net  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3694_paper.pdf)]

- **FFTMed: Leveraging Fast Fourier Transform for a Lightweight Medical Image Segmentation Network** (Scientific Reports, 2025) — U-shaped network operating in Fourier domain with frequency modules and anti-aliasing aggregation.  
  `Tags:` [FFT][Feature] | `Task:` Seg | `Backbone:` CNN/U-Net  
  [[Sci Rep 2025](https://www.nature.com/articles/s41598-025-21799-5)]

- **Exploring a Frequency-Domain Attention-Guided Cascade U-Net for Medical Image Segmentation** (Computers in Biology and Medicine, 2023) — Cascade design with frequency-domain attention modules for segmentation.  
  `Tags:` [FFT][Attention][Feature] | `Task:` Seg | `Backbone:` CNN/U-Net  
  [[Comput Biol Med 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523011137)]

- **Wavelet U-Net++ for Accurate Lung Nodule Segmentation** (Biomedical Signal Processing and Control, 2024) — Combines U-Net++ with wavelet operations for better boundary/detail recovery.  
  `Tags:` [DWT][Feature][MultiScale] | `Task:` Seg | `Backbone:` CNN/U-Net++  
  [[Biomed Signal Process Control 2024](https://www.sciencedirect.com/science/article/abs/pii/S1746809423009424)]


### 1.2 ViT-based
- **WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation** (MICCAI, 2025) — Uses DWT partitioning (low/high sub-bands) and inverse wavelet upsampling for efficient 3D segmentation.  
  `Tags:` [DWT][TokenMix][MultiScale] | `Task:` Seg | `Backbone:` ViT/Transformer  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/1014-Paper4968.html)] [[arXiv 2025](https://arxiv.org/abs/2503.23764)]

- **FreqFiT: Boosting Parameter-Efficient Foundation Model Adaptation via Frequency-based Fine-Tuning** (MICCAI, 2025) — Inserts a frequency-based fine-tuning module between ViT blocks for better adaptation in 2D/3D medical segmentation.  
  `Tags:` [FFT][Feature][Prompt] | `Task:` Seg | `Backbone:` ViT/Foundation  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3066_paper.pdf)]

- **EFMS-Net: Efficient Frequency-Enhanced Multi-Scale Network for Medical Image Segmentation** (MICCAI, 2025) — Frequency-enhanced multi-scale design for CT/MRI segmentation.  
  `Tags:` [FFT][Feature][MultiScale] | `Task:` Seg | `Backbone:` ViT/CNN-Hybrid  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3331_paper.pdf)]
  
- **Robust Appearance-Mixed Learning with Frequency Domain Interpolation for Domain Generalization in Medical Image Segmentation (RAM)** (ECCV, 2022) — Mixes appearance in frequency domain to improve cross-domain robustness.  
  `Tags:` [FFT][Aug][Consistency] | `Task:` Seg | `Backbone:` CNN/Transformer  
  [[ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950102.pdf)]

- **PFD-Net: Pyramid Fourier Deformable Network for Medical Image Segmentation** (Computers in Biology and Medicine, 2024) — Combines deformable attention with pyramid Fourier modules for multi-scale context.  
  `Tags:` [FFT][Feature][MultiScale] | `Task:` Seg | `Backbone:` Transformer  
  [[Comput Biol Med 2024](https://pubmed.ncbi.nlm.nih.gov/39546233/)]


### 1.3 Mamba / SSM-based
- **WMC-Net: Wavelet-Enhanced Mamba with Contextual Fusion Network for Medical Image Segmentation** (Knowledge-Based Systems, 2025) — Enhances Mamba/SSM segmentation with wavelet-based frequency cues and contextual fusion.  
  `Tags:` [DWT][Feature][Fusion] | `Task:` Seg | `Backbone:` Mamba/SSM  
  [[Knowl-Based Syst 2025](https://www.sciencedirect.com/science/article/abs/pii/S0950705125021690)]

- **EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation** (MICCAI, 2024) — Introduces frequency-aware channel mixing into 3D Mamba blocks.  
  `Tags:` [FFT][Feature] | `Task:` Seg | `Backbone:` Mamba/SSM  
  [[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1923_paper.pdf)] [[arXiv 2024](https://arxiv.org/abs/2409.17675)]

- **HybridMamba: A Dual-domain Mamba for 3D Medical Image Segmentation** (MICCAI, 2025) — Dual-domain design coupling spatial and frequency representations in Mamba-style sequence modeling.  
  `Tags:` [FFT][Fusion] | `Task:` Seg | `Backbone:` Mamba/Hybrid  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2815_paper.pdf)]

- **BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency Based Feature Enhancement for Brain Tumor Segmentation** (MICCAI, 2025) — UNet-style Mamba backbone with dual-band frequency enhancement modules.  
  `Tags:` [FFT][Feature][MultiScale] | `Task:` Seg | `Backbone:` Mamba/U-Net  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/0487_paper.pdf)]


### 1.4 Hybrid
- **WMREN: Wavelet Multi-scale Region-Enhanced Network for Medical Image Segmentation** (IJCAI, 2025) — Collaborative downsampling combining wavelet transform and CNN for multi-scale feature retention.  
  `Tags:` [DWT][Fusion][MultiScale] | `Task:` Seg | `Backbone:` Hybrid (CNN + Wavelet)  
  [[IJCAI 2025](https://www.ijcai.org/proceedings/2025/0187.pdf)]

- **UWT-Net: Mining Low-Frequency Feature Information for Medical Image Segmentation** (MICCAI, 2025) — Wavelet-transform-driven frequency decomposition to mine low-frequency structural cues.  
  `Tags:` [DWT][Feature][MultiScale] | `Task:` Seg | `Backbone:` Hybrid  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/1637_paper.pdf)]

- **Frequency-domain Multi-modal Fusion for Language-guided Medical Image Segmentation (FMISeg)** (MICCAI, 2025) — Frequency-domain interaction for language-guided medical segmentation.  
  `Tags:` [FFT][Fusion][Feature] | `Task:` Seg | `Backbone:` Hybrid / Vision-Language  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3678_paper.pdf)]


### 1.5 Other Backbones
- **Adaptive Wavelet-VNet for Single-Sample Test-Time Adaptation** (IEEE TMI, 2024) — Test-time adaptation that leverages wavelet cues for robustness under distribution shift.  
  `Tags:` [DWT][Aug][Consistency] | `Task:` Seg | `Backbone:` V-Net (3D CNN)  
  [[IEEE TMI 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11656288/)]

- **Active Contour Model Combining Frequency Domain Information for Medical Image Segmentation** (Pattern Recognition, 2025) — Classical active contour segmentation enhanced with Fourier/frequency-domain information.  
  `Tags:` [FFT][Explain] | `Task:` Seg | `Backbone:` Classical (Active Contour)  
  [[Pattern Recognit 2025](https://www.sciencedirect.com/science/article/abs/pii/S0031320325007861)]


### 1.6 Backbone-agnostic / Plug-in Modules (SSL / UDA / DG / PEFT)
- **FRCNet: Frequency and Region Consistency for Semi-Supervised Medical Image Segmentation** (MICCAI, 2024) — Adds frequency-domain consistency and multi-granularity region similarity constraints on top of a base segmentor.  
  `Tags:` [FFT][Consistency][Loss] | `Task:` Seg | `Backbone:` Plug-in (SSL)  
  [[MICCAI 2024](https://papers.miccai.org/miccai-2024/340-Paper0245.html)]

- **AdaptFRCNet: Semi-supervised Adaptation of Pre-trained Model with Frequency and Region Consistency** (Medical Image Analysis, 2025) — Extends FRC-style constraints to adapt pre-trained models with limited annotations.  
  `Tags:` [FFT][Consistency][Loss] | `Task:` Seg | `Backbone:` Plug-in (Adaptation)  
  [[Med Image Anal 2025](https://www.sciencedirect.com/science/article/abs/pii/S1361841525001732)]

- **Generalizable Medical Image Segmentation via Random Amplitude Mixup (RAM)** (ECCV, 2022) — Fourier transform on source images and mixing low-frequency amplitudes to improve domain generalization.  
  `Tags:` [FFT][Aug][Consistency] | `Task:` Seg | `Backbone:` Plug-in (DG)  
  [[ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810415.pdf)]

- **FVP: Fourier Visual Prompting for Source-Free UDA of Medical Image Segmentation** (IEEE TMI, 2023) — Learns a low-frequency Fourier-space visual prompt to steer a frozen model on target-domain data.  
  `Tags:` [FFT][Prompt][Consistency] | `Task:` Seg | `Backbone:` Plug-in (SFUDA)  
  [[IEEE TMI 2023](https://doi.org/10.1109/TMI.2023.3306105)] [[arXiv 2023](https://arxiv.org/abs/2304.13672)]

- **Curriculum-Based Augmented Fourier Domain Adaptation (CAFDA)** (arXiv, 2023) — Curriculum-based Fourier amplitude transfer for robust medical image segmentation across domains.  
  `Tags:` [FFT][Aug][Consistency] | `Task:` Seg | `Backbone:` Plug-in (DG/DA)  
  [[arXiv 2023](https://arxiv.org/pdf/2306.03511)]

- **Improving Medical Image Segmentation with Implicit Representations (HFNM uses wavelet decomposition)** (MICCAI, 2025) — Includes a high-frequency module that decomposes images via wavelets and perturbs high-frequency components to regularize training.  
  `Tags:` [DWT][Loss][Explain] | `Task:` Seg | `Backbone:` Plug-in / Hybrid  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/0665_paper.pdf)]


---

## 2. Reconstruction & Super-Resolution

### 2.1 CNN-based

- **Wavelet-Based Enhanced Medical Image Super Resolution (WMSR)** — Early CNN + DWT framework that learns in the wavelet domain and fuses sub-bands for SR of CT/MRI slices.  
  `Tags:` [DWT][Wavelet][MultiScale] | `Task:` SR | `Modality:` CT/MRI | `Backbone:` CNN  
  [[IEEE Access 2020](https://scholar.google.com/scholar?q=Wavelet-Based+Enhanced+Medical+Image+Super+Resolution)]

- **Wavelet-Based Medical Image Super Resolution Using Cross-Connected Residual-in-Dense Grouped CNN** — Extends residual-in-dense CNN with wavelet-domain decomposition and cross-connected groups to better capture multi-frequency details.  
  `Tags:` [DWT][Wavelet][MultiScale] | `Task:` SR | `Modality:` CT/MRI | `Backbone:` CNN  
  [[J Vis Commun Image Represent 2020](https://doi.org/10.1016/j.jvcir.2020.102819)]

- **A Super-Resolution Network for Medical Imaging via Transformation Analysis of Wavelet Multi-Resolution (WMRSR)** — Builds a convolution-based wavelet module and explicitly fuses spatial-domain and wavelet-domain features under a multi-resolution framework.  
  `Tags:` [DWT][Wavelet][MultiDomain][MultiScale] | `Task:` SR | `Modality:` CT/MRI | `Backbone:` CNN  
  [[Neural Networks 2023](https://doi.org/10.1016/j.neunet.2023.07.005)]

- **Deep Learning-based CT Image Super-Resolution via Wavelet Embedding** — Uses wavelet embedding for CT SR, emphasizing high- vs low-frequency components through learned subband fusion.  
  `Tags:` [DWT][Wavelet][Embedding] | `Task:` SR | `Modality:` CT | `Backbone:` CNN  
  [[Radiat Phys Chem 2023](https://scholar.google.com/scholar?q=Deep+learning-based+computed+tomographic+image+super-resolution+via+wavelet+embedding)]

- **A Super-Resolution Network for Medical Imaging via Transformation Analysis of Wavelet Multi-Resolution** — Another wavelet-multi-resolution variant that jointly processes spatial and wavelet-domain inputs to exploit low-/high-frequency correlations.  
  `Tags:` [DWT][Wavelet][MultiDomain][MultiScale] | `Task:` SR | `Modality:` General Med | `Backbone:` CNN  
  [[Neural Networks 2024](https://scholar.google.com/scholar?q=Analysis+of+medical+images+super-resolution+via+a+wavelet+pyramid+recursive+neural+network)]


### 2.2 ViT-based

- **WavTrans: Synergizing Wavelet and Cross-Attention Transformer for Multi-contrast MRI Super-Resolution** — Combines wavelet-domain decomposition with cross-attention Transformer blocks for arbitrary-scale SR on multi-contrast MRI.  
  `Tags:` [DWT][Wavelet][Transformer][MultiDomain] | `Task:` SR | `Modality:` MRI | `Backbone:` ViT/Transformer  
  [[MICCAI 2022](https://scholar.google.com/scholar?q=WavTrans:+Synergizing+wavelet+and+cross-attention+transformer+for+multi-contrast+MRI+super-resolution)]

- **Cross-Fusion Adaptive Feature Enhancement Transformer (CFAFET) for Brain MRI Super-Resolution** — Transformer that explicitly models high-frequency components via cross-fusion and sparse attention to enhance fine anatomical details.  
  `Tags:` [HighFreq][Transformer][Attention] | `Task:` SR | `Modality:` MRI | `Backbone:` ViT/Transformer  
  [[Comput Methods Programs Biomed 2025](https://scholar.google.com/scholar?q=Cross-fusion+adaptive+feature+enhancement+transformer+brain+MRI+super-resolution)]


### 2.3 Mamba / SSM-based

- **FGMamba: Versatile and Efficient Medical Image Super-Resolution via Frequency-Gated Mamba** — Uses FFT-based frequency gating inside Mamba blocks to decouple and fuse low-/high-frequency cues across scales, achieving efficient SR on multiple modalities.  
  `Tags:` [FFT][FrequencyGate][SSM][MultiScale] | `Task:` SR | `Modality:` CT/MRI/Fundus | `Backbone:` Mamba/SSM  
  [[arXiv 2025](https://arxiv.org/abs/2510.27296)]

- **Deform-Mamba Network for MRI Super-Resolution** — Deformable-convolution Mamba encoder that captures long-range dependencies with SSM while preserving local structures; typically used as a strong non-frequency baseline for FDM-type methods.  
  `Tags:` [SSM][DeformConv][Baseline] | `Task:` SR | `Modality:` MRI | `Backbone:` Mamba/SSM  
  [[Preprint / MICCAI 2024](https://scholar.google.com/scholar?q=Deform-Mamba+Network+for+MRI+Super-Resolution)]

- **Global–Local Mamba Network for Multi-Modality Medical Image Super-Resolution (GLMamba)** — Exploits global (state-space) and local (convolutional) branches, often paired with frequency-aware losses to sharpen fine structures.  
  `Tags:` [SSM][GlobalLocal][HighFreq] | `Task:` SR | `Modality:` Multi-modal | `Backbone:` Mamba/SSM  
  [[Pattern Recognit 2025](https://scholar.google.com/scholar?q=Global+and+Local+Mamba+Network+for+Multi-Modality+Medical+Image+Super-Resolution)]


### 2.4 Hybrid (Spatial + Frequency / Multi-Branch)

- **Synthesized 7T MRI from 3T MRI via Deep Learning in Spatial and Wavelet Domains** — Two-branch network operating in spatial and wavelet domains, jointly reconstructing 7T-like MR volumes from 3T scans.  
  `Tags:` [DWT][Wavelet][MultiDomain][SR] | `Task:` SR | `Modality:` MRI (3T→7T) | `Backbone:` CNN (Dual-branch)  
  [[Med Image Anal 2020](https://doi.org/10.1016/j.media.2019.101663)]

- **A Novel Hybrid GAN for CT and MRI Super-Resolution Reconstruction** — Hybrid frequency–spatial GAN: complex residual U-Net in the Fourier domain + enhanced residual U-Net in image domain, trained with frequency-domain and perceptual losses.  
  `Tags:` [FFT][Complex][GAN][MultiDomain] | `Task:` SR | `Modality:` CT/MRI | `Backbone:` CNN+GAN  
  [[Phys Med Biol 2023](https://pubmed.ncbi.nlm.nih.gov/37285848/)]

- **Deep Cascade of Wavelet-Based CNNs for MR Image Reconstruction (DC-WCNN)** — Multi-stage cascade where each stage performs wavelet-domain refinement plus image-domain correction, effectively unrolling a multi-resolution reconstruction process.  
  `Tags:` [DWT][Wavelet][Cascade] | `Task:` Recon | `Modality:` MRI | `Backbone:` CNN (Unrolled)  
  [[ISBI 2020](https://scholar.google.com/scholar?q=Deep+cascade+of+wavelet+based+CNNs+for+MR+image+reconstruction)]


### 2.5 Other Backbones (Diffusion / GAN / Etc.)

- **High-Frequency Space Diffusion Model for Accelerated MRI** — Diffusion model that operates in a learned high-frequency space to reconstruct under-sampled MRI with improved texture and edge fidelity.  
  `Tags:` [HighFreq][Diffusion][Kspace] | `Task:` Recon | `Modality:` MRI | `Backbone:` Diffusion  
  [[IEEE TMI 2024](https://scholar.google.com/scholar?q=High-frequency+space+diffusion+model+for+accelerated+MRI)]

- **DISGAN: Wavelet-informed Discriminator Guides GAN to MRI Super-Resolution with Noise Cleaning** — Uses wavelet-domain inputs in the discriminator to better distinguish high-frequency anatomical details from artifacts, improving GAN-based SR.  
  `Tags:` [DWT][Wavelet][GAN][NoiseClean] | `Task:` SR | `Modality:` MRI | `Backbone:` GAN  
  [[ICCV 2023](https://scholar.google.com/scholar?q=DISGAN:+wavelet-informed+discriminator+guides+GAN+to+MRI+super-resolution+with+noise+cleaning)]

- **Generative Super-Resolution PET Imaging with Fourier Diffusion Models** — Fourier-domain conditional diffusion that models PET images in frequency space to generate SR PET from low-dose/low-res inputs.  
  `Tags:` [FFT][Diffusion][PET] | `Task:` SR | `Modality:` PET | `Backbone:` Diffusion  
  [[SPIE Med Imaging 2025](https://scholar.google.com/scholar?q=Generative+Super-Resolution+PET+Imaging+with+Fourier+Diffusion+Models)]


### 2.6 Backbone-agnostic / Plug-in Modules

- **Fourier Convolution Block (FCB) with Global Receptive Field for MRI Reconstruction** — Plug-in Fourier convolution layer that replaces/augments standard convolutions to capture global context in k-space while remaining compatible with U-Net-style recon networks.  
  `Tags:` [FFT][Plugin][Global][Conv] | `Task:` Recon | `Modality:` MRI | `Backbone:` Any (U-Net / Unrolled / GAN)  
  [[Med Image Anal 2025](https://scholar.google.com/scholar?q=Fourier+Convolution+Block+with+global+receptive+field+for+MRI+reconstruction)]

- **Memory-Enhanced Multi-domain DUN with Frequency-Domain Consistency Learning** — Deep unrolling framework that adds a frequency-domain consistency module; can be viewed as a spectral regulariser attachable to other learned recon pipelines.  
  `Tags:` [FFT][Consistency][Unrolled][Plugin] | `Task:` Recon | `Modality:` MRI | `Backbone:` Unrolled / Any  
  [[IEEE JBHI 2025](https://scholar.google.com/scholar?q=memory-enhanced+multi-domain+learning-based+DUN+network+frequency-domain+consistency+medical+image+reconstruction)]


### 2.7 K-space / Complex-specific

- **k-Space Deep Learning for Accelerated MRI** — Pioneering work that directly learns a mapping from sub-sampled k-space to fully-sampled k-space, followed by image-domain conversion, explicitly modelling aliasing in the frequency domain.  
  `Tags:` [Kspace][FFT][Complex][DataConsistency] | `Task:` Recon | `Modality:` MRI | `Backbone:` CNN  
  [[IEEE TMI 2020](https://scholar.google.com/scholar?q=k-Space+Deep+Learning+for+Accelerated+MRI)]

- **Undersampled MRI Reconstruction Based on Spectral Graph Wavelet Transform** — Designs a graph-based k-space model where spectral graph wavelet transform recovers missing k-space samples using graph spectral priors.  
  `Tags:` [GraphWavelet][Kspace][Sparse] | `Task:` Recon | `Modality:` MRI | `Backbone:` Graph/CNN  
  [[Comput Biol Med 2023](https://doi.org/10.1016/j.compbiomed.2023.106644)]

- **AFTNet: Artificial Fourier Transform Network for Deep Learning-Based MRI Reconstruction** — Complex-valued network that learns an “artificial” Fourier transform, operating directly on k-space and image space to improve reconstruction of under-sampled MRI.  
  `Tags:` [FFT][Complex][Kspace][MultiDomain] | `Task:` Recon | `Modality:` MRI | `Backbone:` CNN  
  [[Comput Biol Med 2025](https://doi.org/10.1016/j.compbiomed.2025.108711)]

- **Super-Resolution MRI Using Phase-Scrambling Fourier Transform Imaging and Unrolling Model-Based Network** — Integrates phase-scrambling Fourier imaging with an unrolled deep network, effectively performing SR by modelling the acquisition physics in k-space.  
  `Tags:` [FFT][PSFT][Unrolled][SR] | `Task:` Recon/SR | `Modality:` MRI | `Backbone:` Model-based + CNN  
  [[IEEE Access 2023](https://scholar.google.com/scholar?q=super-resolution+for+MRI+using+phase-scrambling+Fourier+transform+imaging+and+unrolling+model-based+network)]

---

## 3. Denoising / Enhancement / Artifact Reduction

### 3.1 CNN-based

- **Deep Convolutional Framelet Denoising for Low-Dose CT via Wavelet Residual Network** — Interprets CNN filters as multi-layer framelets and embeds non-subsampled wavelet transforms to suppress LDCT noise while preserving fine structures.  
  `Tags:` [DWT][Framelet][Feature] | `Task:` Denoise (LDCT) | `Backbone:` CNN  
  `Links:` [[IEEE TMI 2018](https://ieeexplore.ieee.org/document/8264712) ] 

- **Wavelet Subband-Specific Learning for Low-Dose CT Image Denoising** — Uses stationary wavelet transform (SWT) to decompose LDCT images into subbands and trains subband-specific CNN branches with an additional frequency-domain loss to avoid over-smoothing and better restore texture.  
  `Tags:` [SWT][Subband][Loss] | `Task:` Denoise (LDCT) | `Backbone:` CNN  
  `Links:` [[PLOS ONE 2022](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274308) ] 

- **Low-Dose CT Image Denoising Using DWT–Anisotropic Gaussian Filter–Based Denoising CNN** — Combines discrete wavelet transform and anisotropic Gaussian filtering with a CNN denoiser, explicitly separating frequency components before deep reconstruction.  
  `Tags:` [DWT][Preproc][Hybrid] | `Task:` Denoise (LDCT) | `Backbone:` CNN  
  `Links:` [[Applied Sciences 2023](https://www.sciencedirect.com/science/article/pii/S0895611123000498) ] 

- **Frequency Feature Enhancement Multi-level U-Net for Low-Dose CT** — Adds FFT-based frequency-domain loss and feature-enhancement blocks to a multi-level U-Net, encouraging restoration of missing high-frequency information in LDCT images.  
  `Tags:` [FFT][Feature][Loss] | `Task:` Denoise (LDCT) | `Backbone:` U-Net  
  `Links:` [[LNCS / MICCAI Workshop 2024](https://link.springer.com/chapter/10.1007/978-981-97-5600-1_15) ] 

- **FEUSNet: Fourier Embedded U-Shaped Network for Image Denoising** — General U-Net–style denoiser with Fourier-embedded residual blocks; widely cited in medical denoising work as a frequency-domain CNN baseline.  
  `Tags:` [FFT][Feature] | `Task:` Denoise (Generic / Med) | `Backbone:` CNN/U-Net  
  `Links:` [[Entropy 2023](https://www.mdpi.com/1099-4300/25/4/654) ]

- **FDDL-Net: Frequency Domain Decomposition Learning for Speckle Reduction in Ultrasound Images** — Decomposes convolutional feature maps into low-/high-frequency components via an interactive dual-branch CNN; median filtering in the high-frequency branch effectively removes ultrasound speckle while preserving structure.  
  `Tags:` [FFT][Feature][DualBranch] | `Task:` Denoise (US Speckle) | `Backbone:` CNN  
  `Links:` [[Multimedia Tools and Applications 2022](https://doi.org/10.1007/s11042-022-13481-z) ] 


### 3.2 ViT-based

- **FSformer: A Combined Frequency Separation Network and Transformer Model for LDCT Denoising** — Introduces a frequency separation network in front of a Transformer backbone and a compound loss with explicit frequency-domain term to improve LDCT noise removal and robustness.  
  `Tags:` [FFT][FreqSep][Loss] | `Task:` Denoise (LDCT) | `Backbone:` CNN + Transformer  
  `Links:` [[Computers in Biology and Medicine 2024](https://www.sciencedirect.com/science/article/abs/pii/S0010482524004621) ] 

- **Wavelet-Domain Frequency-Mixing Transformer Unfolding Network (WFTUNet) for LDCT Denoising** — Unrolls an optimization algorithm in the wavelet domain and injects transformer-style frequency-mixing blocks across scales to jointly denoise low-/high-frequency components.  
  `Tags:` [DWT][TokenMix][Unfolding] | `Task:` Denoise (LDCT) | `Backbone:` Transformer / Unrolled  
  `Links:` [[QIMS 2025](https://qims.amegroups.org/article/view/130142) ] 

- **Semi-SFTrans: Semi-Supervised Spatial–Frequency Transformer for Metal Artifact Reduction in Maxillofacial CT** — Builds a dual-branch spatial–frequency transformer where a frequency-pathway explicitly models Fourier-domain information to suppress streak artifacts in cone-beam CT, under limited labels.  
  `Tags:` [FFT][MultiDomain][SemiSup] | `Task:` Artifact (MAR) | `Backbone:` Transformer  
  `Links:` [[European Journal of Radiology 2025](https://www.sciencedirect.com/science/article/pii/S0720048X25002186) ] 

- **FD-DiT: Frequency Domain–Directed Diffusion Transformer for Low-Dose CT Reconstruction** — A diffusion–Transformer hybrid that guides the generative process via frequency-domain priors to recover fine anatomical details from LDCT scans.  
  `Tags:` [FFT][Diffusion][Transformer] | `Task:` Denoise / Enhance (LDCT) | `Backbone:` Diffusion Transformer  
  `Links:` [[CoRR 2025](https://arxiv.org/abs/2506.23466) ] 

### 3.3 Mamba / SSM-based

- **CT-Mamba: A Hybrid Convolutional State Space Model for Low-Dose CT Denoising** — Combines convolution and Mamba-style state space blocks, and designs a deep noise power spectrum (NPS) loss in the frequency domain to better match realistic CT noise textures.  
  `Tags:` [SSM][FFT][NPSLoss] | `Task:` Denoise (LDCT) | `Backbone:` CNN + Mamba  
  `Links:` [[arXiv 2025](https://arxiv.org/abs/2411.07930) ]

- **Wavelet-Enhanced Mamba for Photoacoustic Image Restoration** — Uses wavelet-enhanced residual optimal transport together with Mamba blocks to address limited-view artifacts and noise in photoacoustic tomography, explicitly manipulating multi-scale frequency content.  
  `Tags:` [DWT][SSM][Artifact] | `Task:` Denoise / Artifact (PAT) | `Backbone:` Mamba / Hybrid  
  `Links:` [[Photoacoustics 2025](https://www.sciencedirect.com/science/article/pii/S2213597925000632) ] 


### 3.4 Hybrid (Multi-domain / Unrolled / Model-based + Deep)

- **TDMAR-Net: A Frequency-Aware Tri-Domain Diffusion Network for CT Metal Artifact Reduction** — Diffusion model that simultaneously leverages priors in the projection, image, and Fourier domains to suppress metal artifacts and restore CT image quality.  
  `Tags:` [FFT][Projection][Diffusion] | `Task:` Artifact (MAR) | `Backbone:` Hybrid (Projection + Image + k-space)  
  `Links:` [[Physics in Medicine & Biology 2025](https://iopscience.iop.org/article/10.1088/1361-6560/ae0efc) ] 
- **Wavelet-Domain Frequency-Mixing Transformer Unfolding (WFTUNet)** — (Also listed above under ViT) can be viewed as a hybrid unrolled network that alternates wavelet-domain proximal steps with transformer-based frequency mixing, bridging classic iterative denoising and deep ViT modules.  
  `Tags:` [DWT][Unfolding][Hybrid] | `Task:` Denoise (LDCT) | `Backbone:` Model-based + Transformer  
  `Links:` [[QIMS 2025](https://qims.amegroups.org/article/view/130142) ]

- **Ca-ResUNet with Noise Power Spectrum (NPS) Loss for LDCT Enhancement** — A 3D cascaded ResUNet trained with a modified noise power spectrum loss computed in the Fourier domain, explicitly penalizing mismatched frequency content to reduce streaks and structured noise.  
  `Tags:` [FFT][NPSLoss][Artifact] | `Task:` Denoise / Enhance (LDCT) | `Backbone:` CNN (Cascaded)  
  `Links:` [[Phys Med Biol 2021](https://pubmed.ncbi.nlm.nih.gov/33992773/) ] 


### 3.5 Other Backbones (GAN / FNO / Physics-informed / Flow / Diffusion)

- **WGAN-DUS: Ultrasound Speckle Reduction Using Wavelet-Based Generative Adversarial Network** — Applies discrete wavelet transform to decompose ultrasound images into subbands and uses a GAN with wavelet reconstruction modules for real-time despeckling while preserving boundary contrast.  
  `Tags:` [DWT][GAN][Subband] | `Task:` Denoise (US Speckle) | `Backbone:` GAN  
  `Links:` [[IEEE JBHI 2022](https://pubmed.ncbi.nlm.nih.gov/35077370/) ] 

- **SELFNet: Denoising Shear Wave Elastography Using Spatial–Temporal Fourier Feature Networks** — Physics-informed neural network that injects random Fourier features in space–time to estimate and denoise displacement fields in shear wave elastography while enforcing governing PDE constraints.  
  `Tags:` [FFT][FourierFeatures][PINN] | `Task:` Denoise (US SWE) | `Backbone:` Physics-Informed NN  
  `Links:` [[Ultrasound in Medicine & Biology 2024](https://pubmed.ncbi.nlm.nih.gov/39317627/) ] 

- **Medical Image Joint Deringing and Denoising Using Fourier Neural Operator** — Uses Fourier Neural Operators operating in k-space to jointly remove Gibbs ringing and noise from MR images, learning integral operators directly in the frequency domain.  
  `Tags:` [FFT][FNO][Kspace] | `Task:` Denoise / Artifact (MRI Gibbs) | `Backbone:` FNO  
  `Links:` [[ICBSP 2023](https://ieeexplore.ieee.org/document/10380609) ] 

- **Frequency-Domain Flow Matching (FFM) for Real-World Ultra-Low-Dose Lung CT Denoising** — Constructs an image purification pipeline and a flow-matching generative model operating in the frequency domain to better preserve anatomical structure under severe noise and misalignment.  
  `Tags:` [FFT][Flow][Generative] | `Task:` Denoise (uLDCT) | `Backbone:` Flow / Generative  
  `Links:` [[arXiv 2025](https://arxiv.org/abs/2510.07492) ]

- **Wavelet-Improved Score-Based Generative Model for Medical Imaging** — Incorporates wavelet transforms into a score-based diffusion model to more flexibly control multi-scale frequency content in medical image restoration tasks (denoising, enhancement).  
  `Tags:` [DWT][Diffusion][Generative] | `Task:` Denoise / Enhance (Multi-modality) | `Backbone:` Score-based Diffusion  
  `Links:` [[IEEE TMI 2024](https://pubmed.ncbi.nlm.nih.gov/38469054/) ] 


### 3.6 Backbone-agnostic / Plug-in Modules (Losses / Training Strategies)

- **Hybrid Loss Function with High-Frequency Information Loss for Low-Dose CT Denoising** — Proposes a hybrid objective combining a weighted patch-wise MAE and a high-frequency information loss (HFLoss) defined in the frequency domain; applicable to various LDCT networks.  
  `Tags:` [FFT][Loss][Patch] | `Task:` Denoise (LDCT) | `Backbone:` Any (plug-in loss)  
  `Links:` [[J Appl Clin Med Phys 2023](https://pubmed.ncbi.nlm.nih.gov/37571834/) ] 

- **Network and Frequency Separation Training (NFST) for Low-Dose CT Denoising** — Introduces a training scheme that decouples network optimization from frequency separation, using a dedicated high-frequency loss to better reconstruct fine structures; can be attached to different LDCT backbones.  
  `Tags:` [FFT][FreqSep][Training] | `Task:` Denoise (LDCT) | `Backbone:` Any (training strategy)  
  `Links:` [[SPIE Medical Imaging 2025](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13683/136830U) ] 

- **Frequency-Domain Structure Losses for CycleGAN-Based CT Enhancement** — Adds Fourier-domain structure-preserving losses to adversarial image-to-image translation (CycleGAN) to better maintain anatomical details when translating between low-dose and standard-dose CT domains.  
  `Tags:` [FFT][Loss][GAN] | `Task:` Enhance / Denoise (CT) | `Backbone:` GAN (CycleGAN, etc.)  
  `Links:` [[Sensors 2023](https://www.mdpi.com/1424-8220/23/3/1089) ] 

- **Frequency-Aware Losses in LDCT Networks (Surveyed in Kim et al., 2024)** — Systematic review highlighting how Fourier or wavelet transforms are used to define structure- or texture-aware losses (e.g., NPS-based, spectrum loss) that can be plugged into many denoising/enhancement architectures.  
  `Tags:` [FFT][Wavelet][Survey] | `Task:` Denoise / Enhance (CT) | `Backbone:` Any  
  `Links:` [[Medical Physics / Systematic Review 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11502640/) ] 


---

## 4. Registration / Motion / Deformation

### 4.1 CNN-based
- **Fourier-Net: Fast Image Registration with Band-Limited Deformation** (AAAI, 2023) — U-Net–style encoder that predicts a low-dimensional, band-limited representation of the deformation field in the Fourier domain, decoded by a parameter-free FFT-based model-driven decoder for fast 2D/3D registration.  
  `Tags:` [FFT][KSpace][ModelDriven][3D] | `Task:` Reg | `Backbone:` CNN/U-Net  
  [[AAAI 2023](https://dl.acm.org/doi/10.1609/aaai.v37i1.25182)] [[arXiv](https://arxiv.org/abs/2211.16342)] [[Code](https://github.com/xi-jia/Fourier-Net)]

- **RegFSC-Net: Medical Image Registration via Fourier Transform With Spatial Reorganization and Channel Refinement Network** (IEEE J-BHI, 2024) — Extends band-limited Fourier deformable models with spatial feature reorganization and channel-refinement modules for accurate 3D brain registration.  
  `Tags:` [FFT][KSpace][Channel][3D] | `Task:` Reg | `Backbone:` CNN/U-Net  
  [[IEEE J-BHI 2024](https://dblp.org/rec/journals/titb/LiuHXSZZ24.html)]

- **Wavelet-Guided Multi-Scale ConvNeXt for Unsupervised Medical Image Registration (WaveMorph)** (Bioengineering, 2025) — ConvNeXt-based unsupervised registration; applies DWT at multiple scales, feeding low-frequency sub-images to a multi-scale network and using wavelet-derived constraints to refine local deformations.  
  `Tags:` [DWT][MultiScale][Unsupervised] | `Task:` Reg | `Backbone:` CNN/ConvNeXt  
  [[Bioengineering 2025](https://doi.org/10.3390/bioengineering12040406)]

- **WTDL-Net: Medical Image Registration Based on Wavelet Transform and Multi-Scale Deep Learning** (Journal of Supercomputing, 2025) — 3D CNN-based coarse-to-fine registration where multi-resolution wavelet low-frequency sub-images drive global alignment and high-frequency constraints preserve details.  
  `Tags:` [DWT][MultiScale][3D] | `Task:` Reg | `Backbone:` CNN  
  [[J Supercomputing 2025](https://doi.org/10.1007/s11227-025-07567-2)]


### 4.2 ViT-based
- **FractMorph: A Fractional Fourier-Based Multi-Domain Transformer for Deformable Image Registration** (arXiv, 2025) — 3D dual-stream Transformer where Fractional Cross-Attention branches apply fractional Fourier transforms at multiple angles plus a log-magnitude branch to jointly capture local, semi-global, and global matching cues.  
  `Tags:` [FrFT][FFT][MultiDomain][Transformer][3D] | `Task:` Reg | `Backbone:` ViT/Transformer + CNN head  
  [[arXiv 2025](https://arxiv.org/abs/2508.12445)] [[Code](https://github.com/shayankebriti/FractMorph)]

- **EFormer: Efficient Transformer for Medical Image Registration Based on Frequency Division and Board Attention** (Computer Science, 2024) — Transformer-based registration model that divides features into different frequency bands and applies “board attention” to balance low- vs high-frequency cues.  
  `Tags:` [FFT][FrequencyDivision][Transformer] | `Task:` Reg | `Backbone:` ViT/Transformer  
  [[Computer Science 2024](https://doaj.org/article/cbc720f8b5f04cb0abe694c919e59e4f)]


### 4.3 Mamba / SSM-based
- **MambaMorph: Mamba-Based Deformable Medical Image Registration with an Annotated Brain MR–CT Dataset** (Computerized Medical Imaging and Graphics, 2025) — Replaces CNN encoders with Mamba SSM blocks to better capture long-range dependencies in 3D MR–CT registration.  
  `Tags:` [SSM][Mamba][3D][MultiModal] | `Task:` Reg | `Backbone:` Mamba/SSM  
  [[CMIG 2025](https://doi.org/10.1016/j.compmedimag.2025.102566)] [[Code](https://github.com/Guo-Stone/MambaMorph)]

- **Associating Frequency-Aware MambaMorph and Diffusion for 4D Volumetric Image Synthesis** (SSRN, 2025) — Builds a frequency-aware MambaMorph-based registration model using spatially adaptive low-pass and high-pass filtering inside a diffusion pipeline to synthesize 4D radiotherapy volumes with realistic motion.  
  `Tags:` [FFT][Mamba][Diffusion][4D][Motion] | `Task:` Motion / 4D Reg | `Backbone:` Mamba/SSM + Diffusion  
  [[SSRN 2025](https://ssrn.com/abstract=5134459)]


### 4.4 Hybrid
- **Dual-Domain Framework for Multimodal Medical Image Registration via Local Phase Consistency and Gradient-Intensity Mutual Information** (Biomedical Signal Processing and Control, 2024) — Combines spatial-domain gradients with frequency-domain local phase consistency (log-Gabor/phase-based features) to improve multimodal registration robustness to intensity and contrast changes.  
  `Tags:` [FFT][Phase][MultiModal][Hybrid] | `Task:` Reg | `Backbone:` Hybrid (Hand-crafted + DL-ready)  
  [[Biomed Signal Process Control 2024](https://doi.org/10.1016/j.bspc.2024.106629)]

- **Deformable Medical Image Registration Based on Wavelet Transform and Linear Attention** (Biomedical Signal Processing and Control, 2024) — Fuses wavelet-based multi-scale features with linear self-attention to model large and complex deformations, mixing CNN-style local modeling with Transformer-style global interactions.  
  `Tags:` [DWT][Attention][MultiScale][Hybrid] | `Task:` Reg | `Backbone:` CNN + Linear Attention  
  [[Biomed Signal Process Control 2024](https://doi.org/10.1016/j.bspc.2024.106508)]


### 4.5 Other Backbones
- **Fast Diffeomorphic Image Registration via Fourier-Approximated Lie Algebras (FLASH)** (IJCV, 2019) — Geodesic-shooting framework where the Lie algebra of velocity fields is approximated by a low-dimensional, band-limited Fourier basis, performing most computations in frequency space.  
  `Tags:` [FFT][Diffeomorphic][ModelDriven] | `Task:` Reg | `Backbone:` Variational / LDDMM  
  [[IJCV 2019](https://link.springer.com/article/10.1007/s11263-018-1130-5)]

- **Medical Image Registration in Fractional Fourier Transform Domain** (Optik, 2013) — Fractional Fourier domain registration that improves robustness to noise and some geometric distortions compared to standard Fourier correlation.  
  `Tags:` [FrFT][Classical][Optimization] | `Task:` Reg | `Backbone:` Analytical  
  [[Optik 2013](https://doi.org/10.1016/j.ijleo.2013.01.021)]

- **Variational Image Registration by a Total Fractional-Order Regularization Approach** (Journal of Computational Physics, 2015) — Variational registration with total fractional-order regularization; fractional derivatives are implemented in the frequency domain, acting as spectral smoothness priors on the deformation field.  
  `Tags:` [FractionalOrder][SpectralReg][Variational] | `Task:` Reg | `Backbone:` PDE / Variational  
  [[J Comput Phys 2015](https://doi.org/10.1016/j.jcp.2014.10.050)]


---

## 5. Classification / Detection / Diagnosis

### 5.1 CNN-based

- **Gabor / wavelet-based deep learning for skin lesion classification** (Computers in Biology and Medicine, 2019; IET Image Processing, 2020) — Uses Gabor / discrete wavelet transforms to generate multi-scale, multi-orientation response maps as inputs to CNNs for dermoscopic skin lesion classification (melanoma vs non-melanoma).  
  `Tags:` [Gabor][DWT][Feature][MultiScale] | `Task:` Cls (Skin Lesions) | `Backbone:` CNN  
  `Refs:` [[Computers in Biology and Medicine 2019](https://pubmed.ncbi.nlm.nih.gov/31499395/)] [[IET Image Processing 2020](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2019.0553)]

- **Wavelet transform-based deep residual network + ReLU-ELM for skin lesion classification** (Expert Systems with Applications, 2023) — Applies wavelet transform to enhance lesion details, then uses a deep residual CNN as feature extractor and an extreme learning machine classifier for robust skin lesion diagnosis.  
  `Tags:` [DWT][Feature][MultiScale] | `Task:` Cls (Skin Lesions) | `Backbone:` CNN + ResNet  
  `Refs:` [[Expert Systems with Applications 2023](https://www.jcancer.org/v16p0506.htm)]

- **A Lightweight Multi-Frequency Feature Fusion Network with Efficient Attention for Breast Tumor Classification in Pathology Images** (Information, 2025) — Designs multi-frequency feature fusion blocks that aggregate low-/mid-/high-frequency components with channel–spatial attention for breast pathology image classification.  
  `Tags:` [DWT][MultiFreq][Attention] | `Task:` Cls (Breast Pathology) | `Backbone:` CNN  
  `Refs:` [[Information 2025](https://www.mdpi.com/2078-2489/16/7/579)]


### 5.2 ViT-based

- **An Enhanced Vision Transformer with Wavelet Position Embedding for Histopathological Image Classification** (Pattern Recognition, 2023) — Introduces wavelet-based position embeddings to mitigate aliasing from naive downsampling and encode multi-scale spatial–frequency cues within a ViT for histopathology WSI patches.  
  `Tags:` [DWT][PosEnc][MultiScale] | `Task:` Cls (Histopathology) | `Backbone:` ViT  
  `Refs:` [[Pattern Recognition 2023](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002327)]

- **Breast Cancer Histopathology Image Classification Using Transformer with Discrete Wavelet Transform (DWNAT-Net)** (Medical Engineering & Physics, 2025) — Combines discrete wavelet transform with a neighborhood attention Transformer; DWT provides multi-resolution frequency tokens that are fed into NAT blocks for breast cancer subtype/grade classification.  
  `Tags:` [DWT][TokenMix][MultiScale] | `Task:` Cls (Breast Histopathology) | `Backbone:` Transformer (NAT)

  `Refs:` [[Medical Engineering & Physics 2025](https://www.sciencedirect.com/science/article/pii/S1350453325000360)]

- **A Vision Transformer Network with Wavelet-Based Features for Breast Ultrasound Classification** (Image Analysis & Stereology, 2024) — Extracts wavelet-domain features from US images and feeds them to a ViT-style architecture, improving breast lesion classification under limited training data.  
  `Tags:` [DWT][Feature][MultiScale] | `Task:` Cls (Breast Ultrasound) | `Backbone:` ViT  
  `Refs:` [[Image Analysis & Stereology 2024](https://www.scilit.com/publications/cf5dec4d29f7c1a46ac0670aef59c01b)]


### 5.3 Mamba / SSM-based

- *[Placeholder]* To our knowledge, explicitly **frequency-domain–aware** Mamba/SSM architectures for medical image classification/detection have not yet appeared in major imaging venues (most frequency–Mamba work is currently on segmentation / SR).  
  `Tags:` [TODO][SSM] | `Task:` Cls/Dx | `Backbone:` Mamba/SSM  


### 5.4 Hybrid (Graph / Multimodal / Others)

- **MS-GWNN: Multi-Scale Graph Wavelet Neural Network for Breast Cancer Diagnosis** (IEEE ISBI / arXiv, 2020–2022) — Converts histopathology images into graphs and performs multi-scale graph wavelet convolutions to capture structural tissue patterns for breast cancer grading/diagnosis.  
  `Tags:` [GraphWavelet][MultiScale][Graph] | `Task:` Cls (Breast Histopathology) | `Backbone:` GCN / Graph Wavelet Network  
  `Refs:` [[arXiv 2020](https://arxiv.org/abs/2012.14619)]

- **Hybrid graph–wavelet frameworks for multi-scale breast cancer diagnosis** (follow-ups building on MS-GWNN, e.g., in MedIA survey / graph pathology works) — Extend graph wavelet ideas with relational graphs or transformers for improved WSI-level diagnosis.  
  `Tags:` [GraphWavelet][MultiScale][Fusion] | `Task:` Cls/Dx (WSI) | `Backbone:` Graph + (CNN/Transformer)  
  `Refs:` [[Medical Image Analysis 2024 survey](https://www.sciencedirect.com/science/article/pii/S1361841524001221)]


### 5.5 Other Backbones / Plug-in Frequency Modules

- **FoPro-KD: Fourier Prompted Effective Knowledge Distillation for Long-Tailed Medical Image Recognition** (IEEE TMI, 2024) — Introduces a Fourier Prompt Generator that perturbs specific frequency bands in input images to expose teacher frequency preferences and guide KD, improving long-tailed GI and skin lesion recognition.  
  `Tags:` [FFT][Prompt][KD] | `Task:` Cls (Long-Tailed) | `Backbone:` Backbone-agnostic (Teacher/Student CNN or ViT)  
  `Refs:` [[IEEE TMI 2024](https://arxiv.org/abs/2305.17421)]


- **Enhancing Breast Cancer Classification Using a Deep Sparse Wavelet Autoencoder Approach** (Scientific Reports, 2025) — Uses wavelet transforms and a sparse autoencoder to construct compact frequency-domain representations, followed by a classifier for multi-class breast cancer recognition.  
  `Tags:` [DWT][AutoEncoder][Sparse] | `Task:` Cls (Breast Cancer) | `Backbone:` Wavelet Autoencoder + Classifier  
  `Refs:` [[Scientific Reports 2025](https://www.nature.com/articles/s41598-025-11816-y)]


---

## 6. Self-supervised / Pretraining / Domain Generalization

### 6.1 Frequency-based SSL objectives

- **FreMIM: Fourier Transform Meets Masked Image Modeling for Medical Image Segmentation** (WACV, 2024) — MIM-style pretraining in the Fourier domain: masks image patches and predicts missing low-/high-frequency spectra with multi-stage supervision to learn better dense representations.  
  `Tags:` [FFT][SSL][MIM][MultiScale] | `Task:` SSL-Seg | `Backbone:` ViT/Hybrid  
  `Paper:` [[WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Wang_FreMIM_Fourier_Transform_Meets_Masked_Image_Modeling_for_Medical_Image_WACV_2024_paper.html)] [[arXiv](https://arxiv.org/abs/2304.10864)] [[Code](https://github.com/Rubics-Xuan/FreMIM)]

- **FRCNet: Frequency and Region Consistency for Semi-supervised Medical Image Segmentation** (MICCAI, 2024) — Adds frequency-domain consistency loss (aligning Fourier spectra of predictions under strong/weak augmentation) plus region-level consistency for unlabeled data.  
  `Tags:` [FFT][SSL][Consistency][Loss] | `Task:` Semi-SL Seg | `Backbone:` Plug-in (U-Net / Swin / others)  
  `Paper:` [[MICCAI 2024](https://papers.miccai.org/miccai-2024/340-Paper0245.html)]

- **AdaptFRCNet: Semi-supervised Adaptation of Pre-trained Model with Frequency and Region Consistency** (Medical Image Analysis, 2025) — Extends FRCNet to adapt pre-trained / foundation models to new domains, enforcing frequency + region consistency on unlabeled target data.  
  `Tags:` [FFT][SSL][Consistency][Adapt] | `Task:` SSL/Adapt Seg | `Backbone:` Plug-in (Pretrained)  
  `Paper:` [[MedIA 2025](https://www.sciencedirect.com/science/article/abs/pii/S1361841525001732)]

- **Semi-Supervised Medical Image Segmentation Based on Frequency Domain Aware Stable Consistency Regularization** (Journal of Imaging Informatics in Medicine, 2025) — Designs frequency-domain–aware consistency terms to stabilise pseudo-label training by aligning spectral statistics between differently augmented views.  
  `Tags:` [FFT][SSL][Consistency][Loss] | `Task:` Semi-SL Seg | `Backbone:` CNN/U-Net  
  `Paper:` [[JIIM 2025](https://link.springer.com/article/10.1007/s10278-025-01397-7)]

- **A Self-Supervised Framework for Improved Generalisability in Ultrasound B-mode Image Segmentation** (Biomedical Signal Processing and Control, 2026) — Ultrasound SSL with relation-contrastive objectives and domain-inspired pretext tasks; uses combined spatial + frequency augmentations to learn robust encoders.  
  `Tags:` [FFT][SSL][Contrastive][US] | `Task:` SSL-Seg (US) | `Backbone:` CNN/ResUNet  
  `Paper:` [[BSPC 2026](https://www.sciencedirect.com/science/article/pii/S1746809425016003)] [[arXiv](https://arxiv.org/abs/2502.02489)]


### 6.2 Frequency augmentation & invariance

- **Fourier-based Augmentation with Applications to Domain Generalization (AmpMix/FACT)** (Pattern Recognition, 2023) — Generic Fourier-based framework for DG: linearly mixes amplitude spectra (AmpMix) while keeping phase, plus multi-view consistency training; includes medical segmentation as a benchmark scenario.  
  `Tags:` [FFT][Aug][Consistency][DG] | `Task:` SSL/DG (Generic + Med) | `Backbone:` Backbone-agnostic  
  `Paper:` [[PR 2023](https://dl.acm.org/doi/10.1016/j.patcog.2023.109474)]

- **Random Amplitude Mixup (RAM) for Generalizable Medical Image Segmentation** (ECCV, 2022) — Perturbs low-frequency amplitude of source images to encourage amplitude-invariant representations and improve cross-domain segmentation.  
  `Tags:` [FFT][Aug][DG] | `Task:` DG-Seg | `Backbone:` Plug-in (U-Net / DeepLab / etc.)  
  `Paper:` [[ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810415.pdf)]

- **Source-Free Domain Adaptation for Medical Image Segmentation with Fourier Style Mining (FSM)** (Medical Image Analysis, 2022) — Decomposes amplitude spectra into “transferable” and “domain-specific” styles, then performs source-free DA using mined style banks.  
  `Tags:` [FFT][Aug][DA][SFDA] | `Task:` DA-Seg | `Backbone:` Plug-in (CNN/Transformer)  
  `Paper:` [[MedIA 2022](https://www.sciencedirect.com/science/article/pii/S1361841522001851)]

- **A Self-Supervised Framework for Improved Generalisability in Ultrasound B-mode Image Segmentation** (BSPC, 2026) — Uses a toolbox of spatial + frequency augmentations as pretext transformations (e.g., frequency filtering, phase perturbation) to learn representations invariant to scanner- and operator-induced artifacts.  
  `Tags:` [FFT][Aug][SSL][US] | `Task:` SSL-Seg | `Backbone:` CNN/ResUNet  
  `Paper:` [[BSPC 2026](https://www.sciencedirect.com/science/article/pii/S1746809425016003)] [[arXiv](https://arxiv.org/abs/2502.02489)]

- **Medical Frequency Domain Learning: Considering Inter-Class and Intra-Class Frequency for Medical Image Segmentation and Classification** (IEEE BIBM, 2021) — Introduces class-aware spectral perturbations and frequency regularizers to enforce invariance to nuisance style while keeping discriminative spectral bands.  
  `Tags:` [FFT][Aug][Loss] | `Task:` Seg + Class | `Backbone:` CNN  
  `Paper:` [[BIBM 2021](https://ieeexplore.ieee.org/document/9669443)]


### 6.3 Robustness under domain shift (DG / DA / Federated)

- **FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space** (CVPR, 2021) — Exchanges amplitude spectra across clients and performs continuous frequency-space interpolation plus episodic meta-learning to improve cross-hospital generalization.  
  `Tags:` [FFT][DG][Fed] | `Task:` FedDG-Seg | `Backbone:` CNN/U-Net  
  `Paper:` [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html)]

- **Frequency-mixed Single-Source Domain Generalization for Medical Image Segmentation (FreeSDG)** (MICCAI, 2023) — Uses frequency-mixed augmentation to diversify style from a single labeled source, with regularizers to preserve semantic structure while varying appearance.  
  `Tags:` [FFT][DG][Aug] | `Task:` SDG-Seg | `Backbone:` Plug-in (CNN/Transformer)  
  `Paper:` [[MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_13)]

- **MFNet: Meta-learning based on Frequency-Space Mix for MRI Image Segmentation** (Frontiers in Oncology, 2023) — Mixes features in frequency space within a meta-learning loop to improve cross-scanner robustness for nasopharyngeal carcinoma MRI.  
  `Tags:` [FFT][DG][Meta][Mix] | `Task:` DG-Seg (MRI) | `Backbone:` CNN/UNet  
  `Paper:` [[Front Oncol 2023](https://www.frontiersin.org/articles/10.3389/fonc.2023.1247263/full)]

- **Domain-Specific Convolution and High-Frequency Reconstruction-based Unsupervised Domain Adaptation for Medical Image Segmentation** (MICCAI, 2022) — Introduces domain-specific convolution branches and high-frequency reconstruction losses to better align structural details across domains in UDA.  
  `Tags:` [FFT][DA][Recons] | `Task:` UDA-Seg | `Backbone:` CNN/UNet  
  `Paper:` [[MICCAI 2022](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_76)]

- **MoreStyle: Relax Low-frequency Constraint of Fourier-based Image Reconstruction in Generalizable Medical Image Segmentation** (MICCAI, 2024) — Revisits Fourier-based reconstruction for SDG, relaxing “low-frequency only” constraints and adding uncertainty-aware weighting to strengthen robustness on multi-site fundus and prostate segmentation.  
  `Tags:` [FFT][DG][Aug][Recons] | `Task:` SDG-Seg | `Backbone:` Plug-in (CNN/Transformer)  
  `Paper:` [[MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/0782_paper.pdf)] [[Code](https://github.com/Maxwell-Zhao/MoreStyle)]


---

## 7. Foundation Models

### 7.1 Segmentation FM + Frequency Adapters

- **I-MedSAM: Implicit Medical Image Segmentation with Segment Anything** — ECCV 2024  
  Leverages a SAM-based encoder with a frequency adapter that aggregates high-frequency information in the spectral domain and an implicit neural representation (INR) decoder for continuous masks.  
  `Tags:` [FFT][Adapter][HighFreq][INR] | `Task:` Seg | `Backbone:` SAM + INR  
  [[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01503.pdf)] [[arXiv](https://arxiv.org/abs/2311.17081)] [[code](https://github.com/ucwxb/I-MedSAM)]

- **FreqSAM2-UNet: Adapter Fine-Tuning Frequency-Aware Network of SAM2 for Universal Medical Segmentation** — ICIC 2025 (LNCS)  
  Freezes the SAM2 Hiera encoder and adds frequency-aware adapters plus adaptive high/low-pass filters during feature fusion to preserve high-frequency boundaries.  
  `Tags:` [FFT][Adapter][Hi/LoPass] | `Task:` Seg | `Backbone:` SAM2 + U-Net  
  [[ICIC 2025](https://link.springer.com/chapter/10.1007/978-981-95-0036-9_26)]

- **UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets** — Expert Systems with Applications, 2025  
  SAM-style ultrasound foundation model trained on US-43d; combines prompt-conditioned segmentation with modality-specific features (including frequency-domain cues) and supports prompted classification.  
  `Tags:` [FFT?][Prompt][FM][Ultrasound] | `Task:` Seg / Cls | `Backbone:` SAM-style ViT  
  [[ESWA 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425038382)] [[arXiv](https://arxiv.org/abs/2411.16222)] [[code](https://github.com/CAMMA-public/UltraSam)]

- **FreqFiT: Frequency Strikes Back – Boosting Parameter-Efficient Foundation Model Adaptation for Medical Imaging** — MICCAI 2025 (Oral)  
  Inserts an FFT-based fine-tuning block between ViT layers to modulate token spectra, improving PEFT (e.g., LoRA) when adapting MedMAE / DINOv2-style vision FMs for 2D/3D medical segmentation and classification.  
  `Tags:` [FFT][PEFT][Adapter][GlobalContext] | `Task:` Seg / Cls | `Backbone:` ViT / FM  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3066_paper.pdf)] [[arXiv](https://arxiv.org/abs/2411.19297)] [[code](https://github.com/tsly123/FreqFiT_medical)]

- **VP-SAM: Video Polyp Segmentation with Fourier Spectrum-guided SAM Adapters** — ECCV 2024  
  Uses Fourier amplitude-based semantic disentanglement adapters on top of SAM for colonoscopy video segmentation; a prototype for frequency-guided prompting around a segmentation FM.  
  `Tags:` [FFT][Adapter][Video][Polyp] | `Task:` Seg | `Backbone:` SAM  
  [[ECCV 2024](https://www.ecva.net)]  <!-- official link to be filled once stable -->
  
- **RobustSAM: Degradation-Robust Segment Anything via Fourier Degradation Suppression** — CVPR 2024  
  Adds a Fourier Degradation Suppression (FDS) module to SAM to mitigate blur/noise and distribution shift, improving robustness of FM-based segmentation under degraded imaging — potentially useful for low-dose / noisy medical scans.  
  `Tags:` [FFT][Robust][Adapter] | `Task:` Generic Seg | `Backbone:` SAM  
  [[CVPR 2024](https://openaccess.thecvf.com)]  <!-- generic FM + Fourier robustness -->


### 7.2 Reconstruction / Diffusion FM + Frequency Constraints

- **High-Frequency Space Diffusion Model for Accelerated MRI** — IEEE TMI, 2024  
  Designs a diffusion process directly in high-frequency k-space, adding multi-scale HF noise during forward SDE and reconstructing via reverse SDE; effectively a frequency-structured diffusion prior for MRI.  
  `Tags:` [k-space][HF][Diffusion][Recon] | `Task:` MRI Recon | `Backbone:` Score-based Diffusion  
  [[IEEE TMI 2024](https://arxiv.org/pdf/2208.05481.pdf)] [[code](https://github.com/Aboriginer/HFS-SDE)]

- **FilterDiff: Noise-Free Frequency-Domain Diffusion Models for Accelerated MRI Reconstruction** — MICCAI 2025  
  Models the diffusion process as a learned frequency-domain filtering operation (instead of Gaussian noise injection), aligning the forward / reverse process with MRI acquisition and enforcing spectral fidelity.  
  `Tags:` [FFT][Filter][Diffusion][k-space] | `Task:` MRI Recon | `Backbone:` Frequency-domain Diffusion  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/1256_paper.pdf)]

- **Fourier Diffusion Models: A Method to Control MTF and NPS in Medical Imaging** — 2025  
  Parameterizes diffusion models in the Fourier domain to explicitly control modulation transfer function (MTF) and noise power spectrum (NPS), enabling reconstruction FMs whose spatial resolution / noise characteristics are tunable by design.  
  `Tags:` [Fourier][MTF][NPS][Theory] | `Task:` Recon / Sim | `Backbone:` Diffusion (Fourier-param.)  
  [[Journal Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12619680/)]

- **Zero-shot Medical Image Translation via Frequency-Guided Diffusion Models (FGDM)** — IEEE TMI, 2023  
  Introduces frequency-domain filters as guidance signals in the diffusion process for cross-modality CT/CBCT/MR translation under zero-shot setting, preserving anatomical structures via spectral constraints.  
  `Tags:` [FFT][Filter][Translation][ZeroShot] | `Task:` Cross-modality Translation / Recon | `Backbone:` Diffusion  
  [[IEEE TMI 2023](https://arxiv.org/abs/2304.02742)] [[code](https://github.com/Kent0n-Li/FGDM)]

- **Prior Frequency Guided Diffusion Model for Limited Angle (LA)-CBCT Reconstruction (PFGDM)** — Physics in Medicine & Biology, 2024 (to appear)  
  Conditions a diffusion prior on high-frequency information from prior CT scans; uses decaying HF conditioning to reconstruct LA-CBCT while preserving fine anatomical details.  
  `Tags:` [FFT][Prior][k-space][Diffusion] | `Task:` CBCT Recon | `Backbone:` Conditional Diffusion  
  [[arXiv 2024](https://arxiv.org/abs/2404.01448)]

> You can treat this subsection as “frequency-aware diffusion / generative FMs for reconstruction / translation” and cross-reference Section 2 for more task-specific models.


### 7.3 VLM / Prompting + Frequency Priors

- **FreqFiT (revisited)** — MICCAI 2025  
  Frequency-based fine-tuning module for ViT FMs, compatible with visual prompt tuning / LoRA; naturally pairs with language-conditioned or report-conditioned medical FMs where visual backbone is ViT.  
  `Tags:` [FFT][PEFT][PromptReady] | `Task:` Seg / Cls | `Backbone:` ViT FM  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3066_paper.pdf)] [[code](https://github.com/tsly123/FreqFiT_medical)]

- **FMISeg: Frequency-domain Multi-modal Fusion for Language-guided Medical Image Segmentation** — MICCAI 2025  
  Aligns vision and text features in the frequency domain for language-guided segmentation, providing a prototype of “frequency-aware vision–language FM” for medical imaging.  
  `Tags:` [FFT][Fusion][VL][GuidedSeg] | `Task:` Lang-guided Seg | `Backbone:` Vision–Language Hybrid  
  [[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/3678_paper.pdf)]

- **FVP: Fourier Visual Prompting for Source-Free UDA of Medical Image Segmentation** — IEEE TMI, 2023  
  Treats low-frequency Fourier perturbations as visual prompts injected into the input space of a frozen segmenter for source-free domain adaptation; a bridge between prompting and frequency priors.  
  `Tags:` [FFT][Prompt][UDA] | `Task:` Seg (SFUDA) | `Backbone:` Frozen Seg + Freq Prompt  
  [[IEEE TMI 2023](https://arxiv.org/abs/2304.13672)]

- **UltraSam (prompted classification)** — ESWA, 2025  
  Extends SAM-style segmentation FM to “prompted classification” by jointly decoding prompts and image features; while not explicitly frequency-prompted, it is a natural host for future frequency-aware prompts (e.g., combining with FreqFiT / FVP-like modules).  
  `Tags:` [Prompt][FM][Ultrasound][Future-Freq] | `Task:` Prompted Cls / Seg | `Backbone:` SAM-style ViT  
  [[ESWA 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425038382)] [[arXiv](https://arxiv.org/abs/2411.16222)]


---
## 8. Datasets & Benchmarks

### Segmentation

- **BraTS (Brain Tumor Segmentation)** — Multi-institutional, multi-modal brain MRI (T1, T1c, T2, FLAIR).  
  `Target:` Glioma subregions (ET/TC/WT)  
  `Metrics:` Dice, HD95, sensitivity, specificity  
  `Notes:` Long-running MICCAI challenge; standard 3D MRI benchmark for tumor segmentation, domain generalization, and frequency-based methods.  
  [[BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/)]

- **Medical Segmentation Decathlon (MSD)** — 10 heterogeneous 3D CT/MR tasks (brain tumours, heart, liver, lung, pancreas, prostate, colon, hepatic vessels, spleen, hippocampus).  
  `Target:` Organ / lesion segmentation across tasks  
  `Metrics:` Dice, NSD / HD95  
  `Notes:` Large-scale multi-organ, multi-modality benchmark; widely used to test cross-task generalization, domain shift, and frequency-aware modules.  
  [[MSD](http://medicaldecathlon.com/)]

- **ACDC (Automated Cardiac Diagnosis Challenge)** — Cine cardiac MRI from multiple patients.  
  `Target:` LV / RV / myocardium segmentation, plus diagnosis labels  
  `Metrics:` Dice, Hausdorff distance, EF / volume error  
  `Notes:` Classic cardiac MRI segmentation dataset, often used for semi-/self-supervised and domain-generalization studies.

- **KiTS19 / KiTS21 (Kidney Tumor Segmentation)** — Abdominal CT volumes from multiple centers.  
  `Target:` Kidney, tumor (and cysts/other structures in KiTS21)  
  `Metrics:` Dice, HD95  
  `Notes:` Standard benchmark for renal tumour segmentation; challenging shapes and class imbalance.

- **ISIC 2018–2020 (Skin Lesion Analysis)** — Dermoscopic RGB images.  
  `Target:` Lesion segmentation, dermoscopic attributes, lesion classification  
  `Metrics:` Jaccard / Dice, pixel-level accuracy, AUC  
  `Notes:` Widely used for evaluating boundary-sensitive models, spectral/texture-aware networks, and joint seg+cls setups.

- **Multi-site Prostate MRI (e.g., PROMISE12, MSD Task05, and extended cohorts)** — Prostate T2 / mp-MRI from different vendors and protocols.  
  `Target:` Prostate gland, peripheral / transition zones, lesions  
  `Metrics:` Dice, HD95, surface distance  
  `Notes:` Strong scanner/protocol variability; common testbed for domain generalization and Fourier-style style-transfer / augmentation.

- **Retinal Fundus Packs (DRISHTI-GS, RIM-ONE r3, REFUGE, DRIVE, STARE, CHASE_DB1, HRF, PAPILA, etc.)** — 2D color fundus photographs.  
  `Target:` Optic disc/cup, vessels, macula, lesions, glaucoma labels  
  `Metrics:` Dice / Jaccard (OD/OC, vessels), AUC (glaucoma), sensitivity / specificity  
  `Notes:` Frequently used in frequency-based DG / DA works (e.g., Fourier mixing, amplitude-style transfer) due to strong inter-dataset appearance differences.


### Reconstruction / Super-Resolution

- **fastMRI (Knee / Brain)** — Raw multi-coil k-space and reconstructed images for knee and brain MRI at various acceleration factors.  
  `Modality:` MRI  
  `Setting:` Accelerated reconstruction, under-sampled k-space  
  `Metrics:` NMSE, PSNR, SSIM, perceptual metrics  
  `Notes:` Main public benchmark for deep MRI reconstruction; many k-space and frequency-aware recon models are evaluated here.  
  [[fastMRI](https://fastmri.med.nyu.edu/)]

- **Calgary–Campinas (CC-359 and related challenges)** — 3D T1-weighted brain MRI (multiple vendors and field strengths).  
  `Modality:` MRI  
  `Setting:` Accelerated recon, resolution enhancement, cross-scanner robustness  
  `Metrics:` PSNR, SSIM, NMSE, brain-structure similarity measures  
  `Notes:` Common benchmark for 3D MRI SR/reconstruction and frequency-domain priors.  
  [[Calgary–Campinas](https://sites.google.com/view/calgary-campinas-dataset)]

- **LoDoPaB-CT (Low-Dose Parallel Beam CT)** — Simulated low-dose chest CT from LIDC-IDRI.  
  `Modality:` CT  
  `Setting:` Low-dose reconstruction with sparse/noisy projections  
  `Metrics:` PSNR, SSIM, NMSE  
  `Notes:` Standard low-dose CT reconstruction benchmark; frequently used with iterative and frequency-based reconstruction methods.  
  [[LoDoPaB-CT](https://zenodo.org/records/3384092)]

- **AAPM 2016 Low Dose CT Grand Challenge (LDCT-and-Projection-Data)** — Abdominal contrast-enhanced CT with full-dose vs quarter-dose data.  
  `Modality:` CT  
  `Setting:` Low-dose (quarter-dose) reconstruction and denoising  
  `Metrics:` PSNR, SSIM, NMSE, plus detection/reading performance in some studies  
  `Notes:` Widely adopted for low-dose CT reconstruction and deep denoising.

- **Paired 3T–7T Hippocampal Subfield Dataset** — Paired 3T/7T multimodal brain MRI with manual hippocampal subfield labels on 7T.  
  `Modality:` MRI  
  `Setting:` 3T→7T synthesis / SR, high-field reconstruction  
  `Metrics:` PSNR, SSIM (image similarity), Dice (subfield segmentation)  
  `Notes:` Good testbed for super-resolution and high-frequency detail recovery with spectral priors.


### Denoising / Enhancement / Artifact Reduction

- **LoDoPaB-CT & AAPM LDCT** — Often reused specifically for image-space denoising and enhancement (in addition to full reconstruction).  
  `Noise / Artifact:` Quantum noise, low-dose artifacts  
  `Metrics:` PSNR, SSIM, NMSE, reader study scores  
  `Notes:` Baseline benchmarks for frequency-aware denoising (wavelet / Fourier losses, NPS-based losses, etc.).

- **Ultrasound Nerve Segmentation (UNS)** — Brachial plexus ultrasound (Kaggle).  
  `Noise / Artifact:` Speckle noise, shadowing  
  `Metrics:` PSNR, SSIM (denoising), Dice (nerve segmentation)  
  `Notes:` Frequently used in works that jointly evaluate despeckling and downstream segmentation.

- **PICMUS / CUBDL Ultrasound Beamforming Data** — Plane-wave RF data and B-mode reference images.  
  `Noise / Artifact:` Speckle, side lobes, clutter; beamforming artifacts  
  `Metrics:` CNR, spatial resolution, geometric distortion, speckle statistics  
  `Notes:` Classic benchmarks for ultrasound beamforming and denoising, especially relevant for k-space / frequency-domain methods.

- **HC18 / BUSI / Fetal-head & Breast Ultrasound datasets** — Public ultrasound datasets for fetal head circumference, breast lesion analysis, etc.  
  `Noise / Artifact:` Medical US speckle, low contrast, acquisition artifacts  
  `Metrics:` PSNR, SSIM (denoising), Dice / IoU (segmentation), AUC (diagnosis)  
  `Notes:` Commonly used to test speckle suppression with preservation of diagnostically relevant edges and textures.


### Registration / Motion / Deformation

- **Learn2Reg (L2R) Suite** — Collection of registration tasks: OASIS/IXI brain MRI, abdominal CT, lung CT, etc.  
  `Setting:` Inter-subject / intra-subject registration across multiple anatomies  
  `Metrics:` Target Registration Error (TRE), Dice (warped labels), surface distance, Jacobian statistics  
  `Notes:` Main deep-learning registration benchmark; spectrum-regularized and band-limited deformation models are often compared here.  
  [[Learn2Reg datasets](https://learn2reg.grand-challenge.org/Datasets/)]

- **COPDGene / DIR-Lab 4DCT lung registration** — Thoracic 4DCT and paired inhale/exhale CT scans with landmark annotations.  
  `Setting:` Large-deformation lung motion (respiratory), 4D registration  
  `Metrics:` TRE on landmarks, Dice for lung/lobe masks, Jacobian regularity  
  `Notes:` Standard benchmark for motion-compensated CT and 4D registration, used by many classical and deep approaches.

- **AbdomenCT (L2R Abdomen)** — 3D abdominal CT with multi-organ annotations (liver, spleen, kidneys, pancreas, vessels, etc.).  
  `Setting:` Inter-subject abdominal registration  
  `Metrics:` Dice, surface distance, TRE  
  `Notes:` Tests registration quality under large inter-subject anatomical variability and strong intensity inhomogeneity.

- **IXI Brain MRI** — Multi-center structural brain MRI (T1/T2/PD/MRA/DTI) of healthy subjects.  
  `Setting:` Inter-subject registration, cross-site generalization  
  `Metrics:` Dice for propagated anatomical labels, TRE  
  `Notes:` Generic brain MRI dataset frequently used for registration, domain shift, and federated learning studies.  
  [[IXI dataset](http://brain-development.org/ixi-dataset/)]


### Common Metrics

- **Segmentation:** Dice coefficient, IoU / Jaccard index, 95th percentile Hausdorff distance (HD95), Average Symmetric Surface Distance (ASSD)  
- **Reconstruction / SR / Denoising:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Normalized Mean Squared Error (NMSE), sometimes perceptual metrics (e.g., LPIPS)  
- **Classification / Detection / Diagnosis:** Area Under the ROC Curve (AUC), Accuracy, F1-score, Sensitivity, Specificity, calibration metrics (e.g., Expected Calibration Error, ECE)  
- **Registration:** Target Registration Error (TRE, landmark distance), Dice of propagated labels, surface distance / ASSD, Jacobian determinant statistics (folding ratio, volume preservation)


---

## 9. Implementation Notes
- Complex FFT handling: magnitude/phase vs real/imag, normalization, `fftshift`
- Padding & boundary effects: ringing, aliasing, windowing
- Wavelet details: level choice, mother wavelet, boundary mode
- Where frequency helps: texture/speckle, artifacts, low-dose noise, protocol shift
- Suggested ablations:
  - remove frequency branch
  - replace FFT with DCT/DWT
  - injection point ablation (Input vs Feature vs Loss)
  - fusion strategy ablation (sum/concat/attention/gating)
  - selection routing ablation (hard vs soft vs gumbel)

---

## 10. Index

### 10.1 Index by Frequency Transform
- `[FFT]`: <!-- list paper links here -->
- `[DWT]`:
- `[DCT]`:
- `[STFT]`:
- `[Shearlet]`:
- `[Curvelet]`:
- `[LearnableFreq]`:
- `[HybridFreq]`:

### 10.2 Index by Backbone
- CNN:
- ViT:
- Mamba/SSM:
- Hybrid:
- Other:

### 10.3 Index by Injection / Usage
- `[Input]`:
- `[Feature]`:
- `[Attention]`:
- `[TokenMix]`:
- `[Loss]`:
- `[Aug]`:
- `[Selection]`:
- `[Fusion]`:
- `[Explain]`:

---
## 11. Contributing

Entry requirements

- Title + venue/year
- `Tags:` at least **1 frequency transform tag** + **1 injection/usage tag**
- `Task:` and `Backbone:` are required
- Code / data links are optional but highly recommended

Dedup rule

- For the same work, prefer the official **conference/journal** version;
  arXiv can be added as an extra link in the same entry.

---

## 12. Citation

```bibtex
@misc{awesome-frequency-domain-medical-imaging,
  title  = {Awesome Frequency-Domain Methods for Medical Imaging},
  author = {Ze Rong},
  year   = {2025},
  url    = {https://github.com/zerong7777-boop/awesome-frequency-domain-medical-imaging}
}
