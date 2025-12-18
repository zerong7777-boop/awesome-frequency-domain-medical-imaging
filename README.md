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
  - [6.3 Robustness under domain shift](#63-robustness-under-domain-shift)
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

## 0. Scope & How to Use
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
- <!-- spectral contrastive, spectral consistency, cross-view spectral agreement -->

### 6.2 Frequency augmentation & invariance
- <!-- frequency masking, band-stop, spectrum mixup, wavelet dropout -->

### 6.3 Robustness under domain shift
- <!-- scanner/protocol shift, style-frequency disentanglement, cross-site generalization -->

---

## 7. Foundation Models

### 7.1 Segmentation FM + Frequency Adapters
- <!-- FM + spectral adapters / LoRA-style freq adapters / plug-in gating -->

### 7.2 Reconstruction / Diffusion FM + Frequency Constraints
- <!-- diffusion priors with spectral constraints / frequency-guided sampling -->

### 7.3 VLM / Prompting + Frequency Priors
- <!-- promptable medical models w/ frequency priors (if applicable) -->

---

## 8. Datasets & Benchmarks

### Segmentation
- Dataset — modality, target, metrics, notes

### Reconstruction / SR
- Dataset — MRI/CT, acceleration/low-dose setting, metrics

### Denoising / Enhancement
- Dataset — noise/artifact type, metrics

### Registration
- Dataset — motion/deformation setting, metrics

### Common Metrics
- Seg: Dice, IoU, HD95, ASSD
- Recon/SR: PSNR, SSIM, NMSE
- Classif: AUC, ACC, F1

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
**Entry requirements**
- Title + Venue/Year
- `Tags:` 至少包含 1 个频域变换标签 + 1 个注入方式标签
- `Task:` 与 `Backbone:` 必填
- Code / Data 若未知可留空，但建议补全

**Dedup rule**
- 同一工作：优先收录正式版本（conference/journal），arXiv 作为补充链接放同条目内。

---

## 12. Citation

```bibtex
@misc{awesome-frequency-domain-medical-imaging,
  title  = {Awesome Frequency-Domain Methods for Medical Imaging},
  author = {Ze Rong},
  year   = {2025},
  url    = {https://github.com/zerong7777-boop/awesome-frequency-domain-medical-imaging}
}
