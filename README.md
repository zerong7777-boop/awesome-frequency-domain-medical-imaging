# Awesome Frequency-Domain Methods for Medical Imaging
> 频域变换（FFT / DWT / DCT / …）在医学影像不同任务中的方法、模块与实践总结（单文件仓库）。

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
- <!-- Example -->
- **[TITLE]** (VENUE, YEAR) — takeaway.  
  `Tags:` [DWT][Feature][MultiScale][CT][2D] | `Task:` Seg | `Backbone:` CNN  
  `Code:`  | `Data:`  | `Notes:`  

### 1.2 ViT-based
- **[TITLE]** (VENUE, YEAR) — takeaway.  
  `Tags:` [FFT][Attention][Fusion][MRI][3D] | `Task:` Seg | `Backbone:` ViT  
  `Code:`  | `Data:`  | `Notes:`  

### 1.3 Mamba / SSM-based
- **[TITLE]** (VENUE, YEAR) — takeaway.  
  `Tags:` [DCT][TokenMix][Selection][US][2D] | `Task:` Seg | `Backbone:` Mamba/SSM  
  `Code:`  | `Data:`  | `Notes:`  

### 1.4 Hybrid
- **[TITLE]** (VENUE, YEAR) — takeaway.  
  `Tags:` [HybridFreq][Fusion][Feature][CT][3D] | `Task:` Seg | `Backbone:` Hybrid  
  `Code:`  | `Data:`  | `Notes:`  

### 1.5 Other Backbones
- <!-- e.g., UNet variants with special mixers, graph nets, implicit nets -->

### 1.6 Backbone-agnostic / Plug-in Modules
- <!-- Frequency plug-ins reusable across backbones (adapters, spectral gates, routing, losses) -->

---

## 2. Reconstruction & Super-Resolution

### 2.1 CNN-based
- **[TITLE]** (VENUE, YEAR) — takeaway.  
  `Tags:` [FFT][Loss][K-space][MRI][2D] | `Task:` Recon/SR | `Backbone:` CNN  
  `Code:`  | `Data:`  | `Notes:`  

### 2.2 ViT-based
- ...

### 2.3 Mamba / SSM-based
- ...

### 2.4 Hybrid
- ...

### 2.5 Other Backbones
- ...

### 2.6 Backbone-agnostic / Plug-in Modules
- ...

### 2.7 K-space / Complex-specific
- <!-- MRI k-space / complex-valued networks / data consistency / spectral priors -->

---

## 3. Denoising / Enhancement / Artifact Reduction

### 3.1 CNN-based
- ...

### 3.2 ViT-based
- ...

### 3.3 Mamba / SSM-based
- ...

### 3.4 Hybrid
- ...

### 3.5 Other Backbones
- ...

### 3.6 Backbone-agnostic / Plug-in Modules
- ...

---

## 4. Registration / Motion / Deformation

### 4.1 CNN-based
- ...

### 4.2 ViT-based
- ...

### 4.3 Mamba / SSM-based
- ...

### 4.4 Hybrid
- ...

### 4.5 Other Backbones
- ...

---

## 5. Classification / Detection / Diagnosis

### 5.1 CNN-based
- ...

### 5.2 ViT-based
- ...

### 5.3 Mamba / SSM-based
- ...

### 5.4 Hybrid
- ...

### 5.5 Other Backbones
- ...

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
> 按任务列常用数据集即可（不求全，但要覆盖主流）。

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
如果该列表对你有帮助，欢迎引用：

```bibtex
@misc{awesome_freq_medimg,
  title  = {Awesome Frequency-Domain Methods for Medical Imaging},
  author = {YOUR_NAME},
  year   = {2025},
  url    = {https://github.com/YOUR_GITHUB/awesome-freq-medimg}
}
