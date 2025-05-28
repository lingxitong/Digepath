# Digepath
### Subspecialty-Specific Foundation Model for Intelligent Gastrointestinal Pathology
<p align="center">
  <a href='https://scholar.google.com/citations?user=nDJI-9oAAAAJ&hl=en'>
  <img src='https://img.shields.io/badge/Arxiv-2404.19759-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://scholar.google.com/citations?user=nDJI-9oAAAAJ&hl=en'>
  <img src='https://img.shields.io/badge/Paper-PDF-purple?style=flat&logo=arXiv&logoColor=yellow'></a> 
  <a href='https://scholar.google.com/citations?user=nDJI-9oAAAAJ&hl=en'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
  <a href='https://scholar.google.com/citations?user=nDJI-9oAAAAJ&hl=en'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
  <a href='[[https://github.com/Dai-Wenxun/MotionLCM](https://scholar.google.com/citations?user=nDJI-9oAAAAJ&hl=en)](https://github.com/lingxitong/Digepath)'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
</p>

[Lianghui Zhu](https://github.com/lingxitong/Digepath)<sup>†</sup>, [Xitong Ling](https://github.com/lingxitong/Digepath)<sup>†</sup>, [Minxi Ouyang](https://github.com/lingxitong/Digepath)<sup>†</sup>, [Xiaoping Liu](https://github.com/lingxitong/Digepath)<sup>†</sup>, [Mingxi Fu](https://github.com/lingxitong/Digepath), [Tian Guan](https://github.com/lingxitong/Digepath), [Fanglei Fu](https://github.com/lingxitong/Digepath), [Xuanyu Wang](https://github.com/lingxitong/Digepath), [Maomao Zeng](https://github.com/lingxitong/Digepath), [Mingxi Zhu](https://github.com/lingxitong/Digepath), [Yibo Jin](https://github.com/lingxitong/Digepath), [Liming Liu](https://github.com/lingxitong/Digepath)<sup>‡</sup>, [Song Duan](https://github.com/lingxitong/Digepath)<sup>‡</sup>, [Yonghong He](https://github.com/lingxitong/Digepath)<sup>‡</sup>, [Sufang Tian](https://github.com/lingxitong/Digepath)<sup>‡</sup>

<sup>†</sup> These authors contributed equally to this work.  
<sup>‡</sup> Corresponding authors.




<img src="https://github.com/lingxitong/Digepath/blob/main/digelogo.png"  width="290px" align="right" />
Gastrointestinal (GI) diseases represent a clinically significant burden, necessitating precise diagnostic approaches to optimize patient outcomes. Conventional histopathological diagnosis, heavily reliant on pathologists’ subjective interpretation, suffers from limited reproducibility and diagnostic variability. To overcome these limitations and address the lack of pathology-specific foundation models for GI diseases, we develop Digepath, a specialized foundation model for GI pathology. Our framework introduces a dual-phase iterative optimization strategy combining pretraining with fine-screening, specifically designed to address the detection of sparsely distributed lesion areas in whole-slide images (WSIs). Digepath is pretrained on more than 353 million images from over 200,000 H&E-stained slides of GI disease. It attains state-of-the-art performance on 33 out of 34 tasks related to GI pathology, including pathological diagnosis, molecular prediction, gene mutation prediction, and prognosis evaluation, particularly in diagnostically ambiguous cases and resolution-agnostic tissue classification. We further translate the intelligent screening module for early GI cancer and achieve near-perfect (99.6%) sensitivity across 9 independent medical institutions nationwide. Digepath’s outstanding performance highlights its potential to bridge critical gaps in histopathological practice. This work not only advances AI-driven precision pathology for GI diseases but also establishes a transferable paradigm for other pathology subspecialties. 

