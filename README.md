# Digepath
### Subspecialty-Specific Foundation Model for Intelligent Gastrointestinal Pathology
[Lianghui Zhu](https://github.com/Dai-Wenxun)<sup>†</sup><br>
[Xitong Ling](https://lhchen.top)<sup>†</sup><br>
[Minxi Ouyang](https://wangjingbo1219.github.io)<sup>†</sup><br>
[Xiaoping Liu](https://moonsliu.github.io/)<sup>†</sup><br>
[Mingxi Fu](https://daibo.info/)<br>
[Tian Guan](https://andytang15.github.io)<br>
[Fanglei Fu](https://andytang15.github.io)<br>
[Xuanyu Wang](https://andytang15.github.io)<br>
[Maomao Zeng](https://andytang15.github.io)<br>
[Mingxi Zhu](https://andytang15.github.io)<br>
[Yibo Jin](https://andytang15.github.io)<br>
[Liming Liu](https://andytang15.github.io)<br>
[Song Duan](https://andytang15.github.io)<br>
[Qiming He](https://andytang15.github.io)<br>
[Yizhi Wang](https://andytang15.github.io)<br>
[Luxi Xie](https://andytang15.github.io)<sup>*</sup><br>
[Houqiang Li](https://andytang15.github.io)<sup>*</sup><br>
[Yonghong He](https://andytang15.github.io)<sup>*</sup><br>
[Sufang Tian](https://andytang15.github.io)<sup>*</sup><br>

<sup>†</sup> These authors contributed equally to this work.  
<sup>*</sup> Corresponding authors.


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
<img src="https://github.com/lingxitong/Digepath/blob/main/digelogo.png"  width="290px" align="right" />
Gastrointestinal (GI) diseases represent a clinically significant burden, necessitating precise diagnostic approaches to optimize patient outcomes. Conventional histopathological diagnosis, heavily reliant on pathologists’ subjective interpretation, suffers from limited reproducibility and diagnostic variability. To overcome these limitations and address the lack of pathology-specific foundation models for GI diseases, we develop Digepath, a specialized foundation model for GI pathology. Our framework introduces a dual-phase iterative optimization strategy combining pretraining with fine-screening, specifically designed to address the detection of sparsely distributed lesion areas in whole-slide images (WSIs). Digepath is pretrained on more than 353 million images from over 200,000 H&E-stained slides of GI disease. It attains state-of-the-art performance on 33 out of 34 tasks related to GI pathology, including pathological diagnosis, molecular prediction, gene mutation prediction, and prognosis evaluation, particularly in diagnostically ambiguous cases and resolution-agnostic tissue classification. We further translate the intelligent screening module for early GI cancer and achieve near-perfect (99.6%) sensitivity across 9 independent medical institutions nationwide. Digepath’s outstanding performance highlights its potential to bridge critical gaps in histopathological practice. This work not only advances AI-driven precision pathology for GI diseases but also establishes a transferable paradigm for other pathology subspecialties. 

