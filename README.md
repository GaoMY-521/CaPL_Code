# CaPL
Causality-guided Prompt Learning for Vision-language Models via Visual Granulation

Paper Link: [arXiv](http://arxiv.org/abs/2509.03803)

The prompt learner is designed and initialized following [SHIP](https://arxiv.org/pdf/2307.07397) (Wang et al., ICCV 2023). Please refer to [initialization](https://github.com/mrflogs/SHIP).

The attribute disentanglement module is designed following [BBDM](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf) (Li et al., CVPR 2023). Please refer to [BBDM-based network](https://github.com/xuekt98/BBDM).

The codes for constructing factual granules and counterfactual granules are in ```granule.py```.

```test.py``` is used for inference, and the learned textual features are in  ```weights``` (which are used solely for inference without additional modules). 
