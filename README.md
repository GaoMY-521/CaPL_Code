# CaPL
Causality-guided Prompt Learning for Vision-language Models via Visual Granulation

## Abstract
Prompt learning has recently attracted much attention for adapting pre-trained vision-language models (e.g., CLIP) to downstream recognition tasks. However, most of the existing CLIP-based prompt learning methods only show a limited ability for handling fine-grained datasets. To address this issue, we propose a causality-guided text prompt learning method via visual granulation for CLIP, called CaPL, where the explored visual granulation technique could construct sets of visual granules for the text prompt to capture subtle discrepancies among different fine-grained classes through casual inference. The CaPL method contains the following two modules: (1) An attribute disentanglement module is proposed to decompose visual features into non-individualized attributes (shared by some classes) and individualized attributes (specific to single classes) using a Brownian Bridge Diffusion Model; (2) A granule learning module is proposed to construct visual granules by integrating the aforementioned attributes for recognition under two causal inference strategies. Thanks to the learned visual granules, more discriminative text prompt is expected to be learned. Extensive experimental results on 15 datasets demonstrate that our CaPL method significantly outperforms the state-of-the-art prompt learning methods, especially on fine-grained datasets.

Paper Link: [arXiv](http://arxiv.org/abs/2509.03803)

## Methodology
We construct factual and counterfactual granules based on the disentangled non-individualized and individualized attributes for text prompt learning. 

### Prompt Learner
The prompt learner is designed following [CoOp](https://arxiv.org/pdf/2109.01134) (Zhou et al., IJCV 2022) and [SHIP](https://arxiv.org/pdf/2307.07397) (Wang et al., ICCV 2023). 

### Feature Disentanglement
Two encoders are involed to extract non-individualized and individualized attributes from visual features, and a BBDM-based network is used to optimize the disentangled attributes. The BBDM-based network is designed following [BBDM](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf) (Li et al., CVPR 2023), with an additional conditional input incorporated into the original design.

### Factual Granule
For each visual feature, we first obtain K individualized visual representations from its individualized attributes, as well as K individualized textual representations from the corresponding prompted textual feature. Each individualized visual representation is then combined with the non-individualized attribute to form a factual granule, resulting in K factual granules in total. Finally, these factual granules are recognized by calculating cosine similarity with the K individualized textual representations in a K-class classification task.

### Counterfactual Granule
For a batch of visual features, we swap their non-individualized attributes and individualzied attributes to construct counterfactual granules, resulting in ![formula](https://latex.codecogs.com/svg.latex?N^2) counterfactual granules in total. These counterfactual granules are recognized by calculating cosine similarity with prompted textual features in a ![formula](https://latex.codecogs.com/svg.latex?C_b)-class classification task (where ![formula](https://latex.codecogs.com/svg.latex?C_b) is the number of base classes). Each counterfactual granule is assigned the label corresponding to its individualized attribute.

### Loss Function
Cross entropy loss for factual granules, cross entropy loss for counterfactual granules, cross entropy loss for original visual features, reconstruction loss for counterfactual granules, contrastive loss for counterfactual granules.

## Code
The codes for constructing factual granules and counterfactual granules are in ```granule.py```.
The codes for BBDM-based network are in ```bbdm_clip.py``` and ```bbdm_model.py```.

Note: The complete code will be uploaded later since some related parts of this work is still under review. Users could refer to the mentioned papers and code files above.


