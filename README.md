# Graph Data Augmentation (GraphDA) for Deep Graph Learning
The repository contains links primarily to conference and journal publications about **graph data augmentation for deep graph learning**. If you find this repository useful, please kindly cite the following paper:

```
@article{ding2022data,
  title={Data Augmentation for Deep Graph Learning: A Survey},
  author={Ding, Kaize and Xu, Zhe and Tong, Hanghang and Liu, Huan},
  journal={arXiv preprint arXiv:2202.08235},
  year={2022}
}
```

## Roadmaps
- [GraphDA for Optimal Graph Learning](#GraphDA-for-Optimal-Graph-Learning)
  - [Optimal Structure Learning](#Optimal-Structure-Learning)
  - [Optimal Feature Learning](#Optimal-Feature-Learning)
- [GraphDA for Low-Resource Graph Learning](#GraphDA-for-Low-resource-Graph-Learning)
  - [Graph Self-Supervised Learning](#Graph-Self-Supervised-Learning)
  - [Graph Self/Co-Training](#Graph-SelfCo-Training)
  - [Graph Interpolation](#Graph-Interpolation)
  - [Graph Consistency Training](#Graph-Consistency-Training)
- [Adversarial Training on Graphs](#Adversarial-Training-on-Graphs)
- [Other Directions](#Other-Directions)
  - [GraphDA for Graph Imbalanced Learning](#GraphDA-for-Graph-Imbalanced-Learning)
  - [GraphDA for Learning on Heterophilic Graphs](#GraphDA-for-Learning-on-Heterophilic-Graphs)

## GraphDA for Optimal Graph Learning

### Optimal Structure Learning
|Name|Paper
|---|---|
|Gasoline|[[TheWebConf 2022] Graph Sanitation with Application to Node Classification](https://arxiv.org/pdf/2105.09384.pdf?ref=https://githubhelp.com) [[Code]](https://github.com/pricexu/GASOLINE)|
|GEN|[[TheWebConf 2021] Graph Structure Estimation Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3442381.3449952?casa_token=4b0qCTk088EAAAAA:sgU4Z5pS1W0dJ42SUo063Psw6XrNDTOor7EnX98HZwGeyj9_ZW_znUo-pb1S6I82_28EqBMd0wQo9g) [[Code]](https://github.com/BUPT-GAMMA/Graph-Structure-Estimation-Neural-Networks)|
|PTDNet|[[WSDM 2021] Learning to drop: Robust graph neural network via topological denoising](https://dl.acm.org/doi/pdf/10.1145/3437963.3441734?casa_token=p-KUZTPYFS0AAAAA:mYTX35RRO_p_EnC_ohXaz0dhk2uTs-cxVDqPD75GXuNvq_W-Uv2IA5yM3PTxp9JLAHiX4ycVaql2QQ) [[Code]](https://github.com/flyingdoog/PTDNet)|
|GAUG|[[AAAI 2021] Data Augmentation for Graph Neural Networks](https://www.aaai.org/AAAI21Papers/AAAI-10012.ZhaoT.pdf) [[Code]](https://github.com/zhao-tong/GAug)|
|HGSL|[[AAAI 2021] Heterogeneous Graph Structure Learning for Graph Neural Networks](https://www.aaai.org/AAAI21Papers/AAAI-3976.ZhaoJ.pdf) [[Code]](https://github.com/Andy-Border/HGSL)|
|ESML|[[ICDE 2021] Edge Sparsification for Graphs via Meta-Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9458885&casa_token=TBXQtrIXATsAAAAA:9XQxBEn9UKRP-iP_-SzY5OkDGJvOUs4WtYIn6MwznDV7LI5jIAgC3qzRWK64OwlkOBKuAt3YHHM)|
|GIB-N|[[NeurIPS 2020] Graph Information Bottleneck](https://proceedings.neurips.cc/paper/2020/file/ebc2aa04e75e3caabda543a1317160c0-Paper.pdf) [[Code]](https://github.com/snap-stanford/GIB)|
|IDGL|[[NeurIPS 2020] Iterative deep graph learning for graph neural networks: Better and robust node embeddings](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) [[Code]](https://github.com/hugochan/IDGL)|
|NeuralSparse|[[ICML 2020] Robust graph representation learning via neural sparsification](http://proceedings.mlr.press/v119/zheng20d/zheng20d.pdf) |
|Pro-GNN|[[KDD 2020] Graph structure learning for robust graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Code]](https://github.com/ChandlerBang/Pro-GNN)|
|GIB-G|[[ICLR 2020] Graph Information Bottleneck for Subgraph Recognition](https://openreview.net/forum?id=bM4Iqfg8M2k)|
|AdaEdge|[[AAAI 2020] Measuring and relieving the over-smoothing problem for graph neural networks from the topological view](https://ojs.aaai.org/index.php/AAAI/article/download/5747/5603)|
|DenSE|[[ICDM 2020] Learning Node Representations from Noisy Graph Structures](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9338405&casa_token=iSZkJT-8nZ4AAAAA:7IJV0NIJzER-E1EXVKc-AOYkrju7b1JL0yMbgVAfflT5nIKKMaTfo9oPS2VvICJEUE63Y1HPmgE&tag=1)|
|VGCN|[[NeurIPS 2020] Variational Inference for Graph Convolutional Networks in the Absence of Graph Data and Adversarial Settings](https://proceedings.neurips.cc/paper/2020/file/d882050bb9eeba930974f596931be527-Paper.pdf) [[Code]](https://github.com/ebonilla/VGCN)|
|GLNN|[[ICME 2020] Exploring structure-adaptive graph learning for robust semi-supervised classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9102726&casa_token=3wh5N07uxPsAAAAA:Rl6NUTj2cr0qwdK6WL3WFx9Gk5g1wgpqkCbDyI332aeR_8-qThEMYs98VVuhtljGsjIYfPag5bw)|
|GNN-Guard|[[NeurIPS 2020] Gnnguard: Defending graph neural networks against adversarial attacks](https://proceedings.neurips.cc/paper/2020/file/690d83983a63aa1818423fd6edd3bfdb-Paper.pdf) [[Code]](https://github.com/mims-harvard/GNNGuard)|
|Grale|[[KDD 2020] Grale: Designing Networks for Graph Learning](https://dl.acm.org/doi/pdf/10.1145/3394486.3403302)|
|TO-GNN|[[IJCAI 2019] Topology Optimization based Graph Convolutional Network](https://www.ijcai.org/proceedings/2019/0563.pdf)|
|LDS|[[ICML 2019] Learning discrete structures for graph neural networks](http://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf) [[Code]](https://github.com/lucfra/LDS-GNN)|
|BGCN|[[AAAI 2019] Bayesian graph convolutional neural networks for semi-supervised classification](https://ojs.aaai.org/index.php/AAAI/article/download/4531/4409) [[Code]](https://github.com/huawei-noah/BGCN)|
|PG-LEARN|[[CIKM 2018] A Quest for Structure: Jointly Learning the Graph Structure and Semi-Supervised Classification](https://dl.acm.org/doi/pdf/10.1145/3269206.3271692) [[Code]](https://github.com/LingxiaoShawn/PG-Learn)|
|DGM|[[arXiv] Differentiable Graph Module (DGM) for Graph Convolutional Networks](https://arxiv.org/pdf/2002.04999.pdf)|


### Optimal Feature Learning
|Name|Paper|
|---|---|
|AirGNN|[[NeurIPS 2021] Graph Neural Networks with Adaptive Residual](https://proceedings.neurips.cc/paper/2021/file/50abc3e730e36b387ca8e02c26dc0a22-Paper.pdf) [[Code]](https://github.com/lxiaorui/AirGNN)|
|FP|[[arXiv 2021] On the Unreasonable Effectiveness of Feature propagation in Learning on Graphs with Missing Node Features](https://arxiv.org/pdf/2111.12128.pdf)|
|GCNMF|[[FGCS 2021] Graph convolutional networks for graphs containing missing features](https://www.sciencedirect.com/science/article/pii/S0167739X20330405)|


## GraphDA for Low-Resource Graph Learning

### Graph Self-Supervised Learning
|Name|Paper
|---|---|
|S^3-CL|[[arXiv 2022] Structural and Semantic Contrastive Learning for Self-supervised Node Representation Learning](https://arxiv.org/pdf/2202.08480.pdf)|
|ARIEL|[[TheWebConf 2022] Adversarial Graph Contrastive Learning with Information Regularization](https://arxiv.org/pdf/2202.06491.pdf)|
|GCA|[[TheWebConf 2021] Graph Contrastive Learning with Adaptive Augmentation](https://dl.acm.org/doi/pdf/10.1145/3442381.3449802?casa_token=5M-ljIDkVKIAAAAA:rGnCn_5bwBtngfI-JN_gsJOCO9FbaOziPyQ8_jSzL6l4u5CNZsl0f7NhAMpZCNU8c3X4Ndq_6wbvgA) [[Code]](https://github.com/CRIPAC-DIG/GCA)|
|CSSL|[[AAAI 2021] Contrastive self-supervised learning for graph classification](https://www.aaai.org/AAAI21Papers/AAAI-7017.ZengJ.pdf)|
|GraphCL|[[NeurIPS 2020] Graph contrastive learning with augmentations](https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf) [[Code]](https://github.com/Shen-Lab/GraphCL)|
|GPT-GNN|[[KDD 2020] Gpt-gnn: Generative pre-training of graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403237?ref=https://githubhelp.com) [[Code]](https://github.com/acbull/GPT-GNN)|
|GCC|[[KDD 2020] GCC: Graph contrastive coding for graph neural network pre-training](https://dl.acm.org/doi/pdf/10.1145/3394486.3403168?casa_token=sv5NdkZuYWcAAAAA:9Rj_x-RZpbNLkYVqzA13ENULyAWA8tkYzbWoKFo5iXJSixCa5wv_S6p8RGyI5mAQCl_rKJRfT6xbuw) [[Code]](https://github.com/THUDM/GCC)|
|MTL|[[ICML 2020] When does self-supervision help graph convolutional networks?](http://proceedings.mlr.press/v119/you20a/you20a.pdf)] [[Code]](https://github.com/Shen-Lab/SS-GCNs)|
|GraphBert|[[arXiv 2020] Graph-bert: Only attention is needed for learning graph representations](https://arxiv.org/pdf/2001.05140.pdf?ref=https://githubhelp.com)|
|Pre-train|[[arXiv 2019] Pre-training graph neural networks for generic structural feature extraction](https://arxiv.org/pdf/1905.13728.pdf)|



### Graph Self/Co-Training
|Name|Paper|
|---|---|
|Meta-PN|[[AAAI 2022] Meta Propagation Networks for Graph Few-shot Semi-supervised Learning](https://arxiv.org/pdf/2112.09810.pdf) [[Code]](https://github.com/kaize0409/Meta-PN)
|NRGNN|[[KDD 2021] NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs](https://dl.acm.org/doi/pdf/10.1145/3447548.3467364?casa_token=vZAPtLCtp8IAAAAA:SdDMsdTcqjEGqcM5gjKc8U4Hk7oNRfsQtv9UouHr6KsOyzp7AzVh5UkjAeYElZhmBlMZ__jJeUOC-w) [[Code]](https://github.com/EnyanDai/NRGNN)|
|PTA|[[TheWebConf 2021] On the equivalence of decoupled graph convolution network and label propagation](https://dl.acm.org/doi/pdf/10.1145/3442381.3449927?casa_token=JWbrqhSlv-oAAAAA:H2Lo2oIXotHolk4d-QBP6Sal6Rzhnrdx6DzOOI8GtlAIUf7AnOxiCzETr0YqCkRklBaErZ_8IhbQ-A) [[Code]](https://github.com/DongHande/PT_propagation_then_training)|
|CGCN|[[AAAI 2020] Collaborative graph convolutional networks: Unsupervised learning meets semi-supervised learning](https://ojs.aaai.org/index.php/AAAI/article/download/5843/5699)|
|M3S|[[AAAI 2020] Multi-stage self-supervised learning for graph convolutional networks on graphs with few labeled nodes](https://ojs.aaai.org/index.php/AAAI/article/download/6048/5904) [[Code]](https://github.com/datake/M3S)|
|Co-GCN|[[AAAI 2020] Co-GCN for Multi-View Semi-Supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/view/5901/5757)|
|ST-GCNs|[[AAAI 2018] Deeper insights into graph convolutional networks for semi-supervised learning](http://www4.comp.polyu.edu.hk/~csxmwu/papers/AAAI-2018-GCN.pdf) [[Code]](https://github.com/liqimai/gcn/tree/AAAI-18/)|

### Graph Interpolation
|Name|Paper|
|---|---|
|Graph Transplant|[[AAAI 2022] Graph Transplant: Node Saliency-Guided Graph Mixup with Local Structure Preservation](https://arxiv.org/pdf/2111.05639.pdf)|
|G-Mixup|[[TheWebConf 2021] Mixup for Node and Graph Classification](https://dl.acm.org/doi/pdf/10.1145/3442381.3449796) [[Code]](https://github.com/vanoracai/MixupForGraph)|
|GraphMix|[[AAAI 2021] GraphMix: Improved Training of GNNs for Semi-Supervised Learning](https://arxiv.org/pdf/1909.11715.pdf) [[Code]](https://github.com/vikasverma1077/GraphMix)|
|GraphMixup|[[arXiv 2021] GraphMixup: Improving Class-Imbalanced Node Classification on Graphs by Self-supervised Context Prediction](https://arxiv.org/pdf/2106.11133.pdf)|
|ifMixup|[[arXiv 2021] Intrusion-Free Graph Mixup](https://arxiv.org/pdf/2110.09344.pdf)|

### Graph Consistency Training
|Name|Paper|
|---|---|
|MH-Aug|[[NeurIPS 2021] Metropolis-Hastings Data Augmentation for Graph Neural Networks](https://proceedings.neurips.cc/paper/2021/file/9e7ba617ad9e69b39bd0c29335b79629-Paper.pdf)|
|GRAND|[[NeurIPS 2020] Graph Random Neural Network for Semi-Supervised Learning on Graphs](https://proceedings.neurips.cc/paper/2020/file/fb4c835feb0a65cc39739320d7a51c02-Paper.pdf) [[Code]](https://github.com/THUDM/GRAND)|
|NodeAug|[[KDD 2020] NodeAug: Semi-Supervised Node Classification with Data Augmentation](https://dl.acm.org/doi/pdf/10.1145/3394486.3403063)|

## Adversarial Training on Graphs
|Name|Paper|
|---|---|
|FLAG|[[arXiv] Flag: Adversarial data augmentation for graph neural networks](https://arxiv.org/pdf/2010.09891.pdf)|
|BVAT|[[arXiv] Batch virtual adversarial training for graph convolutional networks](https://arxiv.org/pdf/1902.09192.pdf)|
|GraphAT|[[TKDE 2019] Graph adversarial training: Dynamically regularizing based on graph structure](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8924766&casa_token=hBUgxc-C6B8AAAAA:-p98b_oAub0Pno9bVP5uq9jOI8on3F0XvZAU2zi6LUnzfaaE9_PcYHPnpAv7pSURqm9CzEHaAJ4) [[Code]](https://github.com/fulifeng/GraphAT)|
|GCNVAT|[[PRCV 2019] Virtual adversarial training on graph convolutional networks in node classification](https://arxiv.org/pdf/1902.11045.pdf)|

More works on **adversarial attack and defense on graphs** can be found in this [survey](https://arxiv.org/pdf/1812.10528.pdf).

## Other Directions

### GraphDA for Graph Imbalanced Learning
|Name|Paper|
|---|---|
|GraphSMOTE|[[WSDM 2021] GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3437963.3441720?casa_token=Wf5mpNwPYBYAAAAA:JAHEAtk0C48YGn8MGiDQY2xr8NTg-tJIpi6PhAgWHOBa8_OeZ_0mEBRdm5C7XGlSc4TXWtenaVKEgA) [[Code]](https://github.com/TianxiangZhao/GraphSmote)|
|ImGAGN|[[KDD 2021] ImGAGN: Imbalanced Network Embedding via Generative Adversarial Graph Networks](https://dl.acm.org/doi/pdf/10.1145/3447548.3467334?casa_token=Ty8EPIVMU30AAAAA:_kNvYoQfWlOvM_sUEZCSpS-hC9bVEux7y0jkLxjwvRtOVSIiU8Cfq2RgeYgOEmZV8ELcBKlzgkHYTw) [[Code]](https://github.com/Leo-Q-316/ImGAGN)|
|GraphMixup|[[arXiv 2021] GraphMixup: Improving Class-Imbalanced Node Classification on Graphs by Self-supervised Context Prediction](https://arxiv.org/pdf/2106.11133.pdf)|
|DR-GCN|[[IJCAI 2020] Multi-class imbalanced graph convolutional network learning](https://par.nsf.gov/servlets/purl/10199469)|

### GraphDA for Learning on Heterophilic Graphs
|Name|Paper|
|---|---|
|WRGAT|[[KDD 2021] Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns](https://dl.acm.org/doi/pdf/10.1145/3447548.3467373) [[Code]](https://github.com/susheels/gnns-and-local-assortativity)|
|SGATs|[[TKDE 2021] Sparse graph attention networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9399811&casa_token=7dOFWpPz370AAAAA:jq9qmU85ggjBEsl2J4KqTG4fgU1SJN2_MOSqG9EMs-EValyhoqZjJz7SIR8eh0KUq4NYjxNiCR0&tag=1) [[Code]](https://github.com/Yangyeeee/SGAT)|
|FAGCN|[[AAAI 2021] Beyond Low-frequency Information in Graph Convolutional Networks](https://www.aaai.org/AAAI21Papers/AAAI-10091.BoD.pdf) [[Code]](http://www.shichuan.org/dataset/FAGCN_27.zip)|
|Geom-GCN|[[ICLR 2019] Geom-GCN: Geometric Graph Convolutional Networks](https://openreview.net/forum?id=S1e2agrFvS) [[Code]](https://github.com/graphdml-uiuc-jlu/geom-gcn)|
