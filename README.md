## Graph Data Augmentation (GraphDA) for Deep Graph Learning
The repository contains links primarily to conference and journal publications about **graph data augmentation for deep graph learning**. If you find this repository useful, please kindly cite the following paper:

```
@article{ding2022data,
  title={Data Augmentation for Deep Graph Learning: A Survey},
  author={Ding, Kaize and Xu, Zhe and Tong, Hanghang and Liu, Huan},
  journal={arXiv preprint arXiv:2202.08235},
  year={2022}
}
```

### GraphDA for Optimal Graph Learning

#### Optimal Structural Learning
|Name|Paper
|---|---|
|GAUG|[[AAAI 2021] Data Augmentation for Graph Neural Networks](https://www.aaai.org/AAAI21Papers/AAAI-10012.ZhaoT.pdf) [[Code]](https://github.com/zhao-tong/GAug)|
|AdaEdge|[[AAAI 2020] Measuring and relieving the over-smoothing problem for graph neural networks from the topological view](https://ojs.aaai.org/index.php/AAAI/article/download/5747/5603)|
|IDGL|[[NeurIPS 2020] Iterative deep graph learning for graph neural networks: Better and robust node embeddings](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) [[Code]](https://github.com/hugochan/IDGL)|
|TO-GNN|[[IJCAI 2019] Topology Optimization based Graph Convolutional Network](https://www.ijcai.org/proceedings/2019/0563.pdf)|
|Pro-GNN|[[KDD 2020] Graph structure learning for robust graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Code]](https://github.com/ChandlerBang/Pro-GNN)|
|Gasoline|[[TheWebConf 2022] Graph Sanitation with Application to Node Classification](https://arxiv.org/pdf/2105.09384.pdf?ref=https://githubhelp.com)] [[Code]](https://github.com/pricexu/GASOLINE)|
|LDS|[[ICML 2019] Learning discrete structures for graph neural networks](http://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf) [[Code]](https://github.com/lucfra/LDS-GNN)|
|NeuralSparse|[[ICML 2020] Robust graph representation learning via neural sparsification](http://proceedings.mlr.press/v119/zheng20d/zheng20d.pdf) |
|PTDNet|[[WSDM 2021] Learning to drop: Robust graph neural network via topological denoising](https://dl.acm.org/doi/pdf/10.1145/3437963.3441734?casa_token=p-KUZTPYFS0AAAAA:mYTX35RRO_p_EnC_ohXaz0dhk2uTs-cxVDqPD75GXuNvq_W-Uv2IA5yM3PTxp9JLAHiX4ycVaql2QQ) [[Code]](https://github.com/flyingdoog/PTDNet)|
|BGCN|[[AAAI 2019] Bayesian graph convolutional neural networks for semi-supervised classification](https://ojs.aaai.org/index.php/AAAI/article/download/4531/4409) [[Code]](https://github.com/huawei-noah/BGCN)|
|GEN|[[TheWebConf 2021] Graph Structure Estimation Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3442381.3449952?casa_token=4b0qCTk088EAAAAA:sgU4Z5pS1W0dJ42SUo063Psw6XrNDTOor7EnX98HZwGeyj9_ZW_znUo-pb1S6I82_28EqBMd0wQo9g) [[Code]](https://github.com/BUPT-GAMMA/Graph-Structure-Estimation-Neural-Networks)|
|GIB-G|[[ICLR 2020] Graph Information Bottleneck for Subgraph Recognition](https://openreview.net/forum?id=bM4Iqfg8M2k)|
|GIB-N|[[NeurIPS 2020] Graph Information Bottleneck](https://proceedings.neurips.cc/paper/2020/file/ebc2aa04e75e3caabda543a1317160c0-Paper.pdf) [[Code]](https://github.com/snap-stanford/GIB)|

#### Optimal Feature Learning
|Name|Paper|
|---|---|
|AirGNN|[[NeurIPS 2021] Graph Neural Networks with Adaptive Residual](https://proceedings.neurips.cc/paper/2021/file/50abc3e730e36b387ca8e02c26dc0a22-Paper.pdf) [[Code]](https://github.com/lxiaorui/AirGNN)|
|FP|[[arXiv] On the Unreasonable Effectiveness of Feature propagation in Learning on Graphs with Missing Node Features](https://arxiv.org/pdf/2111.12128.pdf)|
|GCNMF|[[FGCS 2021] Graph convolutional networks for graphs containing missing features](https://www.sciencedirect.com/science/article/pii/S0167739X20330405)|


### GraphDA for Low-resource Graph Learning

#### Graph Self-Supervised Learning
|Name|Paper
|---|---|
|GPT-GNN|[[KDD 2022] Gpt-gnn: Generative pre-training of graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403237?ref=https://githubhelp.com) [[Code]](https://github.com/acbull/GPT-GNN)|
|MTL|[[ICML 2020] When does self-supervision help graph convolutional networks?](http://proceedings.mlr.press/v119/you20a/you20a.pdf)] [[Code]](https://github.com/Shen-Lab/SS-GCNs)|
|Pre-train|[[arXiv] Pre-training graph neural networks for generic structural feature extraction](https://arxiv.org/pdf/1905.13728.pdf)|
|GraphBert|[[arXiv] Graph-bert: Only attention is needed for learning graph representations](https://arxiv.org/pdf/2001.05140.pdf?ref=https://githubhelp.com)|
|Gcc|[[KDD 2020] Gcc: Graph contrastive coding for graph neural network pre-training](https://dl.acm.org/doi/pdf/10.1145/3394486.3403168?casa_token=sv5NdkZuYWcAAAAA:9Rj_x-RZpbNLkYVqzA13ENULyAWA8tkYzbWoKFo5iXJSixCa5wv_S6p8RGyI5mAQCl_rKJRfT6xbuw) [[Code]](https://github.com/THUDM/GCC)|
|GraphCL|[[NeurIPS 2020] Graph contrastive learning with augmentations](https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf) [[Code]](https://github.com/Shen-Lab/GraphCL)|
|GCA|[[TheWebConf 2021] Graph Contrastive Learning with Adaptive Augmentation](https://dl.acm.org/doi/pdf/10.1145/3442381.3449802?casa_token=5M-ljIDkVKIAAAAA:rGnCn_5bwBtngfI-JN_gsJOCO9FbaOziPyQ8_jSzL6l4u5CNZsl0f7NhAMpZCNU8c3X4Ndq_6wbvgA) [[Code]](https://github.com/CRIPAC-DIG/GCA)|
|CSSL|[[AAAI 2021] Contrastive self-supervised learning for graph classification](https://www.aaai.org/AAAI21Papers/AAAI-7017.ZengJ.pdf)|
|ARIEL|[[TheWebConf 2022] Adversarial Graph Contrastive Learning with Information Regularization](https://arxiv.org/pdf/2202.06491.pdf)|

#### Graph Self/Co-Training
|Name|Paper|
|---|---|
|Meta-PN|[[AAAI 2022] Meta Propagation Networks for Few-shot Semi-supervised Graph Learning](https://arxiv.org/pdf/2112.09810.pdf) [[Code]](https://github.com/kaize0409/Meta-PN)
|GST|[[AAAI 2018] Deeper insights into graph convolutional networks for semi-supervised learning](http://www4.comp.polyu.edu.hk/~csxmwu/papers/AAAI-2018-GCN.pdf) [[Code]](https://github.com/liqimai/gcn/tree/AAAI-18/)|
|CGCN|[[AAAI 2020] Collaborative graph convolutional networks: Unsupervised learning meets semi-supervised learning](https://ojs.aaai.org/index.php/AAAI/article/download/5843/5699)|
|M3S|[[AAAI 2020] Multi-stage self-supervised learning for graph convolutional networks on graphs with few labeled nodes](https://ojs.aaai.org/index.php/AAAI/article/download/6048/5904) [[Code]](https://github.com/datake/M3S)|
|NRGNN|[[KDD 2021] NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs](https://dl.acm.org/doi/pdf/10.1145/3447548.3467364?casa_token=vZAPtLCtp8IAAAAA:SdDMsdTcqjEGqcM5gjKc8U4Hk7oNRfsQtv9UouHr6KsOyzp7AzVh5UkjAeYElZhmBlMZ__jJeUOC-w) [[Code]](https://github.com/EnyanDai/NRGNN)|
|PTA|[[TheWebConf 2021] On the equivalence of decoupled graph convolution network and label propagation](https://dl.acm.org/doi/pdf/10.1145/3442381.3449927?casa_token=JWbrqhSlv-oAAAAA:H2Lo2oIXotHolk4d-QBP6Sal6Rzhnrdx6DzOOI8GtlAIUf7AnOxiCzETr0YqCkRklBaErZ_8IhbQ-A) [[Code]](https://github.com/DongHande/PT_propagation_then_training)|
|Co-GCN|[[AAAI 2020] Co-GCN for Multi-View Semi-Supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/view/5901/5757)|

#### Graph Interpolation
|Name|Paper|
|---|---|
|GraphMix|[[AAAI 2021] GraphMix: Improved Training of GNNs for Semi-Supervised Learning](https://arxiv.org/pdf/1909.11715.pdf) [[Code]](https://github.com/vikasverma1077/GraphMix)|
|G-Mixup|[[TheWebConf 2021] Mixup for Node and Graph Classification](https://dl.acm.org/doi/pdf/10.1145/3442381.3449796) [[Code]](https://github.com/vanoracai/MixupForGraph)|
|GraphMixup|[[arXiv] GraphMixup: Improving Class-Imbalanced Node Classification on Graphs by Self-supervised Context Prediction](https://arxiv.org/pdf/2106.11133.pdf)|
|ifMixup|[[arXiv] Intrusion-Free Graph Mixup](https://arxiv.org/pdf/2110.09344.pdf)|
|Graph Transplant|[[AAAI 2022] Graph Transplant: Node Saliency-Guided Graph Mixup with Local Structure Preservation](https://arxiv.org/pdf/2111.05639.pdf)|

### Underexplored Directions

#### GraphDA for Graph Imbalanced Learning
|Name|Paper|
|---|---|
|GraphSMOTE|[[WSDM 2021] GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3437963.3441720?casa_token=Wf5mpNwPYBYAAAAA:JAHEAtk0C48YGn8MGiDQY2xr8NTg-tJIpi6PhAgWHOBa8_OeZ_0mEBRdm5C7XGlSc4TXWtenaVKEgA) [[Code]](https://github.com/TianxiangZhao/GraphSmote)|
|ImGAGN|[[KDD 2021] ImGAGN: Imbalanced Network Embedding via Generative Adversarial Graph Networks](https://dl.acm.org/doi/pdf/10.1145/3447548.3467334?casa_token=Ty8EPIVMU30AAAAA:_kNvYoQfWlOvM_sUEZCSpS-hC9bVEux7y0jkLxjwvRtOVSIiU8Cfq2RgeYgOEmZV8ELcBKlzgkHYTw) [[Code]](https://github.com/Leo-Q-316/ImGAGN)|
|GraphMixup|[[arXiv] GraphMixup: Improving Class-Imbalanced Node Classification on Graphs by Self-supervised Context Prediction](https://arxiv.org/pdf/2106.11133.pdf)|
|DR-GCN|[[IJCAI 2020] Multi-class imbalanced graph convolutional network learning](https://par.nsf.gov/servlets/purl/10199469)|

#### GraphDA for Learning on Heterophilic Graphs
|Name|Paper|
|---|---|
|WRGAT|[[KDD 2021] Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns](https://dl.acm.org/doi/pdf/10.1145/3447548.3467373) [[Code]](https://github.com/susheels/gnns-and-local-assortativity)|
|Geom-GCN|[[ICLR 2019] Geom-GCN: Geometric Graph Convolutional Networks](https://openreview.net/forum?id=S1e2agrFvS) [[Code]](https://github.com/graphdml-uiuc-jlu/geom-gcn)|
|SGATs|[[TKDE 2021] Sparse graph attention networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9399811&casa_token=7dOFWpPz370AAAAA:jq9qmU85ggjBEsl2J4KqTG4fgU1SJN2_MOSqG9EMs-EValyhoqZjJz7SIR8eh0KUq4NYjxNiCR0&tag=1) [[Code]](https://github.com/Yangyeeee/SGAT)|
|FAGCN|[[AAAI 2021] Beyond Low-frequency Information in Graph Convolutional Networks](https://www.aaai.org/AAAI21Papers/AAAI-10091.BoD.pdf) [[Code]](http://www.shichuan.org/dataset/FAGCN_27.zip)|
