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

#### Graph Self/Co-Training
|Name|Paper|Code
|---|---|---|
|Meta-PN|[[AAAI 2022] Meta Propagation Networks for Few-shot Semi-supervised Graph Learning](https://arxiv.org/pdf/2112.09810.pdf)|[PyTorch](https://github.com/kaize0409/Meta-PN)


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
