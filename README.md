# CNN image encoder 

 a Toy implementation for ARCFace style image encoder for 'cosine similarity' measures 
 
 - The final goal is for map-based VPS project. 
 - Here we use MNIST for test purpose and not inlcude the final code for commerical product.
 - losses: cross-entroy, constrastive, and triplet 
 - similarity measure: cosine similarity (most common measure for general comparison bewteen two vectors.)
 
 

# Installation

Requires [pytorch](http://pytorch.org/) 0.4 with torchvision 0.2.1

For pytorch 0.3 compatibility checkout tag torch-0.3.1

# Code structure
- main.y  
    - create dataset, dataloader, model, then training, and finally evaluate and visualize  


- datasets.py
    - (MNIST)            - mnist dataset of pytorch  
    - SiameseMNIST class - wrapper for a MNIST-like dataset, returning random positive and negative pairs
    - *TripletMNIST* class - wrapper for a MNIST-like dataset, returning random triplets (anchor, positive and negative)
    - ClassPairAnalysisDataset - wrapper for a MNIST-like dataset, returning a pair and their indices (for cosine sim between class i and j)
    - BalancedBatchSampler class - BatchSampler for data loader, randomly chooses *n_classes* and *n_samples* from each class based on labels

- networks.py
    - EmbeddingNet - base network for encoding images into embedding vector (32 conv 5x5 -> PReLU -> MaxPool 2x2 -> 64 conv 5x5 -> PReLU -> MaxPool 2x2 -> Dense 256 -> PReLU -> Dense 256 -> PReLU -> Dense 2).
    - *EmbeddingNetL2* - normalization layer added to EmbeddingNet, so that the vectors are located in hyper-shpere. 
    - ClassificationNet - wrapper for an embedding network, adds a fully connected layer and log softmax for classification
    - SiameseNet - wrapper for an embedding network, processes pairs of inputs
    - *TripletNet* - wrapper for an embedding network, processes triplets of inputs
  
- losses.py

  - ContrastiveLoss - contrastive loss for pairs of embeddings and pair target (same/different)
  - *TripletLoss* - triplet loss for triplets of embeddings
  - OnlineContrastiveLoss - contrastive loss for a mini-batch of embeddings. Uses a *PairSelector* object to find positive and negative pairs within a mini-batch using ground truth class labels and computes contrastive loss for these pairs
  - OnlineTripletLoss - triplet loss for a mini-batch of embeddings. Uses a *TripletSelector* object to find triplets within a mini-batch using ground truth class labels and computes triplet loss

- trainer.py
  - *fit* - unified function for training a network with different number of inputs and different types of loss functions

- metrics.py
  - Sample metrics that can be used with fit function from trainer.py
  
- utils.py

    - PairSelector - abstract class defining objects generating pairs based on embeddings and ground truth class labels. Can be used with OnlineContrastiveLoss.

        - AllPositivePairSelector, HardNegativePairSelector - PairSelector implementations
    - TripletSelector - abstract class defining objects generating triplets based on embeddings and ground truth class labels. Can be used with OnlineTripletLoss.
        - AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector - TripletSelector implementations


We'll go through learning supervised feature embeddings using different loss functions on MNIST dataset. This is just for visualization purposes, thus we'll be using 2-dimensional embeddings which isn't the best choice in practice.


# References

[1] Raia Hadsell, Sumit Chopra, Yann LeCun, [Dimensionality reduction by learning an invariant mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), CVPR 2006

[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering.](https://arxiv.org/abs/1503.03832) CVPR 2015

[3] Alexander Hermans, Lucas Beyer, Bastian Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737), 2017

[4] Brandon Amos, Bartosz Ludwiczuk, Mahadev Satyanarayanan, [OpenFace: A general-purpose face recognition library with mobile applications](http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf), 2016

[5] Yi Sun, Xiaogang Wang, Xiaoou Tang, [Deep Learning Face Representation by Joint Identification-Verification](http://papers.nips.cc/paper/5416-deep-learning-face-representation-by-joint-identification-verification), NIPS 2014


[6] Deng, Jiankang, et al. [Arcface: Additive angular margin loss for deep face recognition] (https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) CVPR 2019.
