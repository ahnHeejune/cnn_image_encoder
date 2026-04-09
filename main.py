'''

   3 Losses: cross-entrpy, contrastive, and triplet, 
   
   arcface 
   
   - 2D space visualization    
   - cosine similarity 


'''

# 1. result analsyis and visualization  
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from trainer import fit, test_epoch

cuda = torch.cuda.is_available()
n_classes = 10
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

# visualize 
def plot_embeddings(embeddings, targets, xlim=None, ylim=None):

    plt.figure(figsize=(10,10))
    for i in range(n_classes): # plot each embedding vector in the same color 
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.show()

# embedding and label 
def extract_embeddings(dataloader, model, embedding_dim = 2):

    '''
        embed all data of dataloader using model's embedding network 
        return the embedding (N,2) and estimated labels (N)
    '''
       
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), embedding_dim))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:  # one batch 
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels

def extract_plot_embedding(model, loader, embedding_dim = 2):

    # validation data
    embeddings, labels = extract_embeddings(loader, model, embedding_dim)
    if embedding_dim > 2: # 고차원을 2차원으로 축소
        from sklearn.manifold import TSNE       
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
          
    plot_embeddings(embeddings_2d, labels)


def _evaluate_similarity(model, loader, type_loader):  
    
    from losses import NullLoss
    from metrics import CosineSimilarityMetric
    
    loss_fn = NullLoss()
    cos_metric = CosineSimilarityMetric()
    loss, metrics = test_epoch(loader, model, loss_fn, cuda, [cos_metric])
    print(f"Final Average Cosine Similarity: {cos_metric.value():.4f}")

def evaluate_similarity(model, loader):

    if hasattr(model, 'embedding_net'):
        encoder = model.embedding_net
    else:
        encoder = model

    encoder.eval()
     # 유사도 합계와 개수를 저장할 행렬 (평균 계산용)
    sim_matrix_sum = np.zeros((10, 10))
    sim_matrix_count = np.zeros((10, 10))
    
    with torch.no_grad():
        for (img_i, img_j), (label_i, label_j) in loader:
            img_i, img_j = img_i.cuda(), img_j.cuda()
            
            # L2 Normalized Embeddings
            z_i = encoder(img_i)
            z_j = encoder(img_j)
            
            # Cosine Similarity (Dot product of L2 normalized vectors)
            sim = torch.sum(z_i * z_j, dim=1)
            
             # 결과를 행렬 인덱스에 누적
            for k in range(len(sim)):
                row = int(label_i[k])
                col = int(label_j[k])
                sim_matrix_sum[row, col] += sim[k].item()
                sim_matrix_count[row, col] += 1

    # 평균 행렬 계산
    # 0으로 나누는 것을 방지하기 위해 count가 0인 곳은 그대로 둠
    avg_sim_matrix = np.divide(sim_matrix_sum, sim_matrix_count, 
                               out=np.zeros_like(sim_matrix_sum), 
                               where=sim_matrix_count!=0)

    # 결과 출력
    print("\n" + "="*50)
    print("  Cosine Similarity Matrix (10x10 Average)")
    print("="*50)
    
    # Numpy 출력 옵션 설정 (소수점 2자리)
    np.set_printoptions(precision=2, suppress=True)
    print(avg_sim_matrix)
    
    # 또는 조금 더 읽기 쉽게 한 줄씩 출력
    # print("\nDetailed Matrix View:")
    # for row in avg_sim_matrix:
    #     print(" ".join([f"{val:6.2f}" for val in row]))
        
    return avg_sim_matrix
    

# 1. load data set 
def load_mnist():

    from torchvision.datasets import MNIST # MNIST dataset (download automatically)
    from torchvision import transforms

    mean, std = 0.1307, 0.3081
    MNIST_dataset_path = 'datasets/MNIST'
    train_dataset = MNIST(MNIST_dataset_path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
    test_dataset = MNIST(MNIST_dataset_path, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
    print(f"MNIST dataset prepared: train:{len(train_dataset)}, test:{len(test_dataset)}")

    return train_dataset, test_dataset

def create_data_loader(dataset, shuffle, batch_size = 256):

    ''' single data loader '''
    
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    return data_loader

# 1. entropy loss based
def train_crossentropyloss(train_loader, test_loader, embedding_dim = 2, n_epochs = 20, bNorm = False):
        
    # Set up the network and training parameters
    from networks import EmbeddingNet, EmbeddingNetL2, ClassificationNet
    from metrics import AccumulatedAccuracyMetric

    embedding_net = EmbeddingNetL2(embedding_dim) if bNorm else EmbeddingNet(embedding_dim)   
    model = ClassificationNet(embedding_net, n_classes=n_classes)
    if cuda:
        model.cuda()
    loss_fn = torch.nn.NLLLoss()
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 50

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

    return model 


def create_contrastive_data_loader(dataset, shuffle, batch_size = 128):

    ''' (img1, img2) and postive/negative '''

    from datasets import SiameseMNIST
    
    contrative_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    contrative_loader = torch.utils.data.DataLoader(contrative_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    return contrative_loader

def train_contrastiveloss(contrastive_train_loader, contrastive_test_loader, embedding_dim =2, n_epochs = 20, bNorm = False):
   
    # Set up the network and training parameters
    from networks import EmbeddingNet, EmbeddingNetL2, SiameseNet
    from losses import ContrastiveLoss

    # SiamesNetwork( EmbeddingNetWork -> )
    embedding_net = EmbeddingNetL2(embedding_dim) if bNorm else EmbeddingNet(embedding_dim)      
    model = SiameseNet(embedding_net)
    if cuda:
       model.cuda()
    margin = 1.
    loss_fn = ContrastiveLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 100

    # train 
    fit(contrastive_train_loader, contrastive_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    return model 


def create_triplet_data_loader(dataset, shuffle,  batch_size = 128):

    # Set up data loaders
    from datasets import TripletMNIST

    triplet_dataset = TripletMNIST(dataset) # Returns triplets of images
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
   
    return triplet_loader

def train_tripletloss(triplet_train_loader, triplet_test_loader, embedding_dim = 2, n_epochs = 20, bNorm = False):

    # Set up the network and training parameters
    from networks import EmbeddingNet, EmbeddingNetL2, TripletNet
    from losses import TripletLoss

    margin = 1.
    embedding_net = EmbeddingNetL2(embedding_dim) if bNorm else EmbeddingNet(embedding_dim)   
   
    model = TripletNet(embedding_net)
    if cuda:
       model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 100

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    return model 


if __name__ == "__main__":

 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=int, default=2, help="0 (cross-entropy), 1 (constrastive), 2 (triplet)")
    parser.add_argument('--nepoch', type=int, default=20, help="training epoch")
    parser.add_argument('--nfeats', type=int, default=128, help="encoding dimension")
    parser.add_argument('--norm', type=bool, default=True, help="Normalized Encoder (Arcface style) or not")

    opt = parser.parse_args()


    
    # Set up data loaders
    train_dataset, test_dataset = load_mnist()
   
    if opt.loss ==0:
        train_loader = create_data_loader(train_dataset, shuffle = True)
        test_loader = create_data_loader(test_dataset, shuffle = False)
        model = train_crossentropyloss(train_loader, test_loader, opt.nfeats, opt.nepoch, opt.norm)
    elif opt.loss == 1:
        # training
        contrastive_train_loader = create_contrastive_data_loader(train_dataset, shuffle = True)
        contrastive_test_loader  = create_contrastive_data_loader(test_dataset, shuffle = False)       
        model = train_contrastiveloss(contrastive_train_loader, contrastive_test_loader, opt.nfeats, opt.nepoch, opt.norm)
    elif opt.loss == 2:
        # training
        triplet_train_loader = create_triplet_data_loader(train_dataset, shuffle = True)
        triplet_test_loader  = create_triplet_data_loader(test_dataset, shuffle = False)
        model = train_tripletloss(triplet_train_loader, triplet_test_loader, opt.nfeats, opt.nepoch, opt.norm)
    else:
        print("unsupported loss")
        exit()
        
    # evaluation 
    eval_train_loader = create_data_loader(train_dataset,shuffle = True)
    eval_test_loader = create_data_loader(test_dataset, shuffle = False)
    #extract_plot_embedding(model, eval_train_loader, opt.nfeats)   
    extract_plot_embedding(model, eval_test_loader, opt.nfeats)   
   
    if opt.norm:    
        from datasets import ClassPairAnalysisDataset
        sim_dataset = ClassPairAnalysisDataset(test_dataset)
        sim_loader = create_data_loader(sim_dataset, shuffle = False)
        evaluate_similarity(model, sim_loader)     


