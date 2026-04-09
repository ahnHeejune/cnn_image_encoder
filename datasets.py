import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class ClassPairAnalysisDataset(Dataset):
    """
    MNIST 숫자 클래스 i와 j 사이의 유사도 측정을 위한 전용 데이터셋
    """
    def __init__(self, mnist_dataset):
        """
        mnist_dataset: load_mnist() 등으로 로드된 원본 MNIST 데이터셋
        """
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        self.transform = mnist_dataset.transform

     
        
        # 0부터 9까지 각 클래스별로 데이터의 인덱스를 미리 정리해둡니다.
        self.labels_set = set(self.targets.numpy())
        self.label_to_indices = {label: np.where(self.targets.numpy() == label)[0]
                                 for label in self.labels_set}
        
        # 분석을 위한 고정된 클래스 쌍 생성 (0 vs 0, 0 vs 1, ... 9 vs 9)
        # 총 100개의 조합을 만듭니다.
        self.analysis_pairs = []
        for i in range(10):
            for j in range(10):
                # 각 클래스에서 분석에 사용할 대표 이미지 하나를 고정(예: 첫 번째 샘플)해서 뽑습니다.
                idx_i = self.label_to_indices[i][0]
                idx_j = self.label_to_indices[j][0]
                self.analysis_pairs.append((idx_i, idx_j, i, j))

    def __getitem__(self, index):
        idx_i, idx_j, label_i, label_j = self.analysis_pairs[index]
        
        img_i = self.data[idx_i]
        img_j = self.data[idx_j]

        # MNIST 텐서를 PIL 이미지로 변환 (MNIST는 Grayscale)
        img_i = Image.fromarray(img_i.numpy(), mode='L')
        img_j = Image.fromarray(img_j.numpy(), mode='L')

        if self.transform:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        # 이미지 쌍과 함께 실제 숫자 라벨을 반환합니다.
        return (img_i, img_j), (label_i, label_j)

    def __len__(self):
        return len(self.analysis_pairs)
        

class _____ClassPairAnalysisDataset(Dataset):
    """
    특정 클래스(좌표) i와 j 사이의 유사도 측정을 위한 전용 데이터셋
    """
    def __init__(self, root_a, root_b, analysis_pairs, transform=None):
        """
        analysis_pairs: [(class_i, class_j), ...] 형태의 리스트
        class_i: (reg, x, y) 튜플
        """
        self.root_a = root_a
        self.root_b = root_b
        self.pairs = analysis_pairs
        self.transform = transform

    def __getitem__(self, index):
        class_i, class_j = self.pairs[index]
        
        # Class i: Anchor (VWorld)
        img_i_path = f"{self.root_a}/tile_{class_i[0]}_{class_i[1]}_{class_i[2]}.png"
        # Class j: Target (Google)
        img_j_path = f"{self.root_b}/tile_{class_j[0]}_{class_j[1]}_{class_j[2]}.png"
        
        img_i = Image.open(img_i_path).convert('L')
        img_j = Image.open(img_j_path).convert('L')

        if self.transform:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        # target 대신 두 클래스의 정보를 넘겨 나중에 분석 시 활용
        return (img_i, img_j), (str(class_i), str(class_j))

    def __len__(self):
        return len(self.pairs)
        

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
