import logging
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from utils.config import FLAGS
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution

def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_cifar10_data(datadir):
    seed = FLAGS.random_seed
    train_transform, test_transform = _data_transforms_cifar10()
    torch.random.manual_seed(seed)
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    torch.random.manual_seed(seed)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)
  

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    seed = FLAGS.random_seed
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        np.random.seed(seed)
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.seed(seed)
                np.random.shuffle(idx_k)
                np.random.seed(seed)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.seed(seed)
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)

def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
# import numpy as np
# from torchvision import datasets, transforms

# def get_dataset_cifar10_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
#     data_dir = '../data/cifar/'
#     apply_transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
#                                    transform=apply_transform)

#     test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
#                                       transform=apply_transform)

#     # Chose euqal splits for every user
#     user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
#     return train_dataset, test_dataset, user_groups_train, user_groups_test

# def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
#     num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
#     num_classes = 10
#     num_imgs_perc_test, num_imgs_test_total = 1000, 10000
#     assert(n_class * num_users <= num_shards_train)
#     assert(n_class <= num_classes)
#     idx_class = [i for i in range(num_classes)]
#     idx_shard = [i for i in range(num_shards_train)]
#     dict_users_train = {i: np.array([]) for i in range(num_users)}
#     dict_users_test = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards_train*num_imgs_train)
#     # labels = dataset.train_labels.numpy()
#     labels = np.array(train_dataset.targets)
#     idxs_test = np.arange(num_imgs_test_total)
#     labels_test = np.array(test_dataset.targets)
#     #labels_test_raw = np.array(test_dataset.targets)

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#     labels = idxs_labels[1, :]

#     idxs_labels_test = np.vstack((idxs_test, labels_test))
#     idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
#     idxs_test = idxs_labels_test[0, :]

#     for i in range(num_users):
#         user_labels = np.array([])
#         rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         unbalance_flag = 0
#         for rand in rand_set:
#             if unbalance_flag == 0:
#                 dict_users_train[i] = np.concatenate(
#                     (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
#                 user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
#             else:
#                 dict_users_train[i] = np.concatenate(
#                     (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
#                 user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
#             unbalance_flag = 1
#         user_labels_set = set(user_labels)
#         #print(user_labels_set)
#         #print(user_labels)
#         for label in user_labels_set:
#             dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
#         #print(set(labels_test_raw[dict_users_test[i].astype(int)]))

#     return dict_users_train, dict_users_test

