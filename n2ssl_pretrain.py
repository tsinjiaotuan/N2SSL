import yaml
from data.transforms import get_byol_HSI_single_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import SSRN_
from trainer_N2SSL_byol import BYOLTrainer
import sys
import torch.backends.cudnn as cudnn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import scipy.io as sio
import time
import random
import torch
from torch.utils import data
import math
import logging
import argparse

def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:, range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, :, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch

def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)),
                            'constant', constant_values=0)
    return new_matrix

class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels, transforms=None):
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        if self.transforms is not None:
            X = self.transforms(X)
        y = self.labels[ID]

        return X, y

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generate_data_from_indices(gt, indices, patch_size, n_band, whole_data, padded_data):
    y_train_byol_1 = gt[indices] - 1
    train_data_byol_1 = np.zeros((len(indices), n_band, 2 * patch_size + 1, 2 * patch_size + 1))
    train_byol_1_assign = indexToAssignment(indices, patch_size, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(indices)):
        train_data_byol_1[i] = selectNeighboringPatch(padded_data, patch_size, train_byol_1_assign[i][0],
                                                      train_byol_1_assign[i][1])

    return train_data_byol_1, y_train_byol_1


def main(pretrain, n_gpu, dataset_name, train_byol_data, train_byol_lbl, train_similar_data, train_similar_lbl, train_dataset_eval, test_dataset_eval, nb_classes):
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    ####################################################################################################################
    #-------------------------------------------- setting, same_seeds --------------------------------------------------
    same_seeds(0)
    device = 'cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    torch.cuda.set_device(config['trainer']['gpu'])
    input_shape = config['data_transforms']['input_shape']
    transform_probs1 = np.zeros(4)
    transform_probs2 = np.zeros(4)
    i_augmentation = 5     # cropping+sub-horizontal flip VS  cropping+sub-vertical flip, refer to the analysis of data augmentation in the manuscript.
    if i_augmentation > 0:
        i_1 = int((i_augmentation - 1) / 4)
        i_2 = (i_augmentation - 1) % 4
        transform_probs1[i_1] = 1
        transform_probs2[i_2] = 1
    target_transform, online_transform = get_byol_HSI_single_transforms(input_shape=input_shape,
                                                                        probs1=transform_probs1,
                                                                        probs2=transform_probs2)

    save_path = os.path.join('weights_BYOL_linear_classification_K=20', '{}_pretrain{}'.format(dataset_name, pretrain))
    ####################################################################################################################
    # -------------------------------------------- network definition -------------------------------------------------#
    # online_encoder_projection
    online_network_1 = SSRN_(name=dataset_name)
    predictor_1 = MLPHead(in_channels=online_network_1.projection.net[-1].out_features,
                        **config['network']['projection_head'])

    # target_encoder_projection
    target_network_1 = SSRN_(name=dataset_name)

    #######################################################################################
    # if config['trainer']['n_gpu'] == 1:                                                 #
    #     if torch.cuda.device_count() > 1:                                               #
    #         print("let's use", torch.cuda.device_count(), "GPUs!")                      #
    #         target_network_1 = torch.nn.DataParallel(target_network_1)                  #
    #         predictor_1 = torch.nn.DataParallel(predictor_1)                            #
    #         online_network_1 = torch.nn.DataParallel(online_network_1)                  #
    #######################################################################################
    target_network_1 = target_network_1.to(device)
    online_network_1 = online_network_1.to(device)
    predictor_1 = predictor_1.to(device)

    ####################################################################################################################
    #-------------------------------------------------- optimizer setting ---------------------------------------------#
    warm_up_iter = 10
    r1 = 0.50
    max_epoch = config['trainer']['max_epochs']
    if dataset_name == 'IP' or 'KSC':
        max_epoch = 200
    lr_max = config['optimizer']['params']['lr']
    lr_min = 1.0e-3
    lr_lambda = lambda epoch_iter: (epoch_iter*(1-r1) + warm_up_iter*r1-1)/(warm_up_iter-1) if epoch_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((epoch_iter - warm_up_iter)/(max_epoch - warm_up_iter) * math.pi)))/lr_max

    optimizer_1 = torch.optim.SGD(list(online_network_1.parameters()) + list(predictor_1.parameters()),
                                **config['optimizer']['params'])
    scheduler_1 = torch.optim.lr_scheduler.LambdaLR(optimizer_1, lr_lambda=[lr_lambda])

    ####################################################################################################################
    #------------------------------------------------ training ---------------------------------------------------------
    trainer = BYOLTrainer(online_network=online_network_1,
                          target_network=target_network_1,
                          optimizer=optimizer_1,
                          scheduler=scheduler_1,
                          predictor=predictor_1,
                          device=device,
                          save_path=save_path,
                          **config['trainer'])

    acc = trainer.train(train_data_byol=train_byol_data,
                        train_lbl_byol=train_byol_lbl,
                        train_similar_data=train_similar_data,
                        train_similar_lbl =train_similar_lbl,
                        data_transform_1=target_transform,
                        data_transform_2 = online_transform,
                        train_dataset_eval=train_dataset_eval,
                        test_dataset_eval=test_dataset_eval,
                        batch_size_eval=128,
                        nb_classes=nb_classes,
                        dataset_name=dataset_name)
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser("UP")
    parser.add_argument('--dataset', type=int, default=1) #0:UP 1:IP 2:KSC
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=20)
    args = parser.parse_args()
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = config['datasets'][args.dataset]
    IN_PATH = './datasets'
    pretrain = args.pretrain  # precentage of samples for pretraining
    n_band = 100
    if dataset == 'UP':
        mat_data = sio.loadmat(IN_PATH + '/UP/PaviaU.mat')
        data_IN = mat_data['paviaU']
        mat_gt = sio.loadmat(IN_PATH + '/UP/PaviaU_gt.mat')
        gt_IN = mat_gt['paviaU_gt']
        nb_classes = 9
        n_band = 103
        pretrain = 20
    elif dataset == 'IP':
        mat_data = sio.loadmat(IN_PATH + '/IP/Indian_pines_corrected.mat')
        data_IN = mat_data['indian_pines_corrected']
        mat_gt = sio.loadmat(IN_PATH + '/IP/Indian_pines_gt.mat')
        gt_IN = mat_gt['indian_pines_gt']
        nb_classes = 16
        n_band = 200
        pretrain = 20
    else:
        mat_data = sio.loadmat(IN_PATH + '/KSC/KSC.mat')
        data_IN = mat_data['KSC']
        mat_gt = sio.loadmat(IN_PATH + '/KSC/KSC_gt.mat')
        gt_IN = mat_gt['KSC_gt']
        nb_classes = 13
        n_band = 176
        pretrain = 40 # 60 when train N2SSL for KSC->IP/UP finetune

    import os
    load_path = os.path.join('weights_BYOL_linear_classification_K=20', '{}_pretrain{}'.format(dataset, pretrain))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(load_path, 'N2SSL_pretrain{}_log_{}.txt'.format(pretrain, time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    patch_size = 4
    new_gt_IN = gt_IN
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )
    MAX = data_IN.max()
    data_IN = np.transpose(data_IN, (2, 0, 1))  # c*h*w
    data_IN = data_IN - np.mean(data_IN, axis=(1, 2), keepdims=True)
    data_IN = data_IN / MAX                   # normalize
    whole_data = data_IN.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2]) #c*h*w
    padded_data = zeroPadding_3D(whole_data, patch_size) #

    indices_  = sio.loadmat('./datasets/{}/Label_indices_200.mat'.format(dataset)) # 200 labeled samples for linear classification
    train_indices_eval = indices_['train_indices_eval']  # indexes for classification
    test_indices_eval = indices_['test_indices_eval']

    acc = []
    logging.info('################################################################################')
    logging.info('##################################  N2SSL Training  ############################')
    pretrain_indices_ = sio.loadmat('./datasets/{}/pretrain_indices_{}.mat'.format(dataset, pretrain))

    byol_indices_ = pretrain_indices_['indices_byol']     # indexes of samples for pretraining, i.e. N*1
    similar_indices_ = pretrain_indices_['indices_similar']       # precomputed similar indexs, i.e. 5N*4 with K=20=5*4
    logging.info('##################################  Prepare Data  ##############################')
    train_similar_data = {}
    train_similar_lbl = {}
    for j in range(similar_indices_.shape[1]):
        train_similar_data_j, train_similar_lbl_j = generate_data_from_indices(gt,
                                                                               similar_indices_[:, j]-1,
                                                                               patch_size,
                                                                               n_band,
                                                                               whole_data,
                                                                               padded_data)
        train_similar_data[j] = torch.from_numpy(np.asarray(train_similar_data_j, dtype=np.float32))  # 5N*1
        train_similar_lbl[j] = torch.from_numpy(train_similar_lbl_j.astype(np.longlong))

    train_data_eval, y_train_eval = generate_data_from_indices(gt,
                                                                 train_indices_eval[:, 0],
                                                                 patch_size,
                                                                 n_band,
                                                                 whole_data,
                                                                 padded_data)
    train_data_eval = torch.from_numpy(np.asarray(train_data_eval, dtype=np.float32))
    y_train_eval = torch.from_numpy(y_train_eval.astype(np.longlong))

    test_data_eval, y_test_eval = generate_data_from_indices(gt,
                                                                 test_indices_eval[:, 0],
                                                                 patch_size,
                                                                 n_band,
                                                                 whole_data,
                                                                 padded_data)

    test_data_eval = torch.from_numpy(np.asarray(test_data_eval, dtype=np.float32))
    y_test_eval = torch.from_numpy(y_test_eval.astype(np.longlong))

    train_dataset_eval = HSIDataset(range(len(train_indices_eval)), train_data_eval, y_train_eval)
    test_dataset_eval = HSIDataset(range(len(test_indices_eval)), test_data_eval, y_test_eval)

    train_byol_data, train_byol_lbl = generate_data_from_indices(gt,
                                                                 byol_indices_[:, 0]-1,
                                                                 patch_size,
                                                                 n_band,
                                                                 whole_data,
                                                                 padded_data)
    train_data_byol = torch.from_numpy(np.asarray(train_byol_data, dtype=np.float32))
    y_train_byol = torch.from_numpy(train_byol_lbl)
    logging.info('##################################  Prepare Data OK   ##########################')
    acc_i = main(pretrain, args.n_gpu, dataset, train_data_byol, y_train_byol, train_similar_data, train_similar_lbl, train_dataset_eval, test_dataset_eval, nb_classes)
    acc.append(acc_i)
    time.sleep(1)
    logging.info('  ')
    logging.info('  ')
    logging.info('  ')

