
import os
from torch.utils.data.dataloader import DataLoader
import yaml
from models.resnet_base_network import SSRN_
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import sys
sys.path.append('~/PycharmProjects/SSTN-main')  # add the SSTN root path to environment path

import torch.backends.cudnn as cudnn
from operator import truediv
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import scipy.io as sio

import time
import random
import torch
from torch.utils import data
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from utils import cal_results

import argparse
import logging
from n2ssl_pretrain import zeroPadding_3D,same_seeds,generate_data_from_indices

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

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def get_features_from_encoder(encoder, loader, device):
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            x = x.to(device)
            feature_vector = encoder(x)
            feature_vector = F.avg_pool2d(feature_vector, feature_vector.size()[-1])
            feature_vector = torch.squeeze(feature_vector)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, drop_last=False)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, drop_last=False)
    return train_loader, test_loader


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(y_pred, y_test, name):
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'UP':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    else:
        target_names = ['Scrub', 'Willow_swamp', 'CP_hammock', 'Slash_pine', 'Oak', 'Hardwood', 'Swap',
                        'G_marsh', 'S_march', 'C_march', 'Salt_march', 'M_flats', 'Water']

    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser("UP")
    parser.add_argument('--dataset', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=20)
    parser.add_argument('--n_lbl', type=str, default='200')
    parser.add_argument('--n_gpu', type=int, default=1)
    args = parser.parse_args()
    n_lbl = args.n_lbl
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = config['datasets'][args.dataset]
    save_path = os.path.join('acc_linear_classification_K=20','{}_pretrain{}'.format(dataset, args.pretrain))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, 'N2SSL_Linear_classification_{}_{}.txt'.format(n_lbl, time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    IN_PATH = './datasets'
    if dataset == 'UP':
        mat_data = sio.loadmat(IN_PATH + '/UP/PaviaU.mat')
        data_IN = mat_data['paviaU']
        mat_gt = sio.loadmat(IN_PATH + '/UP/PaviaU_gt.mat')
        gt_IN = mat_gt['paviaU_gt']
        nb_classes = 9
        n_band = 103

    elif dataset == 'IP':
        mat_data = sio.loadmat(IN_PATH + '/IP/Indian_pines_corrected.mat')
        data_IN = mat_data['indian_pines_corrected']
        mat_gt = sio.loadmat(IN_PATH + '/IP/Indian_pines_gt.mat')
        gt_IN = mat_gt['indian_pines_gt']
        nb_classes = 16
        n_band = 200

    elif dataset == 'SA':
        mat_data = sio.loadmat(IN_PATH + '/SA/Salinas_corrected.mat')
        data_IN = mat_data['salinas_corrected']
        mat_gt = sio.loadmat(IN_PATH + '/SA/Salinas_gt.mat')
        gt_IN = mat_gt['salinas_gt']
        nb_classes = 16
        n_band = 204

    elif dataset == 'KSC':
        mat_data = sio.loadmat(IN_PATH + '/KSC/KSC.mat')
        data_IN = mat_data['KSC']
        mat_gt = sio.loadmat(IN_PATH + '/KSC/KSC_gt.mat')
        gt_IN = mat_gt['KSC_gt']
        nb_classes = 13
        n_band = 176

    patch_size = 4
    new_gt_IN = gt_IN
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )
    MAX = data_IN.max()
    data_IN = np.transpose(data_IN, (2, 0, 1))  # c*h*w
    data_IN = data_IN - np.mean(data_IN, axis=(1, 2), keepdims=True)
    data_IN = data_IN / MAX                   # normalize
    whole_data = data_IN.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2]) #c*h*w
    padded_data = zeroPadding_3D(whole_data, patch_size) #

    indices_  = sio.loadmat('./datasets/{}/Label_indices_{}.mat'.format(dataset, args.n_lbl))
    train_indices_eval = indices_['train_indices_eval']
    test_indices_eval = indices_['test_indices_eval']
    acc = []
    n_trial_eval = 20
    logging.info('################################################################################')
    logging.info('###########################  Linear classification  ############################')
    for i_trial_eval in range(n_trial_eval):
        logging.info('--------------------------------- Trial %d --------------------------------', i_trial_eval+1)
        train_data_eval, y_train_eval = generate_data_from_indices(gt,
                                                                     train_indices_eval[:, i_trial_eval],
                                                                     patch_size,
                                                                     n_band,
                                                                     whole_data,
                                                                     padded_data)

        test_data_eval, y_test_eval = generate_data_from_indices(gt,
                                                                     test_indices_eval[:, i_trial_eval],
                                                                     patch_size,
                                                                     n_band,
                                                                     whole_data,
                                                                     padded_data)

        train_data_eval = torch.from_numpy(np.asarray(train_data_eval, dtype=np.float32))
        y_train_eval = torch.from_numpy(y_train_eval.astype(np.longlong))

        test_data_eval = torch.from_numpy(np.asarray(test_data_eval, dtype=np.float32))
        y_test_eval = torch.from_numpy(y_test_eval.astype(np.longlong))

        training_set = HSIDataset(range(len(y_train_eval)), train_data_eval, y_train_eval)
        validation_set = HSIDataset(range(len(y_test_eval)), test_data_eval, y_test_eval)

        same_seeds(0)
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=25, shuffle=True, num_workers=0, drop_last=False)
        validationloader = torch.utils.data.DataLoader(validation_set, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

        device = 'cuda:{}'.format(args.n_gpu) if torch.cuda.is_available() else 'cpu'
        print(f"Training with: {device}")

        ####################################################################################################################
        # -------------------------------------------- network definition -------------------------------------------------#
        cls_network = SSRN_(name=dataset)
        output_feature_dim = cls_network.projection.net[0].in_features
        n_load = 0
        load_path = os.path.join('weights_BYOL_linear_classification_K=20', '{}_pretrain{}'.format(dataset, args.pretrain))
        load_params = torch.load(os.path.join(load_path, 'model_n2ssl.pth'),
                                 map_location=torch.device(torch.device(device)))

        encoder = torch.nn.Sequential(*list(cls_network.children())[:-1])
        encoder_dict = encoder.state_dict()
        from collections import OrderedDict
        pretrained_dict = OrderedDict()
        kk = []
        for k in load_params['online_network_state_dict']:
            kk.append(k)
        kk2 = []
        for k in encoder_dict:
            kk2.append(k)
        for k_i in range(len(kk2)):
            encoder_dict[kk2[k_i]] = load_params['online_network_state_dict'][kk[k_i]]
        encoder.load_state_dict(encoder_dict)
        print("Parameters successfully loaded.")

        encoder = encoder.to(device)
        encoder.eval()
        same_seeds(0)
        logreg = LogisticRegression(output_feature_dim, nb_classes)
        logreg = logreg.to(device)

        x_train, y_train = get_features_from_encoder(encoder, trainloader, device)
        x_test, y_test = get_features_from_encoder(encoder, validationloader, device)

        if len(x_train.shape) > 2:
            x_train = torch.mean(x_train, dim=[2, 3])
            x_test = torch.mean(x_test, dim=[2, 3])

        print("Training data shape:", x_train.shape, y_train.shape)
        print("Testing data shape:", x_test.shape, y_test.shape)

        train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(logreg.parameters(), lr=2.0e-3)
        eval_every_n_epochs = 100
        best_pred = 0.0
        best_acc = 0.0
        for epoch in range(4000):
            #     train_acc = []
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                def tclip(x, alpha):
                    return alpha * torch.tanh(x / alpha)

                logits = logreg(x)
                predictions = torch.argmax(logits, dim=1)

                loss = criterion(tclip(logits, 20.0), y) + 1.0e-2 * torch.mean(torch.sqrt(tclip(x, 20.0)))

                loss.backward()
                optimizer.step()

            total = 0
            if epoch % eval_every_n_epochs == 0:
                correct = 0
                label_val = []
                pred_val = []

                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    logits = logreg(x)
                    predictions = torch.argmax(logits, dim=1)

                    total += y.size(0)
                    correct += (predictions == y).sum().item()

                    label_val.append(y)
                    pred_val.append(predictions)

                label_val_cpu = [x.cpu() for x in label_val]
                pred_val_cpu = [x.cpu() for x in pred_val]

                label_cat = np.concatenate(label_val_cpu)
                pred_cat = np.concatenate(pred_val_cpu)

                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc
                    label_best = label_cat
                    pred_best = pred_cat

                print(f"Testing accuracy: {np.mean(acc)}")

        matrix = metrics.confusion_matrix(label_best, pred_best)

        OA, AA_mean, Kappa, AA = cal_results(matrix)

        logging.info('OA, AA_Mean, Kappa: %f, %f, %f, ', 100*OA, AA_mean, Kappa)
        logging.info(str(("AA for each class: ", AA)))

        classification, confusion, Test_accuracy, each_acc, aa, kappa = reports(pred_best, label_best, dataset)

        classification = str(classification)
        confusion = str(confusion)

        file_name = os.path.join(save_path, "N2SSL_{}_Linear_classification_report_{}.txt".format(i_trial_eval+1, args.n_lbl))
        with open(file_name, 'w') as x_file:
            x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(kappa))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(Test_accuracy))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(aa))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))


