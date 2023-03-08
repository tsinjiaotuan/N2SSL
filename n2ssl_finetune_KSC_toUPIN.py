import os
from torch.utils.data.dataloader import DataLoader
import yaml
from models.resnet_base_network import SSRN_FN
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import sys
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import scipy.io as sio

import time

import torch
from utils import cal_results
from n2ssl_pretrain import zeroPadding_3D,same_seeds,generate_data_from_indices
from n2ssl_linear_classification import HSIDataset, AA_andEachClassAccuracy, reports
import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser("UP")
    parser.add_argument('--n_lbl', type=str, default='0.05')
    parser.add_argument('--dataset', type=int, default=0) #0UP 1IP

    args = parser.parse_args()
    n_lbl = args.n_lbl
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = config['datasets'][args.dataset]

    save_path = os.path.join('acc_finetuned_fromKSC_classification_pretrain40', dataset)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, '{}_labelled{}_fintune_from_KSC_pretrain_{}.txt'.format(dataset,n_lbl,
                                                                               time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    IN_PATH = './datasets'
    if dataset == 'UP':
        mat_data = sio.loadmat(IN_PATH + '/UP/PaviaU_KSC.mat')
        data_IN = mat_data['paviaU']
        mat_gt = sio.loadmat(IN_PATH + '/UP/PaviaU_gt.mat')
        gt_IN = mat_gt['paviaU_gt']
        nb_classes = 9
        n_band = 176

    elif dataset == 'IP':
        mat_data = sio.loadmat(IN_PATH + '/IP/Indian_pines_corrected_KSC.mat')
        data_IN = mat_data['indian_pines_corrected']
        mat_gt = sio.loadmat(IN_PATH + '/IP/Indian_pines_gt.mat')
        gt_IN = mat_gt['indian_pines_gt']
        nb_classes = 16
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

    indices_  = sio.loadmat('./datasets/{}/Label_indices_{}.mat'.format(dataset, n_lbl))
    train_indices_eval = indices_['train_indices_eval']
    test_indices_eval = indices_['test_indices_eval']
    acc = []
    n_trial = 20

    logging.info('################################################################################')
    logging.info('###########################  Classification with FN #####################')
    for i_trial in range(n_trial):
        logging.info('--------------------------------- Trial %d --------------------------------', i_trial + 1)
        train_data_eval, y_train_eval = generate_data_from_indices(gt,
                                                                     train_indices_eval[:, i_trial],
                                                                     patch_size,
                                                                     n_band,
                                                                     whole_data,
                                                                     padded_data)

        test_data_eval, y_test_eval = generate_data_from_indices(gt,
                                                                     test_indices_eval[:, i_trial],
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

        if len(y_train_eval)<50*4:
            bts = int(len(y_train_eval)/4)
        else:
            bts = 50
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=bts, shuffle=True, num_workers=1)
        validationloader = torch.utils.data.DataLoader(validation_set, batch_size=50, shuffle=False, num_workers=1)

        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        print(f"Training with: {device}")
        torch.cuda.set_device(config['trainer']['gpu'])
        same_seeds(0)
        ####################################################################################################################
        # -------------------------------------------- network definition -------------------------------------------------#

        cls_network = SSRN_FN(name='KSC_{}'.format(dataset))
        load_path = os.path.join('weights_BYOL_linear_classification_K=20', 'KSC_pretrain60')
        load_params = torch.load(os.path.join(load_path, 'model_n2ssl_backup.pth'),
                                 map_location=torch.device(torch.device(device)))
        encoder_proj_dict = cls_network.state_dict()
        from collections import OrderedDict
        pretrained_dict = OrderedDict()
        kk = []
        for k in load_params['online_network_state_dict']:
            kk.append(k)
        kk2 = []
        for k in encoder_proj_dict:
            kk2.append(k)
        for k_i in range(len(kk2)-2):
            encoder_proj_dict[kk2[k_i]] = load_params['online_network_state_dict'][kk[k_i]]
        cls_network.load_state_dict(encoder_proj_dict)
        print("Parameters successfully loaded.")

        cls_network = cls_network.to(device)
        cls_network.train()

        criterion = nn.CrossEntropyLoss()

        linear = list(map(id, cls_network.linear.parameters()))
        base_params = filter(lambda p: id(p) not in linear, cls_network.parameters())
        optimizer = optim.Adam([{'params': base_params, 'lr':2.0e-4}, {'params': cls_network.linear.parameters(), 'lr': 2.0e-3}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10)
        best_pred = 0.0
        NN = 200
        for epoch in range(NN):  # loop over the dataset multiple times
            running_loss = 0.0
            t_loss = 0.0
            cls_network = cls_network.train()
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = cls_network(inputs.float())
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                t_loss += loss.item()
                if i % 4 == 3:  # print every 2000 mini-batches
                    logging.info('[%d, %2d] loss: %.4f' %
                                 (epoch + 1, i + 1, running_loss / 4))
                    running_loss = 0.0
            scheduler.step(t_loss)

            if epoch>NN-20:
                correct = 0
                total = 0
                net = cls_network.eval()
                counter = 0
                label_val = []
                pred_val = []
                with torch.no_grad():

                    for data in validationloader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images.float())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.long()).sum().item()
                        label_val.append(labels.long())
                        pred_val.append(predicted)

                label_val_cpu = [x.cpu() for x in label_val]
                pred_val_cpu = [x.cpu() for x in pred_val]
                label_cat = np.concatenate(label_val_cpu)
                pred_cat = np.concatenate(pred_val_cpu)

                new_pred = correct / total
                logging.info('Accuracy of the network on the validation set: %.5f %%' % (
                        100 * new_pred))

                if new_pred > best_pred:
                    logging.info('new_pred %f', new_pred)
                    logging.info('best_pred %f', best_pred)

                    best_pred = new_pred
                    label_best = label_cat
                    pred_best = pred_cat

        logging.info('Finished Training')

        # Validation Stages
        matrix = metrics.confusion_matrix(label_best, pred_best)
        OA, AA_mean, Kappa, AA = cal_results(matrix)

        logging.info('OA, AA_Mean, Kappa: %f, %f, %f, ', OA, AA_mean, Kappa)
        logging.info(str(("AA for each class: ", AA)))

        classification, confusion, Test_accuracy, each_acc, aa, kappa = reports(pred_best, label_best, dataset)

        classification = str(classification)
        confusion = str(confusion)
        file_name = os.path.join(save_path, "{}_{}_FN_from_KSC_classification_report_{}.txt".format(dataset, n_lbl, i_trial))

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

