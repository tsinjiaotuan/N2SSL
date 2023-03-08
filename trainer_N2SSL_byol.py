import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import logging
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from sklearn import metrics
from utils import cal_results

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
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, samples_1, samples_2, labels, transforms_1=None, transforms_2=None):
        self.list_IDs = list_IDs
        self.samples_1 = samples_1
        self.samples_2 = samples_2
        self.labels = labels
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X1 = self.samples_1[ID]
        X2 = self.samples_2[ID]
        if self.transforms_1 is not None:
            X1 = self.transforms_1(X1)
            X2 = self.transforms_2(X2)
        y = self.labels[ID]

        return X1, X2, y

class HSIDataset_2(torch.utils.data.Dataset):
    def __init__(self, list_IDs, samples_1, samples_2, labels,  transforms_1=None, transforms_2=None):
        self.list_IDs = list_IDs
        self.samples_1 = samples_1
        self.samples_2 = samples_2
        self.labels = labels
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X1 = self.samples_1[ID]
        X2 = self.samples_2[ID]
        if self.transforms_1 is not None:
            X3 = self.transforms_1(X1)
            X4 = self.transforms_2(X2)
        else:
            X3 = X1
            X4 = X2
        y = self.labels[ID]

        return X3, X4, X1, X2, y

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, scheduler, device, save_path, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.save_path = save_path
        self.n_gpu = params['n_gpu']
        # utils.create_exp_dir(self.save_path, scripts_to_save=glob.glob('trainer_evolution_byol.py'))
        # shutil.copyfile('./data/HSI_transforms.py', os.path.join(self.save_path,'scripts','HSI_transforms.py'))
        # shutil.copyfile('./data/transforms.py', os.path.join(self.save_path, 'scripts', 'transforms.py'))
        # shutil.copyfile('./config/config.yaml', os.path.join(self.save_path, 'scripts', 'config.yaml'))
        # shutil.copyfile('./models/resnet_base_network.py', os.path.join(self.save_path, 'scripts', 'resnet_base_network.py'))

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient



    def train(self, train_data_byol, train_lbl_byol, train_similar_data, train_similar_lbl,
              data_transform_1, data_transform_2, train_dataset_eval, test_dataset_eval, batch_size_eval,  nb_classes,
              dataset_name):

        train_dataset_byol = HSIDataset(range(len(train_data_byol)),
                                          samples_1=train_data_byol,
                                          samples_2=train_data_byol,
                                          labels=train_lbl_byol,
                                          transforms_1=data_transform_1,
                                          transforms_2=data_transform_2)
        num_epoch_byol = 40  # number of epoch 1 in the manuscript, i.e. n_{e1}
        if dataset_name == 'IP' or dataset_name == 'KSC':
            self.batch_size = 32
            self.max_epochs = 200
            num_epoch_byol = 80

        byol_train_loader = DataLoader(train_dataset_byol, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        self.initializes_target_network()
        loss_min = 100.0
        num_epoch_save = int(0.25*num_epoch_byol)
        logging.info('##################################  Stage 1 BYOL Training Begins ###############')
        for epoch_counter in range(num_epoch_byol):
            running_loss = 0.0
            batch_cnt = 0

            for batch_view_1, batch_view_2, _ in byol_train_loader:
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                loss = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                # print statistics
                running_loss += loss.item()
                batch_cnt += 1
                if batch_cnt % 16 == 15:  #
                    logging.info('Stage 1 (Byol Training) [%d, %d] loss:     %.4f' %
                             (epoch_counter + 1, batch_cnt + 1, running_loss / 16))

                    if loss_min > running_loss/16:
                        loss_min = running_loss/16
                        logging.info('Stage 1 (Byol Training) [%d, %d] min_loss: %.4f'%(epoch_counter + 1, batch_cnt + 1, loss_min))
                        if epoch_counter > num_epoch_save:
                            self.save_model(os.path.join(self.save_path, 'model_ssl.pth'))
                    running_loss = 0.0
            self.scheduler.step()

        logging.info('##################################  Stage 1 BYOL Training Over  ##############################')

        train_data_byol_5seq = train_similar_data[0]       #5N*1: repeat train_data_byol 5 times in the axis of column
        train_lbl_byol_5seq = train_similar_lbl[0]
        n_train_per_j = 2  # number of epoch 2 in the manuscript, i.e. n_{e2}
        if dataset_name == 'IP':
            n_train_per_j = 4
        # training_set_online/target_updated: training dataset for extraction of RN2P, including both initial training
        # dataset, i.e. train_data_byol, and extracted RN2P.
        training_set_online_updated = train_data_byol
        training_set_target_updated = train_data_byol

        ratio_added = 0.05  #K=20=5*4 columns, thus add 25% N2P with least loss to training_set_online/target_updated
        lbl_online_added_total = []
        lbl_target_added_total = []
        lbl_online_added_count_total = np.zeros(nb_classes)
        lbl_target_added_count_total = np.zeros(nb_classes)
        logging.info('##################################  Stage 2 Extraction of RN2P  ##############################')
        for j in range(4):
            if j>0:
                training_set_online = training_set_online_updated
                training_set_target = training_set_target_updated
                # initial training dataset and extracted RN2P
                training_set_j = HSIDataset(range(len(training_set_online)),
                                                  samples_1=training_set_online,
                                                  samples_2=training_set_target,
                                                  labels=torch.ones(len(training_set_online)),
                                                  transforms_1=data_transform_1,
                                                  transforms_2=data_transform_2)
                trainingset_j_loader = DataLoader(training_set_j, batch_size=self.batch_size,
                                                 num_workers=self.num_workers, drop_last=True, shuffle=True)
                for k in range(n_train_per_j):
                    batch_cnt = 0
                    running_loss = 0.0
                    for batch_view_1, batch_view_2, batch_lbl in trainingset_j_loader:
                        batch_view_1 = batch_view_1.to(self.device)
                        batch_view_2 = batch_view_2.to(self.device)

                        loss = self.update(batch_view_1, batch_view_2)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self._update_target_network_parameters()  # update the key encoder
                        batch_cnt += 1
                        running_loss += loss.item()
                        if batch_cnt % 16 == 15:
                            logging.info('Stage 2 (Extraction of RN2P) [%dth column, %dth epoch, %dth batch], loss: %.4f' %
                                         (j, k, batch_cnt + 1, running_loss / 16))
                            running_loss = 0.0

            idx_j_valid = range(len(train_lbl_byol_5seq))
            n_j_valid = len(idx_j_valid)
            n_train = n_j_valid // len(train_data_byol)  #equals to 5
            train_data_byol_j = train_data_byol_5seq[idx_j_valid]
            train_lbl_byol_j = train_lbl_byol_5seq[idx_j_valid]
            train_data_similar_j = train_similar_data[j+1][idx_j_valid]
            train_lbl_similar_j = train_similar_lbl[j+1][idx_j_valid]

            for i in range(n_train): # 0,1,2,3,4, when j=1, i means K ranging from 1NN to 5NN, j=2, i means K ranging from 6NN to 10NN
                train_data_similar_ji = train_data_similar_j[i*len(train_data_byol):min(((i+1)*len(train_data_byol), len(idx_j_valid)))]
                train_data_byol_ji = train_data_byol_j[i*len(train_data_byol):min(((i+1)*len(train_data_byol), len(idx_j_valid)))]
                train_lbl_similar_ji = train_lbl_similar_j[i*len(train_data_byol):min(((i + 1) * len(train_data_byol), len(idx_j_valid)))]
                train_lbl_byol_ji = train_lbl_byol_j[i*len(train_data_byol):min(((i+1)*len(train_data_byol), len(idx_j_valid)))]

                # compute loss between byol data and similar data
                loss_byol_similar, train_lbl_byol_ji_stack, train_data_similar_ji_stack, train_lbl_ji_stack = self.forward(data1=train_data_byol_ji,
                                                                                                                           data2=train_data_similar_ji,
                                                                                                                           lbl1=train_lbl_byol_ji,
                                                                                                                           lbl2=train_lbl_similar_ji,
                                                                                                                           data_transform_1=data_transform_1,
                                                                                                                           data_transform_2=data_transform_2)
                idx_sorted_based_loss = np.argsort(loss_byol_similar.cpu())
                n_selected = int(ratio_added * len(idx_sorted_based_loss))

                n_selected = min((n_selected, len(train_data_byol)))
                loss_median = torch.median(loss_byol_similar.cpu())
                idx_tmp =  np.argsort(torch.abs(loss_byol_similar.cpu()-loss_median))  # the index close to the median loss
                n_begin = np.argwhere(idx_sorted_based_loss == idx_tmp[0]) # begin from the median loss to select n_selected samples to ensure diversity
                idx_added = idx_sorted_based_loss[n_begin:n_begin+n_selected]
                lbl_added = train_lbl_ji_stack[idx_added]

                lbl_online_added_total.append(lbl_added[:,0])
                lbl_target_added_total.append(lbl_added[:,1])
                similar_total = 0
                n_total = 0
                for m in range(len(lbl_online_added_total)):
                    similar_total += torch.sum(lbl_online_added_total[m] == lbl_target_added_total[m])
                    n_total += len(lbl_online_added_total[m])
                acc_added_total = similar_total / n_total
                acc_ji = torch.sum(lbl_added[:,0]== lbl_added[:,1])/len(idx_added)
                import collections
                lbl_online_added_count = collections.Counter(np.array(lbl_added[:,0].cpu()))
                lbl_target_added_count = collections.Counter(np.array(lbl_added[:,1].cpu()))
                training_set_online_updated = torch.cat((training_set_online_updated, train_lbl_byol_ji_stack[idx_added]))
                training_set_target_updated = torch.cat((training_set_target_updated, train_data_similar_ji_stack[idx_added]))
                logging.info('Stage 2 (Extraction of RN2P) Add %d RN2P: '%(n_selected))
                logging.info('Stage 2 (Extraction of RN2P) Accuracy of RN2P added: %f'%(acc_ji.cpu()))
                logging.info('Stage 2 (Extraction of RN2P) GroTruth of RN2P added:')
                for p in range(nb_classes):
                    lbl_online_added_count_total[p] += lbl_online_added_count[p]
                    lbl_target_added_count_total[p] += lbl_target_added_count[p]
                    logging.info('Stage 2 (Extraction of RN2P) ------ Class %d: %d vs %d------'%(p, lbl_online_added_count[p], lbl_target_added_count[p]))
                logging.info('Stage 2 (Extraction of RN2P) Accuracy of all RN2P: %f'%(acc_added_total))
                logging.info('Stage 2 (Extraction of RN2P) GroTruth of all RN2P:')
                for z in range(nb_classes):
                    logging.info('Stage 2 (Extraction of RN2P) ------ Class %d: %d vs %d------'%(z, lbl_online_added_count_total[z], lbl_target_added_count_total[z]))

        # ################################################################################################################
        logging.info('##################################  Stage 3 Training based on RN2P  ##############################')
        training_set_online_updated = training_set_online_updated[len(train_data_byol):]
        training_set_target_updated = training_set_target_updated[len(train_data_byol):]
        train_dataset_byol = HSIDataset(range(len(training_set_online_updated)),
                                          samples_1=training_set_online_updated,
                                          samples_2=training_set_target_updated,
                                          labels=torch.ones(len(training_set_online_updated)),
                                          transforms_1=data_transform_1,
                                          transforms_2=data_transform_2)

        byol_train_loader = DataLoader(train_dataset_byol, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)
        loss_min = 100.0
        n_epoch_remain = self.max_epochs - num_epoch_byol - n_train_per_j * 15 # number of epoch 3 in the manuscript, i.e. n_{e3}
        num_epoch_save = int(0.0*(n_epoch_remain))
        for epoch_counter in range(n_epoch_remain):
            running_loss = 0.0
            batch_cnt = 0

            for batch_view_1, batch_view_2, _ in byol_train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                loss = self.update(batch_view_1, batch_view_2)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                # print statistics
                running_loss += loss.item()
                batch_cnt += 1
                if batch_cnt % 16 == 15:  # print every 2000 mini-batches
                    logging.info('Stage 3 (Training based on RN2P) [%d, %d] loss:     %.4f' %
                             (epoch_counter + 1, batch_cnt + 1, running_loss / 16))

                    if loss_min > running_loss/16:
                        loss_min = running_loss/16
                        logging.info('Stage 3 (Training based on RN2P) [%d, %d] min_loss: %.4f'%(epoch_counter + 1, batch_cnt + 1, loss_min))
                        if epoch_counter >= num_epoch_save:
                            self.save_model(os.path.join(self.save_path, 'model_n2ssl.pth'))
                    running_loss = 0.0

            self.scheduler.step()
            if epoch_counter == n_epoch_remain - 1:
                self.save_model(os.path.join(self.save_path, 'model_n2ssl_final.pth'))

        load_params = torch.load(os.path.join(self.save_path, 'model_n2ssl.pth'),
                                 map_location=torch.device(self.device))
        best_acc_evolution = self.linear_eval(load_params, train_dataset_eval, test_dataset_eval, batch_size_eval, nb_classes, dataset_name)
        logging.info('##################################  Stage 4 Linear Classification  ##############################')
        logging.info('Linear Classification Acc: %.4f' % (best_acc_evolution))
        return best_acc_evolution

    def linear_eval(self, load_params, train_dataset_eval,test_dataset_eval, batch_size_eval, nb_classes, dataset_name):
        if self.n_gpu == 1:
            if isinstance(self.online_network, torch.nn.DataParallel):
                if 'online_network_state_dict' in load_params:
                    self.online_network.module.load_state_dict(load_params['online_network_state_dict'])
                    print("Parameters successfully loaded.")
                encoder = torch.nn.Sequential(*list(self.online_network.module.children())[:-1])
                output_feature_dim = self.online_network.module.projection.net[0].in_features
            else:
                if 'online_network_state_dict' in load_params:
                    self.online_network.load_state_dict(load_params['online_network_state_dict'])
                    print("Parameters successfully loaded.")
                encoder = torch.nn.Sequential(*list(self.online_network.children())[:-1])
                output_feature_dim = self.online_network.projection.net[0].in_features
        else:
            if 'online_network_state_dict' in load_params:
                self.online_network.load_state_dict(load_params['online_network_state_dict'])
                print("Parameters successfully loaded.")
            encoder = torch.nn.Sequential(*list(self.online_network.children())[:-1])
            output_feature_dim = self.online_network.projection.net[0].in_features

        encoder = encoder.to(self.device)
        logreg = LogisticRegression(output_feature_dim, nb_classes)
        logreg = logreg.to(self.device)
        encoder.eval()

        train_loader = DataLoader(train_dataset_eval, batch_size=batch_size_eval,
                                  num_workers=0, drop_last=False, shuffle=True)

        test_loader = DataLoader(test_dataset_eval, batch_size=batch_size_eval,
                                 num_workers=0, drop_last=False, shuffle=True)

        x_train, y_train = get_features_from_encoder(encoder, train_loader, self.device)
        x_test, y_test = get_features_from_encoder(encoder, test_loader, self.device)

        if len(x_train.shape) > 2:
            x_train = torch.mean(x_train, dim=[2, 3])
            x_test = torch.mean(x_test, dim=[2, 3])

        print("Training data shape:", x_train.shape, y_train.shape)
        print("Testing data shape:", x_test.shape, y_test.shape)

        train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)
        optimizer = torch.optim.Adam(logreg.parameters(), lr=2.0e-3)
        criterion = torch.nn.CrossEntropyLoss()
        eval_every_n_epochs = 100
        best_acc = 0.0
        for epoch in range(4000):
            #     train_acc = []
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

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
                    x = x.to(self.device)
                    y = y.to(self.device)

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

        logging.info('OA, AA_Mean, Kappa: %f, %f, %f, ', OA, AA_mean, Kappa)
        logging.info(str(("AA for each class: ", AA)))

        classification, confusion, Test_accuracy, each_acc, aa, kappa = reports(pred_best, label_best, dataset_name)

        classification = str(classification)
        confusion = str(confusion)
        file_name = os.path.join(self.save_path,"{}_200_linear_report.txt".format(dataset_name))
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

        return best_acc

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        self.predictor.train()
        self.online_network.train()
        self.target_network.train()
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean() 

    def forward(self, data1, data2, lbl1, lbl2, data_transform_1, data_transform_2):
        self.predictor.eval()
        self.online_network.eval()
        self.target_network.eval()
        loss_ = []
        data_1 = []
        data_2 = []
        lbl_ = []
        with torch.no_grad():
            _dataset = HSIDataset_2(range(len(data1)),
                                            samples_1=data1,
                                            samples_2=data2,
                                            labels=torch.cat((torch.unsqueeze(lbl1,dim=1), torch.unsqueeze(lbl2, dim=1)), dim=-1),
                                            transforms_1=data_transform_1,
                                            transforms_2=data_transform_2)
            _loader = DataLoader(_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, drop_last=False, shuffle=True)
            for batch_view_1, batch_view_2, batch_1, batch_2, lbl_12 in _loader:
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
                predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
                targets_to_view_2 = self.target_network(batch_view_1)
                targets_to_view_1 = self.target_network(batch_view_2)
                loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
                loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
                loss_.extend(loss.data)
                data_1.extend(batch_1)
                data_2.extend(batch_2)
                lbl_.extend(lbl_12)
        data_1 = torch.stack(data_1)
        data_2 = torch.stack(data_2)
        loss_ = torch.stack(loss_)
        lbl_ = torch.stack(lbl_)
        return loss_, data_1, data_2, lbl_


    def save_model(self, PATH):
        if self.n_gpu ==1:
            if isinstance(self.online_network, torch.nn.DataParallel):
                torch.save({
                    'online_network_state_dict': self.online_network.module.state_dict(),
                    'target_network_state_dict': self.target_network.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, PATH)
            else:
                torch.save({
                    'online_network_state_dict': self.online_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, PATH)
        else:
            torch.save({
                'online_network_state_dict': self.online_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)