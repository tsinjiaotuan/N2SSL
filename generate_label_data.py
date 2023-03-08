import numpy as np
import scipy.io as sio
import random
import yaml

sample_200 = [31, 88, 10, 13, 8, 23, 9, 14, 4]
def rsampling(groundTruth, sample_num=None):  # divide dataset into train and test datasets
    if sample_num is None:
        sample_num = sample_200
    labels_loc = {}
    labeled = {}
    test = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        labeled[i] = indices[:sample_num[i]]
        test[i] = indices[sample_num[i]:]
    whole_indices = []
    labeled_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        labeled_indices += labeled[i]
        test_indices += test[i]
        np.random.shuffle(labeled_indices)
        np.random.shuffle(test_indices)
    return whole_indices, labeled_indices, test_indices

def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return whole_indices, train_indices, test_indices

def same_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

IN_PATH = './datasets'
config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
sample_n = [2,3,4,5,6,7,8,9,10, 100,200,300,400,500,600]

for data_i in range(2):
    # if data_i <=2 :
    #     continue
    dataset = config['datasets'][data_i]
    if dataset == 'UP':
        mat_data = sio.loadmat(IN_PATH + '/UP/PaviaU.mat')
        data_IN = mat_data['paviaU']
        mat_gt = sio.loadmat(IN_PATH + '/UP/PaviaU_gt.mat')
        gt_IN = mat_gt['paviaU_gt']
        sample_100 = [15, 44, 5, 7, 4, 11, 5, 7, 2]
        sample_200 = [31, 88, 10, 13, 8, 23, 9, 14, 4]
        sample_p_5 = [331, 932, 104, 153, 67, 251, 66, 184, 47]
        sample_p_10 = [663, 1865, 210, 306, 135, 503, 133, 368, 95]
        sample_n =[20, 0.05, 0.1]
    elif dataset == 'IN':
        mat_data = sio.loadmat(IN_PATH + '/IN/Indian_pines_corrected.mat')
        data_IN = mat_data['indian_pines_corrected']
        mat_gt = sio.loadmat(IN_PATH + '/IN/Indian_pines_gt.mat')
        gt_IN = mat_gt['indian_pines_gt']
        sample_100 = [1, 13, 10, 2, 4, 7, 1, 5, 2, 12, 20, 7, 2, 9, 4, 1]
        sample_200 = [2, 27, 19, 4, 9, 14, 2, 10, 3, 24, 41, 14, 4, 18, 7, 2]
        sample_p_10 = [5,143,83,24,48,73,3,48,2,97,246,59,21,127,39,9]
        sample_p_15 = [7, 215, 125, 36, 73, 110, 5, 72, 3, 146, 369, 89, 31, 190, 58, 14]
        sample_n = [0.1, 0.15]
    elif dataset == 'SA':
        mat_data = sio.loadmat(IN_PATH + '/SA/Salinas_corrected.mat')
        data_IN = mat_data['salinas_corrected']
        mat_gt = sio.loadmat(IN_PATH + '/SA/Salinas_gt.mat')
        gt_IN = mat_gt['salinas_gt']
        sample_100 = [3, 7, 4, 3, 5, 7, 7, 21, 11, 6, 2, 3, 2, 2, 13, 4]
        sample_200 = [7, 14, 7, 5, 10, 15, 13, 42, 23, 12, 4, 7, 3, 4, 27, 7]
    elif dataset == 'KSC':
        mat_data = sio.loadmat(IN_PATH + '/KSC/KSC.mat')
        data_IN = mat_data['KSC']
        mat_gt = sio.loadmat(IN_PATH + '/KSC/KSC_gt.mat')
        gt_IN = mat_gt['KSC_gt']
        sample_100 = [14, 5, 5, 5, 3, 5, 2, 8, 10, 8, 8, 10, 17]
        sample_200 = [29, 9, 10, 10, 6, 9, 4, 17, 20, 16, 16, 19, 35]

    new_gt_IN = gt_IN
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )
    n_trial = 50
    for i_sample_n in range(len(sample_n)):
        if sample_n[i_sample_n]<1:
            if sample_n[i_sample_n]==0.05:
                sample_ = sample_p_5
            elif sample_n[i_sample_n]==0.1:
                sample_ = sample_p_10
            elif sample_n[i_sample_n]==0.15:
                sample_ = sample_p_15
        elif sample_n[i_sample_n] <= 10 and sample_n[i_sample_n]>1:
            sample_ = [sample_n[i_sample_n] for i_tmp in range(len(sample_100))]
        elif sample_n[i_sample_n] == 20:
            sample_ = [sample_n[i_sample_n] for i_tmp in range(len(sample_100))]
        elif sample_n[i_sample_n] == 100:
            sample_ = sample_100
        elif sample_n[i_sample_n] == 200:
            sample_ = sample_200
        else:
            sample_ = [int(sample_n[i_sample_n]/100)*sample_100[i_tmp] for i_tmp in range(len(sample_100))]
        a = 1
        _, train_indices_eval, test_indices_eval = rsampling(gt, sample_num=sample_)
        #
        train_indices_eval = np.empty((len(train_indices_eval), n_trial), dtype=np.int)
        test_indices_eval  = np.empty((len(test_indices_eval), n_trial), dtype=np.int)
        #
        for i in range(n_trial):
            same_seeds(i)
            _, train_indices_eval[:,i], test_indices_eval[:,i] = rsampling(gt, sample_num=sample_)
        sio.savemat('./datasets/{}/Label_indices_{}.mat'.format(dataset, sample_n[i_sample_n]),
                    {"train_indices_eval": train_indices_eval, "test_indices_eval": test_indices_eval})