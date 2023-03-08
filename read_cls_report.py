import os
import numpy as np
import argparse
# ######################## Read Acc 200 labelled ##################
dataset = 'UP' # UP IP KSC
n_lbl = '0.05'
pretrain=20
n = 23
n_cls = 8
# n_e = [6600,18561,2089,3051,1337,5006,1321,3668,943]
if dataset=='KSC':
    n=27
    n_cls = 12
    pretrain=40
    # n_e = [732,234,246,242,155,220,101,414,500,388,403,484,892]
if dataset=='IP':
    # n_e = [44,1401,811,233,474,716,26,468,17,948,2414,579,201,1247,379,91]
    n=30
    n_cls = 31
kappa_r = []
aa_r = []
oa_r = []
n_start = 8
for j in range(20):
    #path = os.path.join('acc_linear_classification_K=20','{}_pretrain{}'.format(dataset, pretrain),'N2SSL_{}_Linear_classification_report_{}.txt'.format(j+1, n_lbl))
    path = os.path.join('acc_finetuned_fromKSC_classification_pretrain40', dataset,
                        '{}_{}_FN_from_KSC_classification_report_{}.txt'.format(dataset, n_lbl,j))
    f = open(path, 'r')
    n_each = []
    cnt=0
    n_e = []
    for i in range(n+n_cls):
        line = f.readline()
        line = line.rstrip()
        if dataset=='IP':

            if i==0:
                oa_r.append(float(np.asarray(line[:8], dtype=np.float)))
            if i==2:
                kappa_r.append(float(np.asarray(line[:8], dtype=np.float)))
                a =1
            if i==4:
                aa_r.append(float(np.asarray(line[:8], dtype=np.float)))
                a =1
            elif i<n-1:
                if i>=n_start and i <=n_start+15:
                    parts = line.split(' ')
                    cnt_i = 0
                    for P in parts:
                        if P.isdigit():
                            cnt_i+=1
                            if cnt_i==1:
                                n_e.append(int(P))
                else:
                    continue
            elif i == n-1:
                parts = line[2:].split(' ')
                numbers = []
                for P in parts:
                    if P.isdigit():
                        numbers.append(int(P))
                # print(numbers)
                n_each.append(numbers[cnt])
                cnt += 1
            elif i>=n and i<=n+n_cls-5:
                if i%2==1:
                    parts = line.split(' ')
                    numbers = []
                    for P in parts:
                        if P.isdigit():
                            numbers.append(int(P))
                    # print(numbers)
                    n_each.append(numbers[cnt])
                    cnt+=1
            elif i>=n+n_cls-4 and i<=n+n_cls-2:
                if i%2==0:
                    parts = line.split(' ')
                    numbers = []
                    for P in parts:
                        if P.isdigit():
                            numbers.append(int(P))
                    # print(numbers)
                    n_each.append(numbers[0])
            elif i>=n+n_cls-1:
                parts = line[:-2].split(' ')
                numbers = []
                for P in parts:
                    if P.isdigit():
                        numbers.append(int(P))
                n_each.append(numbers[-1])
                # print(numbers)
        elif dataset == 'KSC' or 'UP':
            if i==0:
                oa_r.append(float(np.asarray(line[:8], dtype=np.float)))
            if i==2:
                kappa_r.append(float(np.asarray(line[:8], dtype=np.float)))
                a =1
            if i==4:
                aa_r.append(float(np.asarray(line[:8], dtype=np.float)))
                a =1
            elif i < n - 1:
                if i >= n_start and i <= n_start + n_cls:
                    parts = line.split(' ')
                    cnt_i = 0
                    for P in parts:
                        if P.isdigit():
                            cnt_i += 1
                            if cnt_i == 1:
                                n_e.append(int(P))
                else:
                    continue
            elif i == n - 1:
                parts = line[2:].split(' ')
                numbers = []
                for P in parts:
                    if P.isdigit():
                        numbers.append(int(P))
                # print(numbers)
                n_each.append(numbers[cnt])
                cnt += 1
            elif i > n - 1 and i<=n + n_cls - 2:
                parts = line.split(' ')
                numbers = []
                for P in parts:
                    if P.isdigit():
                        numbers.append(int(P))
                # print(numbers)
                n_each.append(numbers[cnt])
                cnt += 1
            elif i >=n + n_cls - 1:
                parts = line[:-2].split(' ')
                numbers = []
                for P in parts:
                    if P.isdigit():
                        numbers.append(int(P))
                n_each.append(numbers[-1])
                # print(numbers)
    for n_acc in n_each:
        print(n_acc,end=" ")
    print("    ")

    # print(np.asarray(n_each))
    oa = sum(n_each)/sum(n_e)*100
    # print(oa, oa_r)
for n_acc in oa_r:
    print(n_acc)

print("    ")
print("Kappa    ")

for n_acc in kappa_r:
    print(n_acc)

print("    ")
print("AA    ")
for n_acc in aa_r:
    print(n_acc)