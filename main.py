# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
#
#
# ########################################################################################################################
# # f_NHestablish.m 이웃 설정을 통해 데이터의 이웃 관계를 정의
#
# import torch
#
# def f_NHestablish(distbook, Gs, dist):
#     TP = len(distbook[Gs])
#     L = TP * Gs
#
#     nh_index = [None] * TP
#     nh_weigh = torch.zeros(TP)
#     for ks in range(Gs):
#         tempbook = [None] * Gs
#         for kk in range(Gs):
#             if tempbook[kk] is None:
#                 tempbook[kk] = torch.abs(torch.tensor(distbook[kk][ks]))  # tril
#
#         for kk in range(Gs):
#             if tempbook[kk] is None:
#                 tempbook[kk] = torch.abs(torch.tensor(distbook[ks][kk])).transpose(0, 1)  # triu
#
#         for kk in range(Gs):
#             if tempbook[kk] is None:
#                 tempbook[kk] = torch.zeros(TP, TP)  # add up zeros
#
#         tempbook = torch.cat(tempbook, dim=0)
#         tempbook[tempbook < dist] = 0
#         tempbook, sInd = torch.sort(tempbook, descending=True)
#
#         for kt in range(TP):
#             # neighborhood ruling out
#             NHtemp = tempbook[:10 * Gs, kt]
#             cutoff = torch.nonzero(NHtemp > 0, as_tuple=True)[0][-1].item() + 1 if NHtemp.gt(0).any() else 0
#             NHtemp = NHtemp[:cutoff]
#
#             if NHtemp.sum() > Gs and cutoff > Gs * 0.5:
#                 NumNH = sInd[:cutoff, kt]
#             else:
#                 cutoff = f_assemNH(NHtemp, Gs)
#                 NumNH = sInd[:cutoff, kt]
#
#             nh_index[kt] = torch.sort(NumNH)[0]
#             nh_weigh[kt] = NHtemp[:len(NumNH)].sum()
#
#     NH = {'INX': nh_index, 'SOW': nh_weigh}
#     return NH
#
#
# def f_assemNH(D, Gs):
#     cutoff = torch.nonzero(D > 0, as_tuple=True)[0][-1].item() + 1 if D.gt(0).any() else 0
#     D = D[:cutoff]
#
#     if cutoff > Gs * 0.5 and D.sum() > 1:  #이웃의 크기가 그룹 크기의 0.5배 이상이고 해당 범위 내의 데이터 합계가 1보다 크다면, 유효한 이웃으로 간주
#         return cutoff
#     else:
#         return 0
#
#
#
# ########################################################################################################################
# # f_GSL_appr.m  이웃을 바탕으로 데이터의 재구성 및 차원 축소를 수행
#
# import torch
#
# def f_GSL_appr(neighbors, Esvd):
#     N = Esvd['vecs'].shape[1] #고유벡터의 수
#     P = Esvd['vecs'][0].shape[0] #각 고유벡터의 길이
#
#     Y = {'npe': [], 'cof': []}
#
#     for kn in range(N):
#         NHs = neighbors['INX'][:, kn]
#         KI = neighbors['KI'][kn]
#         NoK = neighbors['Len'][kn]
#         TP = Esvd['vecs'][kn].shape[1]
#         Ytemp = torch.zeros((P, NoK))
#         Coef = torch.zeros((TP, NoK))
#
#         for kt in range(NoK):
#             Ind = NHs[KI[kt]] - 1
#             Rind = Ind % TP
#             Dind = Ind // TP
#             Lcomps = len(Ind)
#             comps = torch.zeros((P, Lcomps))
#
#             for ks in range(Lcomps):
#                 comps[:, ks] = Esvd['vecs'][Dind[ks]][:, Rind[ks]]
#
#             centraComp = Esvd['vecs'][kn][:, KI[kt]]
#             w = torch.linalg.lstsq(comps, centraComp).solution
#             Ytemp[:, kt] = torch.matmul(comps, w)
#             Coef[:, kt] = Esvd['coef'][kn][:, KI[kt]]
#
#         Y['npe'].append(Ytemp)
#         Y['cof'].append(Coef)
#
#     return Y
#
#
#
# ########################################################################################################################
# # f_NPE_main.m 위의 모든 단계를 통합하여 전체 NPE 기반 차원 축소 프로세스를 실행하고, 최종 결과를 저장
#
# import torch
# import os
# from sklearn.decomposition import PCA
#
# def f_NPE_main(fdir, otdir, opts):
#     # 초기 파싱
#     opts = f_npe_parse(fdir, opts)
#
#     # NPE 기반 차원 축소
#     if not opts['initPOOL']:
#         Y = f_NPE_OneGroup(fdir, opts)
#     else:
#         # Y = f_NPE_POOL(fdir, opts) # 추가 구현 필요
#         pass
#
#     # 결과 저장
#     ndir = os.path.join(otdir, 'NPE')
#     if not os.path.exists(ndir):
#         os.makedirs(ndir)
#
#     for ks in range(opts['num_subjects']):
#         strs = fdir[Y['IND'][ks]].split('\\')
#         name = 'npe_' + strs[-1][:-4]
#         Ynpe = Y['npe'][ks]
#         torch.save(Ynpe, os.path.join(ndir, name))
#
#     # 시간적 연결 PCA
#     cY = torch.cat(Y['npe'], dim=1)
#     pca = PCA(n_components=min(cY.size()))  # PCA 구성 요소 수 설정 필요
#     score = pca.fit_transform(cY.T.numpy())  # PyTorch 텐서를 NumPy 배열로 변환
#     coeff = pca.components_
#     latent = pca.explained_variance_
#
#     Z = {'score': score, 'coeff': coeff, 'latent': latent}
#     return Z
#
# # f_npe_parse, f_NPE_OneGroup 함수는 PyTorch로 구현 필요
# # ...



#####################################################################################################################################################
# 필요한 라이브러리 임포트
import numpy as np

def compute_weigth(training,region_num):
    S=[]
    w=[]
    summm=0

   ####normalizing each eigen_value_vector seperately
    for sub_iter in range(training.shape[0]):
        eigen_values = training[sub_iter][0].copy()

        summm=np.sum(eigen_values)

        for j_iter in range(len(eigen_values)):
            eigen_values[j_iter]=eigen_values[j_iter]/summm

        S.append(eigen_values)


    S=np.array(S)

    summ=0

    for i_iter in range(region_num):
        w.append(np.mean(S[:,i_iter]))
        summ=summ+np.mean(S[:,i_iter])#W[i_iter]

    for i_iter in range(region_num):
        w[i_iter]=w[i_iter]/summ

    w=np.array(w)

    return (w)


def cosine_sim(a, b):
    # cos_sim = inner(a, b)/(norm(a)*norm(b))
    return abs(np.inner(a, b))  # abs(cos_sim)

def Eros(sub_test_index, sub_train_index, eigen_vecs_vals, eigen_vecs_vals_test, W):
    eig_vecs1 = eigen_vecs_vals[sub_train_index]
    eig_vecs2 = eigen_vecs_vals_test[sub_test_index]
    summ = 0
    for ie in range(3):  # for over number of regions
        summ = summ + (W[ie] * cosine_sim(eig_vecs1[ie], eig_vecs2[ie]))

    return summ




def knn_Eros(train, test, k, W):
    count_dorost = 0
    count_healthy = 0  # predicted healthy correct
    count_adhd = 0  # predicted adhd correct
    total_predicted_adhd = 0
    healthy = 0
    adhd = 0
    trainlabel = train[:, 2]
    testlabel = test[:, 2]

    eigen_vecs_train = train[:, 1]
    eigen_vecs_test = test[:, 1]

    for i in range(len(test)):
        if (testlabel[i] == 0):
            healthy = healthy + 1
        if (testlabel[i] != 0):
            adhd = adhd + 1

        eros_dis = np.ones(len(train))  # list o distance of all training subjects to s test subject
        for j in range(len(train)):
            sim = Eros(i, j, eigen_vecs_train, eigen_vecs_test, W)

            eros_dis[j] = sim
        kkg = k * (-1)
        k_neighbors = np.argsort(np.array(eros_dis))[kkg:]

        knn_labels = trainlabel[k_neighbors]

        mode_data = mode(np.array(knn_labels), axis=0)
        mode_label = mode_data[0]
        if mode_data[0][0] != 0:
            total_predicted_adhd = total_predicted_adhd + 1

        if mode_data[0][0] == testlabel[i]:
            count_dorost = count_dorost + 1
            if mode_data[0][0] == 0:
                count_healthy = count_healthy + 1

            if mode_data[0][0] != 0:
                count_adhd = count_adhd + 1

        if mode_data[0][0] != testlabel[i] and mode_data[0][0] != 0 and testlabel[i] != 0:
            count_adhd = count_adhd + 1

    if adhd != 0:
        sensitivity = count_adhd / adhd
    else:
        print("------there is no adhd subject in test sub sample-------")
        print(testlabel)
        print("-------------")
        sensitivity = -1

    if healthy != 0:
        specificity = count_healthy / healthy
    else:
        specificity = -1

    total_acc = count_dorost / len(test)
    return total_acc, sensitivity, specificity, sensitivity + specificity


# 5명의 참가자 데이터 생성: 각각의 데이터는 3x4 임의 행렬, 레이블은 0부터 4까지
traindatacut = [(np.random.rand(3, 4), i) for i in range(5)]

# 각 데이터에 대해 고유값과 고유벡터 추출 및 저장
train_all_eigen_vals_vecs_labels = []

for j in range(len(traindatacut)):
    sub = traindatacut[j][0]  # 참가자 데이터 (3, 4)
    label = traindatacut[j][1]  # 레이블 0

    # 고유값과 고유벡터 계산
    eig_vals, eig_vecs = np.linalg.eig(np.cov(sub)) #np.cov(sub).shape -> (3, 3)
    eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]

    # 고유값에 따라 내림차순 정렬
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # 고유값과 고유벡터를 리스트에 추가
    temp_val = [eig_pair[0] for eig_pair in eig_pairs]
    temp_vec = [eig_pair[1] for eig_pair in eig_pairs]

    train_all_eigen_vals_vecs_labels.append([temp_val, temp_vec, label])

# NumPy 배열로 변환
train_all_eigen_vals_vecs_labels = np.array(train_all_eigen_vals_vecs_labels, dtype=object)
W=compute_weigth(train_all_eigen_vals_vecs_labels,3)
acc,sens,spef,sens_spef=knn_Eros(train_all_eigen_vals_vecs_labels,train_all_eigen_vals_vecs_labels,2,W) # k = 0~10

