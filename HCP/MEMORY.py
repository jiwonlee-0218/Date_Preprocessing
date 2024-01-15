import csv
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ranksums


# 읽어올 엑셀 파일 지정
# filename = '/home/djk/Documents/important/2022.12.13/ABIDEI_qc_filtered-lists_800.xlsx'

''' NUM OF SUBJECT '''
# 엑셀 파일 읽어 오기
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7])
#
# ASD_list = []
# TC_list = []
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#         print(row['SUB_ID'], '---------->', row['DX_GROUP'])
#         ASD_list.append(row['SUB_ID'])
#
#     if row['DX_GROUP'] == 2:
#         TC_list.append(row['SUB_ID'])
#
#
# print(ASD_list)
# print(TC_list)
# print()
# print('ASD num_of_subject: ',len(ASD_list))
# print('TC num_of_subject: ',len(TC_list))


''' SEX '''
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 10])
#
# TC_1_list = []
# TC_2_list = []
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 2:
#         if row['SEX'] == 1:
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['SEX'])
#             TC_1_list.append(row['SEX'])
#
#         if row['SEX'] == 2:
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['SEX'])
#             TC_2_list.append(row['SEX'])
#
#
# print(TC_1_list)
# print(TC_2_list)
# print()
# print('ASD num_of_1: ',len(TC_1_list))
# print('ASD num_of_2: ',len(TC_2_list))


''' AGE '''

## DX_GROUP = 1
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 9])
#
# ASD_AGE = 0
# count = 0
# list_ASD_AGE = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#             ASD_AGE += row['AGE_AT_SCAN']
#             list_ASD_AGE.append(row['AGE_AT_SCAN'])
#
#             count += 1
#             print(count)
#
# print(ASD_AGE)
# print(count)
# print()
#
# ASD_arr = np.array(list_ASD_AGE)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_AGE = ASD_AGE / count
# print(avg_ASD_AGE)


## DX_GROUP = 2
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 9])
#
# TC_AGE = 0
# count = 0
# list_TC_AGE = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 2:
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#             TC_AGE += row['AGE_AT_SCAN']
#             list_TC_AGE.append(row['AGE_AT_SCAN'])
#
#             count += 1
#             print(count)
#
# print(TC_AGE)
# print(count)
# print()
#
# ASD_arr = np.array(list_TC_AGE)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_AGE = TC_AGE / count
# print(avg_ASD_AGE)



''' FIQ '''

## DX_GROUP = 1
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 11])
# df = df.fillna('')
#
# ASD_IQ = 0
# count = 0
# list_ASD_IQ = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             if row['FIQ'] == -9999.0 or row['FIQ'] == '':
#                     print('except subject: ', row['SUB_ID'])
#                     print()
#                     continue
#
#
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['FIQ'])
#             ASD_IQ += row['FIQ']
#             list_ASD_IQ.append(row['FIQ'])
#
#             count += 1
#             print(count)
#
# print(ASD_IQ)
# print(count)
# print()
#
# ASD_arr = np.array(list_ASD_IQ)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_IQ = ASD_IQ / count
# print('avg_ASD_IQ: ', avg_ASD_IQ)


## DX_GROUP = 2
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 11])
# df = df.fillna('')
#
# TC_IQ = 0
# count = 0
# list_TC_IQ = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 2:
#             if row['FIQ'] == -9999.0 or row['FIQ'] == '':
#                     print('except subject: ', row['SUB_ID'])
#                     print()
#                     continue
#
#
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['FIQ'])
#             TC_IQ += row['FIQ']
#             list_TC_IQ.append(row['FIQ'])
#
#             count += 1
#             print(count)
#
# print()
# print(TC_IQ)
# print(count)
# print()
#
# TC_arr = np.array(list_TC_IQ)
# print('ASD_mean: ', np.mean(TC_arr))
# print('ASD_std: ', np.std(TC_arr))
# print()
#
# avg_TC_IQ = TC_IQ / count
# print('avg_ASD_IQ: ', avg_TC_IQ)



''' ADOS-ASD '''

## DX_GROUP = 1
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 12])
# df = df.fillna('')
#
# ASD_ADOS = 0
# count = 0
# list_ASD_ADOS = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             if row['ADOS'] == -9999.0 or row['ADOS'] == '':
#                     print('except subject: ', row['SUB_ID'], 'because: ', row['ADOS'])
#                     print()
#                     continue
#
#
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['ADOS'])
#             ASD_ADOS += row['ADOS']
#             list_ASD_ADOS.append(row['ADOS'])
#
#             count += 1
#             print(count)
#
# print(ASD_ADOS)
# print(count)
# print()
#
# ASD_arr = np.array(list_ASD_ADOS)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_IQ = ASD_ADOS / count
# print('avg_ASD_IQ: ', avg_ASD_IQ)


## DX_GROUP = 2
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 12])
# df = df.fillna('')
#
# TC_ADOS = 0
# count = 0
# list_TC_ADOS = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 2:
#             if row['ADOS'] == -9999.0 or row['ADOS'] == '' or row['ADOS']:
#                     print('except subject: ', row['SUB_ID'], 'because: ', row['ADOS'])
#                     print()
#                     continue
#
#
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['ADOS'])
#             TC_ADOS += row['ADOS']
#             list_TC_ADOS.append(row['ADOS'])
#
#             count += 1
#             print(count)
#
# print(TC_ADOS)
# print(count)
# print()
#
# ASD_arr = np.array(list_TC_ADOS)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_IQ = TC_ADOS / count
# print('avg_ASD_IQ: ', avg_ASD_IQ)



# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 12])
# df = df.fillna('')
#
# ASD_ADOS = 0
# count = 0
# list_ASD_ADOS = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             if row['ADOS'] == 0 :
#                     print('except subject: ', row['SUB_ID'], 'because: ', row['ADOS'])
#                     print()
#                     continue






''' t-test age '''
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 9])
#
# ASD_AGE = 0
# count = 0
# list_ASD_AGE = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             # print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#             ASD_AGE += row['AGE_AT_SCAN']
#             list_ASD_AGE.append(row['AGE_AT_SCAN'])
#
#             count += 1
#             # print(count)
#
# print(ASD_AGE)
# print(count)
# print()
#
# ASD_arr = np.array(list_ASD_AGE)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_AGE = ASD_AGE / count
# print(avg_ASD_AGE)
#
#
#
# TC_AGE = 0
# count_T = 0
# list_TC_AGE = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 2:
#             # print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#             TC_AGE += row['AGE_AT_SCAN']
#             list_TC_AGE.append(row['AGE_AT_SCAN'])
#
#             count_T += 1
#             # print(count_T)
#
# print(TC_AGE)
# print(count_T)
# print()
#
# TC_arr = np.array(list_TC_AGE)
# print('TC_mean: ', np.mean(TC_arr))
# print('TC_std: ', np.std(TC_arr))
# print()
#
# avg_TC_AGE = TC_AGE / count_T
# print(avg_TC_AGE)
#
#
# T, P = ttest_ind(ASD_arr, TC_arr)









''' t-test FIQ '''

## DX_GROUP = 1
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 11])
# df = df.fillna('')
#
# ASD_IQ = 0
# count = 0
# list_ASD_IQ = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             if row['FIQ'] == -9999.0 or row['FIQ'] == '':
#                     # print('except subject: ', row['SUB_ID'])
#                     # print()
#                     continue
#
#
#             # print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['FIQ'])
#             ASD_IQ += row['FIQ']
#             list_ASD_IQ.append(row['FIQ'])
#
#             count += 1
#             # print(count)
#
# print(ASD_IQ)
# print(count)
# print()
#
# ASD_arr = np.array(list_ASD_IQ)
# print('ASD_mean: ', np.mean(ASD_arr))
# print('ASD_std: ', np.std(ASD_arr))
# print()
#
# avg_ASD_IQ = ASD_IQ / count
# print('avg_ASD_IQ: ', avg_ASD_IQ)
#
#
# ## DX_GROUP = 2
#
# TC_IQ = 0
# count_T = 0
# list_TC_IQ = []
#
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 2:
#             if row['FIQ'] == -9999.0 or row['FIQ'] == '':
#                     # print('except subject: ', row['SUB_ID'])
#                     # print()
#                     continue
#
#
#             # print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['FIQ'])
#             TC_IQ += row['FIQ']
#             list_TC_IQ.append(row['FIQ'])
#
#             count_T += 1
#             # print(count_T)
#
# print()
# print(TC_IQ)
# print(count_T)
# print()
#
# TC_arr = np.array(list_TC_IQ)
# print('ASD_mean: ', np.mean(TC_arr))
# print('ASD_std: ', np.std(TC_arr))
# print()
#
# avg_TC_IQ = TC_IQ / count_T
# print('avg_ASD_IQ: ', avg_TC_IQ)
#
#
# T, P = ttest_ind(ASD_arr, TC_arr)
# # print(np.round(T, 5))
# # print(np.round(P, 5))









''' t-test sex '''
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 7, 10])
#
# ASD_sex_list = []
# TC_sex_list = []
# for i, row in df.iterrows():
#     if row['DX_GROUP'] == 1:
#             ASD_sex_list.append(row['SEX'])
#
#
#     if row['DX_GROUP'] == 2:
#             TC_sex_list.append(row['SEX'])
#
#
# print(len(ASD_sex_list))
# print(len(TC_sex_list))
# ASD_sex_arr = np.array(ASD_sex_list)
# TC_sex_arr = np.array(TC_sex_list)
#
# T, P = ranksums(ASD_sex_arr,TC_sex_arr)
# print(T)
# print(P)
#
#
# print(np.round(T, 5))
# print(np.round(P, 5))




######################################################### MDD ##########################################################
# 읽어올 엑셀 파일 지정
filename = '/DataCommon/jwlee/Stat_Sub_Info_848MDDvs794NC_dropsite4_830MDDvs771NC.xlsx'

''' NUM OF SUBJECT '''
# 엑셀 파일 읽어 오기
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2])
#
# MDD_list = []
# NC_list = []
# for i, row in df.iterrows():
#     if row['Dx'] == 1:
#         print(row['SubID'], '---------->', row['Dx'])
#         MDD_list.append(row['SubID'])
#
#     if row['Dx'] == -1:
#         NC_list.append(row['SubID'])
#
#
# print(MDD_list)
# print(NC_list)
# print()
# print('MDD num_of_subject: ',len(MDD_list))
# print('NC num_of_subject: ',len(NC_list))






''' SEX '''
df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 3])

# MDD_1_list = []
# MDD_2_list = []
# for i, row in df.iterrows():
#     if row['Dx'] == 1:
#         if row['Sex'] == 1:
#             print(row['SubID'], '---------->', row['Dx'], row['Sex'])
#             MDD_1_list.append(row['Sex'])
#
#         if row['Sex'] == 2:
#             print(row['SubID'], '---------->', row['Dx'], row['Sex'])
#             MDD_2_list.append(row['Sex'])
#
#
# print(MDD_1_list)
# print(MDD_2_list)
# print()
# print('MDD num_of_1: ',len(MDD_1_list))
# print('MDD num_of_2: ',len(MDD_2_list))
# print('sum MDD', len(MDD_1_list) + len(MDD_2_list))


# NC_1_list = []
# NC_2_list = []
# for i, row in df.iterrows():
#     if row['Dx'] == -1:
#         if row['Sex'] == 1:
#             print(row['SubID'], '---------->', row['Dx'], row['Sex'])
#             NC_1_list.append(row['Sex'])
#
#         if row['Sex'] == 2:
#             print(row['SubID'], '---------->', row['Dx'], row['Sex'])
#             NC_2_list.append(row['Sex'])
#
#
# print(NC_1_list)
# print(NC_2_list)
# print()
# print('NC num_of_1: ',len(NC_1_list))
# print('NC num_of_2: ',len(NC_2_list))
# print('sum NC', len(NC_1_list) + len(NC_2_list))




''' AGE '''

## Dx = 1 (MDD)
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 4])
#
# MDD_AGE = 0
# count = 0
# list_MDD_AGE = []
#
# for i, row in df.iterrows():
#     if row['Dx'] == 1:
#             print(row['SubID'], '---------->', row['Dx'], row['Age'])
#             MDD_AGE += row['Age']
#             list_MDD_AGE.append(row['Age'])
#
#             count += 1
#             print(count)
# print()
# print(MDD_AGE)
# print(count)
# print()
#
# MDD_arr = np.array(list_MDD_AGE)
# print('MDD_mean: ', np.round_(np.mean(MDD_arr), 2))
# print('MDD_std: ', np.round_(np.std(MDD_arr), 2))
# print()
#
# avg_MDD_AGE = MDD_AGE / count
# print(np.round_(avg_MDD_AGE, 2))


## Dx = -1 (NC)
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 4])
#
# NC_AGE = 0
# count = 0
# list_NC_AGE = []
#
# for i, row in df.iterrows():
#     if row['Dx'] == -1:
#             print(row['SubID'], '---------->', row['Dx'], row['Age'])
#             NC_AGE += row['Age']
#             list_NC_AGE.append(row['Age'])
#
#             count += 1
#             print(count)
#
# print()
# print(NC_AGE)
# print(count)
# print()
#
# NC_arr = np.array(list_NC_AGE)
# print('NC_mean: ', np.round_(np.mean(NC_arr), 2))
# print('NC_std: ', np.round_(np.std(NC_arr), 2))
# print()
#
# avg_NC_AGE = NC_AGE / count
# print(np.round_(avg_NC_AGE, 2))










''' EDU '''

# # Dx = 1 (MDD)
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 6])
#
# MDD_Edu = 0
# count = 0
# list_MDD_Edu = []
#
# for i, row in df.iterrows():
#     if row['Dx'] == 1:
#
#             print(row['SubID'], '---------->', row['Dx'], row['Edu'])
#             MDD_Edu += row['Edu']
#             list_MDD_Edu.append(row['Edu'])
#
#             count += 1
#             print(count)
#
# print()
# print(MDD_Edu)
# print(count)
# print()
#
# MDD_arr = np.array(list_MDD_Edu)
# print('MDD_mean: ', np.round_(np.mean(MDD_arr), 2))
# print('MDD_std: ', np.round_(np.std(MDD_arr), 2))
# print()
#
# avg_MDD_Edu = MDD_Edu / count
# print('avg_MDD_Edu: ', np.round_(avg_MDD_Edu, 2))


## Dx = -1 (NC)
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 6])
#
# NC_Edu = 0
# count = 0
# list_NC_Edu = []
#
# for i, row in df.iterrows():
#     if row['Dx'] ==  -1:
#
#             print(row['SubID'], '---------->', row['Dx'], row['Edu'])
#             NC_Edu += row['Edu']
#             list_NC_Edu.append(row['Edu'])
#
#             count += 1
#             print(count)
#
# print()
# print(NC_Edu)
# print(count)
# print()
#
# NC_arr = np.array(list_NC_Edu)
# print('NC_mean: ', np.round_(np.mean(NC_arr), 2))
# print('NC_std: ', np.round_(np.std(NC_arr), 2))
# print()
#
# avg_NC_Edu = NC_Edu / count
# print('avg_NC_Edu: ', np.round_(avg_NC_Edu, 2))












''' t-test age '''
df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 4])

MDD_AGE = 0
count = 0
list_MDD_AGE = []

for i, row in df.iterrows():
    if row['Dx'] == 1:
            # print(row['SubID'], '---------->', row['Dx'], row['Age'])
            MDD_AGE += row['Age']
            list_MDD_AGE.append(row['Age'])

            count += 1
            # print(count)

print(MDD_AGE)
print(count)
print()

MDD_arr = np.array(list_MDD_AGE)
print('MDD_mean: ', np.mean(MDD_arr))
print('MDD_std: ', np.std(MDD_arr))
print()

avg_MDD_AGE = MDD_AGE / count
print('MDD 나이 평균 ', np.round_(avg_MDD_AGE,2))
print()


NC_AGE = 0
count_T = 0
list_NC_AGE = []

for i, row in df.iterrows():
    if row['Dx'] == -1:
            # print(row['SubID'], '---------->', row['Dx'], row['Age'])
            NC_AGE += row['Age']
            list_NC_AGE.append(row['Age'])

            count_T += 1
            # print(count_T)

print(NC_AGE)
print(count_T)
print()

NC_arr = np.array(list_NC_AGE)
print('NC_mean: ', np.mean(NC_arr))
print('NC_std: ', np.std(NC_arr))
print()

avg_NC_AGE = NC_AGE / count_T
print('NC 나이 평균', np.round_(avg_NC_AGE, 2))

print()
T, P = ttest_ind(MDD_arr, NC_arr)
TT, PP = ttest_ind(NC_arr, MDD_arr)
print(f' t값은 {T:.4f} 이다. ')
print(f' p값은 {P:.4f} 이다. ')
print()
print(f' t값은 {TT:.4f} 이다. ')
print(f' p값은 {PP:.4f} 이다. ')







''' t-test EDU '''

# #Dx = 1 (MDD)
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 2, 6])
#
#
# MDD_Edu = 0
# count = 0
# list_MDD_Edu = []
#
# for i, row in df.iterrows():
#     if row['Dx'] == 1:
#             # print(row['SUB_ID'], '---------->', row['Dx'], row['Edu'])
#             MDD_Edu += row['Edu']
#             list_MDD_Edu.append(row['Edu'])
#
#             count += 1
#             # print(count)
# print()
# # print(MDD_Edu)
# # print(count)
# # print()
#
# MDD_arr = np.array(list_MDD_Edu)
# print('MDD_mean: ', np.mean(MDD_arr))
# print('MDD_std: ', np.std(MDD_arr))
# print()
#
# avg_MDD_Edu = MDD_Edu / count
# print('avg_MDD_Edu: ', np.round_(avg_MDD_Edu, 2))
# print(MDD_arr)
#
#
#
#
# #Dx = -1 (NC)
# NC_Edu = 0
# count_T = 0
# list_NC_Edu = []
#
# for i, row in df.iterrows():
#     if row['Dx'] == -1:
#
#             # print(row['SUB_ID'], '---------->', row['Dx'], row['Edu'])
#             NC_Edu += row['Edu']
#             list_NC_Edu.append(row['Edu'])
#
#             count_T += 1
#             # print(count_T)
#
# print()
# # print(NC_Edu)
# # print(count_T)
# # print()
#
# NC_arr = np.array(list_NC_Edu)
# print('MDD_mean: ', np.mean(NC_arr))
# print('MDD_std: ', np.std(NC_arr))
# print()
#
# avg_NC_Edu = NC_Edu / count_T
# print('avg_MDD_Edu: ', np.round_(avg_NC_Edu, 2))
# print(NC_arr)
#
#
# print()
# T, P = ttest_ind(MDD_arr, NC_arr)
# TT, PP = ttest_ind(NC_arr, MDD_arr)
# print(f' t값은 {T:.4f} 이다. ')
# print(f' p값은 {P:.4f} 이다. ')
# print()
# print(f' t값은 {TT:.4f} 이다. ')
# print(f' p값은 {PP:.4f} 이다. ')








### 남(MDD, NC) Vs 녀(MDD, NC)
# from scipy.stats import chi2_contingency
#
# ''' t-test sex '''
# df = pd.read_excel(filename, engine='openpyxl', usecols=[2, 3])
#
#
# crosstab = pd.crosstab(df['Sex'], df['Dx'])
# chi2, pvalue, dof, expected = chi2_contingency(crosstab)
# print(f' p값은 {pvalue:.4f} 이다. ')
# print()
#
#
#
#
# # crosstab = pd.crosstab(df['Dx'], df['Sex'])
# # chi2, pvalue, dof, expected = chi2_contingency(crosstab)
# # print(f' p값은 {pvalue:.4f} 이다. ')
# # print()
#
# # crosstab = pd.crosstab(df['Sex'], df['Dx'])
# # chi2, pvalue, dof, expected = chi2_contingency(crosstab)
# # print(f' p값은 {pvalue:.4f} 이다. ')







