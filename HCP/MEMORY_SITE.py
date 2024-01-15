import csv
import pandas as pd
import numpy as np

# 읽어올 엑셀 파일 지정
filename = '/home/djk/Documents/2022.12.13/ABIDEI_qc_filtered-lists_800.xlsx'

site = ['PITT','OLIN','OHSU','SDSU','TRINITY','UM_1','UM_2','USM','YALE','CMU','LEUVEN_1','LEUVEN_2','KKI','NYU','STANFORD','UCLA_1','UCLA_2','MAX_MUN','CALTECH','SBL']
print('site: ', len(site))

''' NUM OF SUBJECT '''
# # 엑셀 파일 읽어 오기
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7])

# ASD_list = []
# TC_list = []
# for i, row in df.iterrows():
#     if row['SITE_ID'] == 'SBL':
#         if row['DX_GROUP'] == 1:
#             print(row['SUB_ID'], '---------->', row['DX_GROUP'])
#             ASD_list.append(row['SUB_ID'])
#
#         if row['DX_GROUP'] == 2:
#             TC_list.append(row['SUB_ID'])
#
#
# print(ASD_list)
# print(TC_list)
# print()
# print('ASD num_of_subject: ',len(ASD_list))
# print('TC num_of_subject: ',len(TC_list))



## total
# total_list = []
#
# for list in site:
#     count = 0
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#             count += 1
#     total_list.append(count)
#
# print(total_list)

# [50, 28, 25, 27, 44, 88, 34, 67, 41, 11, 28, 28, 33, 172, 25, 64, 21, 46, 15, 26]

''' sex '''
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 10])




# for list in site:
#     TC_1_list = []
#     TC_2_list = []
#
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#
#             if row['DX_GROUP'] == 2:
#                 if row['SEX'] == 1:
#                     print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['SEX'])
#                     TC_1_list.append(row['SEX'])
#
#                 if row['SEX'] == 2:
#                     print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['SEX'])
#                     TC_2_list.append(row['SEX'])
#
#     print('SITE: ', list)
#     print('ASD num_of_1: ', len(TC_1_list))
#     print('ASD num_of_2: ', len(TC_2_list))
#     print(len(TC_1_list),'/',len(TC_2_list))
#     print()
#     print()


## TOTAL

# for list in site:
#     TC_1_list = []
#     TC_2_list = []
#
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#
#             if row['SEX'] == 1:
#                 print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['SEX'])
#                 TC_1_list.append(row['SEX'])
#
#             if row['SEX'] == 2:
#                 print(row['SUB_ID'], '---------->', row['DX_GROUP'], row['SEX'])
#                 TC_2_list.append(row['SEX'])
#
#     print('SITE: ', list)
#     print('ASD num_of_1: ', len(TC_1_list))
#     print('ASD num_of_2: ', len(TC_2_list))
#     print(len(TC_1_list),'/',len(TC_2_list))
#     print()
#     print()



''' AGE '''
## DX_GROUP = 1
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 9])
#
#
#
# for list in site:
#     ASD_AGE = 0
#     count = 0
#     list_ASD_AGE = []
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#             if row['DX_GROUP'] == 1:
#                     # print(row['SITE_ID'] , row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#                     ASD_AGE += row['AGE_AT_SCAN']
#                     list_ASD_AGE.append(row['AGE_AT_SCAN'])
#
#                     count += 1
#                     # print(count)
#
#
#
#     print(list)
#     print(ASD_AGE)
#     print(count)
#     print()
#
#     ASD_arr = np.array(list_ASD_AGE)
#     print('ASD_mean: ', np.mean(ASD_arr))
#     print('ASD_std: ', np.std(ASD_arr))
#     print()
#
#     avg_ASD_AGE = ASD_AGE / count
#     print(avg_ASD_AGE)
#     print(np.round_(np.mean(ASD_arr),5),'±',np.round_(np.std(ASD_arr),5))
#     print()
#     print()
#     print()


## DX_GROUP = 2
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 9])
#
#
#
# for list in site:
#     TC_AGE = 0
#     count = 0
#     list_TC_AGE = []
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#             if row['DX_GROUP'] == 2:
#                     # print(row['SITE_ID'] , row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#                     TC_AGE += row['AGE_AT_SCAN']
#                     list_TC_AGE.append(row['AGE_AT_SCAN'])
#
#                     count += 1
#                     # print(count)
#
#
#
#     print(list)
#     print(TC_AGE)
#     print(count)
#     print()
#
#     ASD_arr = np.array(list_TC_AGE)
#     print('ASD_mean: ', np.mean(ASD_arr))
#     print('ASD_std: ', np.std(ASD_arr))
#     print()
#
#     avg_ASD_AGE = TC_AGE / count
#     print(avg_ASD_AGE)
#     print(np.round_(np.mean(ASD_arr),5),'±',np.round_(np.std(ASD_arr),5))
#     print()
#     print()
#     print()



# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 9])
#
#
#
# for list in site:
#     TOTAL_AGE = 0
#     count = 0
#     list_TOTAL_AGE = []
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#                 print(row['SITE_ID'] , row['SUB_ID'], '---------->', row['DX_GROUP'], row['AGE_AT_SCAN'])
#                 TOTAL_AGE += row['AGE_AT_SCAN']
#                 list_TOTAL_AGE.append(row['AGE_AT_SCAN'])
#
#                 count += 1
#
#
#
#
#     print(list)
#     print(len(list_TOTAL_AGE))
#     print(TOTAL_AGE)
#     print(count)
#     print()
#
#     ASD_arr = np.array(list_TOTAL_AGE)
#     print('ASD_mean: ', np.mean(ASD_arr))
#     print('ASD_std: ', np.std(ASD_arr))
#     print()
#
#     avg_ASD_AGE = TOTAL_AGE / count
#     print(avg_ASD_AGE)
#     print(np.round_(np.mean(ASD_arr),5),'±',np.round_(np.std(ASD_arr),5))
#     print()
#     print()
#     print()










''' FIQ '''

## DX_GROUP = 1
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 11])
# df = df.fillna('')
#
#
#
# for list in site:
#
#     ASD_IQ = 0
#     count = 0
#     list_ASD_IQ = []
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#             if row['DX_GROUP'] == 1:
#                     if row['FIQ'] == -9999.0 or row['FIQ'] == '':
#                             print('site: ', row['SITE_ID'],'except subject: ', row['SUB_ID'])
#                             print()
#                             continue
#
#
#                     print('site: ', row['SITE_ID'], row['SUB_ID'], '---------->', row['DX_GROUP'], row['FIQ'])
#                     ASD_IQ += row['FIQ']
#                     list_ASD_IQ.append(row['FIQ'])
#
#                     count += 1
#                     print(count)
#
#     if count == 0 and ASD_IQ == 0:
#         print( 'count 0, ASD_IQ 0')
#         print()
#         continue
#
#
#     print(ASD_IQ)
#     print(len(list_ASD_IQ))
#     print(count)
#     print()
#
#     ASD_arr = np.array(list_ASD_IQ)
#     print('ASD_mean: ', np.mean(ASD_arr))
#     print('ASD_std: ', np.std(ASD_arr))
#     print()
#
#     avg_ASD_IQ = ASD_IQ / count
#     print('avg_ASD_IQ: ', avg_ASD_IQ)
#     print(np.round_(np.mean(ASD_arr), 5), '(', np.round_(np.std(ASD_arr), 5), ')', '→', len(list_ASD_IQ))
#     print()
#     print()
#     print()
#     print()


## total
# df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 11])
# df = df.fillna('')
#
#
#
# for list in site:
#
#     TC_IQ = 0
#     count = 0
#     list_TC_IQ = []
#
#
#     for i, row in df.iterrows():
#         if row['SITE_ID'] == list:
#
#             if row['FIQ'] == -9999.0 or row['FIQ'] == '':
#                     print('site: ', row['SITE_ID'],'except subject: ', row['SUB_ID'])
#                     print()
#
#                     continue
#
#
#             print('site: ', row['SITE_ID'], row['SUB_ID'], '---------->', row['DX_GROUP'], row['FIQ'])
#             TC_IQ += row['FIQ']
#             list_TC_IQ.append(row['FIQ'])
#             count += 1
#
#             print(count)
#
#
#     if count == 0 and TC_IQ == 0:
#         print('SITE: ', list, 'count 0, ASD_IQ 0')
#         print()
#         print()
#         print()
#         continue
#
#
#     print(TC_IQ)
#     print(len(list_TC_IQ))
#     print(count)
#     print()
#
#     ASD_arr = np.array(list_TC_IQ)
#     print('ASD_mean: ', np.mean(ASD_arr))
#     print('ASD_std: ', np.std(ASD_arr))
#     print()
#
#     avg_ASD_IQ = TC_IQ / count
#     print('avg_ASD_IQ: ', avg_ASD_IQ)
#     print(np.round_(np.mean(ASD_arr), 5), '(', np.round_(np.std(ASD_arr), 5), ')', '→', len(list_TC_IQ))
#     print()
#     print()
#     print()
#     print()










''' ADOS-ASD '''

## DX_GROUP = 1
df = pd.read_excel(filename, engine='openpyxl', usecols=[1, 4, 7, 12])
df = df.fillna('')


for list in site:
    ASD_ADOS = 0
    count = 0
    list_ASD_ADOS = []

    for i, row in df.iterrows():
        if row['SITE_ID'] == list:
            if row['DX_GROUP'] == 1:
                    if row['ADOS'] == -9999.0 or row['ADOS'] == '':
                            print('except subject: ', row['SUB_ID'], 'because: ', row['ADOS'])
                            print()
                            continue


                    print('site: ', row['SITE_ID'], row['SUB_ID'], '---------->', row['DX_GROUP'], row['ADOS'])
                    ASD_ADOS += row['ADOS']
                    list_ASD_ADOS.append(row['ADOS'])

                    count += 1
                    print(count)

    if count == 0 and ASD_ADOS == 0:
        print('SITE: ', list, 'count 0, ASD_IQ 0')
        print()
        print()
        print()
        continue



    print(ASD_ADOS)
    print(len(list_ASD_ADOS))
    print(count)
    print()

    ASD_arr = np.array(list_ASD_ADOS)
    print('ASD_mean: ', np.mean(ASD_arr))
    print('ASD_std: ', np.std(ASD_arr))
    print()

    avg_ASD_IQ = ASD_ADOS / count
    print('avg_ASD_IQ: ', avg_ASD_IQ)
    print(np.round_(np.mean(ASD_arr), 5), '(', np.round_(np.std(ASD_arr), 5), ')', '→', len(list_ASD_ADOS))
    print()
    print()
    print()
    print()




#
#     ASD_arr = np.array(list_ASD_IQ)
#     print('ASD_mean: ', np.mean(ASD_arr))
#     print('ASD_std: ', np.std(ASD_arr))
#     print()
#
#     avg_ASD_IQ = ASD_IQ / count
#     print('avg_ASD_IQ: ', avg_ASD_IQ)
#     print(np.round_(np.mean(ASD_arr), 5), '(', np.round_(np.std(ASD_arr), 5), ')', '→', len(list_ASD_IQ))
#     print()
#     print()
#     print()
#     print()