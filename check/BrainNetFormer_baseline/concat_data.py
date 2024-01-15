import os
import numpy as np
import scipy.io
import glob
import csv


phenotype = '/DataRead/ABIDE/ABIDE_I_Raw_Download/Phenotypic_V1_0b_preprocessed1.csv'
main_path = '/DataRead/ABIDE/ABIDE_I_Raw_Download/ABIDE_pcp/'
for prec_type in ['dparsf']:
    for atlas in ['cc200']:
        for prec_glob in ['filt_noglobal']:

            subject_list = []
            subject_list2 = []
            conn_list = []
            labl_list = []
            timeseries_list = []
            file_path = main_path + '{}/{}/{}/*/'.format(prec_type, atlas, prec_glob)
            for a_path in sorted(glob.glob(file_path)):
                ro_file = [f for f in os.listdir(a_path) if f.endswith('_rois_' + atlas + '.1D')]
                fl = os.path.join(a_path, ro_file[0])
                print("Reading timeseries file %s" % fl)
                data = np.loadtxt(fl, skiprows=0)
                if data.shape[0] == 77:
                    continue
                else:
                    subject_list.append(os.path.split(a_path)[0].split(os.sep)[-1])
                    timeseries_list.append(data)

                    for b_path in glob.glob(a_path+'/*_correlation.mat'):
                        conn = scipy.io.loadmat(b_path)['connectivity']
                        conn_list.append(conn)

            with open(phenotype) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    if row['SUB_ID'] in subject_list:
                        labl_list.append(int(row['DX_GROUP']))
                        subject_list2.append(row['SUB_ID'])
            assert subject_list == subject_list2

            print('{}_{}_{}'.format(prec_type, atlas, prec_glob))
            save_path = main_path + '/{}_{}_{}_without77.npz'.format(prec_type, atlas, prec_glob)
            np.savez(save_path, signal=np.array(timeseries_list), conn=np.array(conn_list), labels=np.array(labl_list), subID=subject_list)




# Get timeseries arrays for list of subjects
# def get_timeseries(subject_list, atlas_name, silence=False):
#     """
#         subject_list : list of short subject IDs in string format
#         atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
#     returns:
#         time_series  : list of timeseries arrays, each of shape (timepoints x regions)
#     """
#
#     timeseries = []
#     for i in range(len(subject_list)):
#         subject_folder = os.path.join(data_folder, subject_list[i])
#         ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
#         fl = os.path.join(subject_folder, ro_file[0])
#         if silence != True:
#             print("Reading timeseries file %s" % fl)
#         timeseries.append(np.loadtxt(fl, skiprows=0))
#
#     return timeseries