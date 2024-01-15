import os
import random

import matplotlib.pyplot as plt
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import glob
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sklearn.metrics as metrics
from tensorflow.keras.optimizers import Adam, Adadelta, Nadam
import os
from bert2.transformer import ProjectionLayer
from bert2.attention import Attention_SV, Attention_translation
from utils import MyAttention, BertEmbeddingsLayer
from params_flow.activations import gelu
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def slice2parts(x, h1, h2, h3):
 """ Define a tensor slice function
 """
 return x[:,h1:h2, :], x[:,h2:h3,:]

def slice(x, h1, h2):
 """ Define a tensor slice function
 """
 return x[:,h1:h2, :]
def gelu_erf(x):

    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))

def STCAL(max_seq_len, embedding_size):
    time_input = keras.layers.Input(shape=(max_seq_len, embedding_size), dtype='float32', name="time_input")
    ROI_input = keras.layers.Permute((2, 1), name='ROI_input')(time_input)
    # pheno_input = keras.layers.Input((4,), name='pheno_input')

    # position embedding
    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=embedding_size, vocab_size=1)(time_input) #TensorShape([None, 90, 116])
    ROI_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=max_seq_len, vocab_size=1)(ROI_input) #TensorShape([None, 116, 90])

    time_input_spliting1 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 30})(time_position_embedding)
    time_input_spliting2 = keras.layers.Lambda(slice, arguments={'h1': 15, 'h2': 45})(time_position_embedding)
    time_input_spliting3 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 60})(time_position_embedding)
    time_input_spliting4 = keras.layers.Lambda(slice, arguments={'h1': 45, 'h2': 75})(time_position_embedding)
    time_input_spliting5 = keras.layers.Lambda(slice, arguments={'h1': 60, 'h2': 90})(time_position_embedding)

    time_valueS1_l1, time_attentionS1_l1 = Attention_SV(num_heads=4, size_per_head=29)(time_input_spliting1) #TensorShape([None, 30, 116])
    time_valueS2_l1, time_attentionS2_l1 = Attention_SV(num_heads=4, size_per_head=29)(time_input_spliting2)
    time_valueS3_l1, time_attentionS3_l1 = Attention_SV(num_heads=4, size_per_head=29)(time_input_spliting3)
    time_valueS4_l1, time_attentionS4_l1 = Attention_SV(num_heads=4, size_per_head=29)(time_input_spliting4)
    time_valueS5_l1, time_attentionS5_l1 = Attention_SV(num_heads=4, size_per_head=29)(time_input_spliting5)

    time_attention_probS1_l1 = keras.layers.Softmax()(time_attentionS1_l1)
    time_attention_probS1_l1 = keras.layers.Dropout(0.1)(time_attention_probS1_l1)
    time_attention_scoreS1_l1 = MyAttention(num_heads=4, size_per_head=29)([time_valueS1_l1, time_attention_probS1_l1])
    projection_timeS1_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS1_l1, time_input_spliting1])
    intermediate_timeS1_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS1_l1)
    bert_out_timeS1_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS1_l1, projection_timeS1_l1])

    time_attention_probS2_l1 = keras.layers.Softmax()(time_attentionS2_l1)
    time_attention_probS2_l1 = keras.layers.Dropout(0.1)(time_attention_probS2_l1)
    time_attention_scoreS2_l1 = MyAttention(num_heads=4, size_per_head=29)([time_valueS2_l1, time_attention_probS2_l1])
    projection_timeS2_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS2_l1, time_input_spliting2])
    intermediate_timeS2_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS2_l1)
    bert_out_timeS2_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS2_l1, projection_timeS2_l1])

    time_attention_probS3_l1 = keras.layers.Softmax()(time_attentionS3_l1)
    time_attention_probS3_l1 = keras.layers.Dropout(0.1)(time_attention_probS3_l1)
    time_attention_scoreS3_l1 = MyAttention(num_heads=4, size_per_head=29)([time_valueS3_l1, time_attention_probS3_l1])
    projection_timeS3_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS3_l1, time_input_spliting3])
    intermediate_timeS3_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS3_l1)
    bert_out_timeS3_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS3_l1, projection_timeS3_l1])

    time_attention_probS4_l1 = keras.layers.Softmax()(time_attentionS4_l1)
    time_attention_probS4_l1 = keras.layers.Dropout(0.1)(time_attention_probS4_l1)
    time_attention_scoreS4_l1 = MyAttention(num_heads=4, size_per_head=29)([time_valueS4_l1, time_attention_probS4_l1])
    projection_timeS4_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS4_l1, time_input_spliting4])
    intermediate_timeS4_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS4_l1)
    bert_out_timeS4_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS4_l1, projection_timeS4_l1])

    time_attention_probS5_l1 = keras.layers.Softmax()(time_attentionS5_l1)
    time_attention_probS5_l1 = keras.layers.Dropout(0.1)(time_attention_probS5_l1)
    time_attention_scoreS5_l1 = MyAttention(num_heads=4, size_per_head=29)([time_valueS5_l1, time_attention_probS5_l1])
    projection_timeS5_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS5_l1, time_input_spliting5])
    intermediate_timeS5_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS5_l1)
    bert_out_timeS5_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS5_l1, projection_timeS5_l1])

    bert_out_timeS1_l1_part1, bert_out_timeS1_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15, 'h3': 30})(bert_out_timeS1_l1)
    bert_out_timeS2_l1_part1, bert_out_timeS2_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15, 'h3': 30})(bert_out_timeS2_l1)
    bert_out_timeS3_l1_part1, bert_out_timeS3_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15, 'h3': 30})(bert_out_timeS3_l1)
    bert_out_timeS4_l1_part1, bert_out_timeS4_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15, 'h3': 30})(bert_out_timeS4_l1)
    bert_out_timeS5_l1_part1, bert_out_timeS5_l1_part2 = keras.layers.Lambda(slice2parts, arguments={'h1': 0, 'h2': 15, 'h3': 30})(bert_out_timeS5_l1)

    bert_out_timeS1_begin = bert_out_timeS1_l1_part1
    bert_out_timeS1_one = keras.layers.Add()([bert_out_timeS1_l1_part2 * 0.5, bert_out_timeS2_l1_part1 * 0.5])
    bert_out_timeS1_two = keras.layers.Add()([bert_out_timeS2_l1_part2 * 0.5, bert_out_timeS3_l1_part1 * 0.5])
    bert_out_timeS1_three = keras.layers.Add()([bert_out_timeS3_l1_part2 * 0.5, bert_out_timeS4_l1_part1 * 0.5])
    bert_out_timeS1_four = keras.layers.Add()([bert_out_timeS4_l1_part2 * 0.5, bert_out_timeS5_l1_part1 * 0.5])
    bert_out_timeS1_end = bert_out_timeS5_l1_part2

    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_timeS1_begin, bert_out_timeS1_one])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1, bert_out_timeS1_two])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1, bert_out_timeS1_three])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1, bert_out_timeS1_four])
    bert_out_time_l1 = keras.layers.Concatenate(axis=1)([bert_out_time_l1, bert_out_timeS1_end])

    bert_out_time_l1 = keras.layers.LayerNormalization()(bert_out_time_l1) #shape=(None, 90, 116)

    ROI_value_l1, ROI_attention_l1 = Attention_SV(num_heads=10, size_per_head=9)(ROI_position_embedding)
    ROI_attention_prob_l1 = keras.layers.Softmax()(ROI_attention_l1)
    ROI_attention_prob_l1 = keras.layers.Dropout(0.1)(ROI_attention_prob_l1)
    ROI_attention_score_l1 = MyAttention(num_heads=10, size_per_head=9)([ROI_value_l1, ROI_attention_prob_l1])
    projection_ROI_l1 = ProjectionLayer(hidden_size=max_seq_len)([ROI_attention_score_l1, ROI_position_embedding])
    intermediate_ROI_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_ROI_l1)
    bert_out_ROI_l1 = ProjectionLayer(hidden_size=max_seq_len)([intermediate_ROI_l1, projection_ROI_l1])

    # -----------------------------------------------------------------------------------------------------------------
    # layer two resbert
    time_value_l2, time_attention_l2 = Attention_SV(num_heads=4, size_per_head=29)(bert_out_time_l1)
    ROI_value_l2, ROI_attention_l2 = Attention_SV(num_heads=10, size_per_head=9)(bert_out_ROI_l1)

    time_attention_prob_l2 = keras.layers.Softmax()(time_attention_l2)
    time_attention_prob_l2 = keras.layers.Dropout(0.1)(time_attention_prob_l2)
    time_attention_score_l2 = MyAttention(num_heads=4, size_per_head=29)([time_value_l2, time_attention_prob_l2])
    projection_time_l2 = ProjectionLayer(hidden_size=embedding_size)([time_attention_score_l2, bert_out_time_l1])
    intermediate_time_l2 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_time_l2)
    bert_out_time_l2 = ProjectionLayer(hidden_size=embedding_size)([intermediate_time_l2, projection_time_l2])

    ROI_attention_prob_l2 = keras.layers.Softmax()(ROI_attention_l2)
    ROI_attention_prob_l2 = keras.layers.Dropout(0.1)(ROI_attention_prob_l2)
    ROI_attention_score_l2 = MyAttention(num_heads=10, size_per_head=9)([ROI_value_l2, ROI_attention_prob_l2])
    projection_ROI_l2 = ProjectionLayer(hidden_size=max_seq_len)([ROI_attention_score_l2, bert_out_ROI_l1])
    intermediate_ROI_l2 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_ROI_l2)
    bert_out_ROI_l2 = ProjectionLayer(hidden_size=max_seq_len)([intermediate_ROI_l2, projection_ROI_l2])

    # -------------------------------------------------------------------------------
    # layer three co-attention and resbert

    time_value_l3 = Attention_SV(num_heads=4, size_per_head=29, score=False)(bert_out_time_l2)
    ROI_value_l3, ROI_attention_l3 = Attention_SV(num_heads=10, size_per_head=9)(bert_out_ROI_l2)

    ROI_attention_reshape = keras.layers.Reshape((-1, embedding_size, embedding_size))(ROI_attention_l3) #(None, None, 116, 116)
    mean_ROI_score = keras.layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(ROI_attention_reshape) #(None, 116, 116)
    mean_ROI_score = keras.layers.Permute((2, 1))(mean_ROI_score)
    time_coattention_score = keras.layers.Dense(90)(mean_ROI_score)
    time_coattention_score = keras.layers.Permute((2, 1))(time_coattention_score)
    time_coattention_prob = keras.layers.Softmax()(time_coattention_score)
    # time_coattention_prob = keras.layers.Dropout(0.1)(time_coattention_prob)
    fusion_time = keras.layers.Multiply()([time_coattention_prob, time_value_l3])
    projection_time = ProjectionLayer(hidden_size=embedding_size)([fusion_time, bert_out_time_l2])
    intermediate_time = keras.layers.Dense(units=4 * embedding_size, activation=gelu_erf)(projection_time)
    bert_out_time = ProjectionLayer(hidden_size=embedding_size)([intermediate_time, projection_time])

    ROI_attention_prob_l3 = keras.layers.Softmax()(ROI_attention_l3)
    # ROI_attention_prob_l3 = keras.layers.Dropout(0.1)(ROI_attention_prob_l3)
    ROI_attention_score_l3 = MyAttention(num_heads=10, size_per_head=9)([ROI_value_l3, ROI_attention_prob_l3])
    projection_ROI_l3 = ProjectionLayer(hidden_size=max_seq_len)([ROI_attention_score_l3, bert_out_ROI_l2])
    intermediate_ROI_l3 = keras.layers.Dense(units=4 * max_seq_len, activation=gelu_erf)(projection_ROI_l3)
    bert_out_ROI = ProjectionLayer(hidden_size=max_seq_len)([intermediate_ROI_l3, projection_ROI_l3])

    bert_out_time = keras.layers.Dropout(0.5)(bert_out_time) #(None, 90, 116)
    bert_out_ROI = keras.layers.Dropout(0.5)(bert_out_ROI) #(None, 116, 90)

    bert_out_time = keras.layers.GlobalAveragePooling1D()(bert_out_time) #(None, 116)
    bert_out_ROI = keras.layers.GlobalAveragePooling1D()(bert_out_ROI) #(None, 90)

    bert_out_concate = keras.layers.Concatenate()([bert_out_time, bert_out_ROI]) #shape=(None, 206)
    # merge = keras.layers.Concatenate()([bert_out_concate, pheno_input])
    dense = keras.layers.Dense(64, activation='linear')(bert_out_concate)
    dense = keras.layers.Dropout(0.5)(dense)
    dense02 = keras.layers.Dense(10, activation='linear')(dense)
    output = keras.layers.Dense(7, activation='softmax')(dense02) #TensorShape([None, 7])

    model_ADHD = keras.Model(inputs=[time_input], outputs=output)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ADHD.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model_ADHD



def writelog(file, line):
    file.write(line + '\n')
    print(line)






def data_augment(timeseries, label, task_list, crops):
    augment_data = []
    augment_label = []
    sk_data = []
    sk_label = []
    for i in range(len(timeseries)):
        data = (timeseries[i] - np.mean(timeseries[i], axis=0, keepdims=True)) / (np.std(timeseries[i], axis=0, keepdims=True) + 1e-9)
        sk_data.append(data)
        sk_label.append(label[i])
        max = timeseries[i].shape[0]
        range_list = range(90 + 1, int(max))
        random_index = random.sample(range_list, crops)

        for j in range(crops):
            r = random_index[j]
            augment_data.append(data[r - 90:r])
            for task_idx, _task in enumerate(task_list):
                if label[i] == _task:
                    label_idx = task_idx
                    augment_label.append(label_idx)

    return np.array(augment_data), np.array(augment_label), np.array(sk_data), np.array(sk_label)


def training_function(args):
    # make directories
    os.makedirs(os.path.join(args.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # GPU Configuration
    # gpu_id = args.gpu_id
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # gpu번호
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)



    crops = 10
    # define dataset
    timeseries_list, label_list = torch.load('/home/jwlee/HMM/STCAL/data/train_hcp-task_roi-aal.pth') #전체데이터 5944 -> 59440
    task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
    task_list = list(task_timepoints.keys())
    task_list.sort()



    aug_timeseries, aug_label, sk_data, sk_label = data_augment(timeseries_list, label_list, task_list, crops) #(59440, 90, 116) #(59440,)



    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=24)




    j_count = 0
    for train_index, test_index in kf.split(sk_data, sk_label):
        K.clear_session()
        os.makedirs(os.path.join(args.targetdir, 'model_weights', str(j_count)), exist_ok=True)
        os.makedirs(os.path.join(args.targetdir, 'model', str(j_count)), exist_ok=True)


        aug_x_train_list = []
        aug_x_test_list = []
        count = 0
        for i in range(sk_data.shape[0]):
            if i in train_index:
                for k in range(crops):
                    aug_x_train_list.append(count)
                    count = count + 1
            else:
                for k in range(crops):
                    aug_x_test_list.append(count)
                    count = count + 1

                    # training data
        x_train = aug_timeseries[aug_x_train_list]
        y_train = aug_label[aug_x_train_list]

        # shuffle
        index = [i for i in range(x_train.shape[0])]
        random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]

        max_seq_len = x_train.shape[1]
        embedding_size = x_train.shape[2]
        print(max_seq_len, embedding_size)

        model = STCAL(max_seq_len, embedding_size)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2)



        ckpt_full_path = os.path.join(args.targetdir, 'model_weights', str(j_count), '{epoch:02d}-{val_accuracy:.5f}.h5')
        checkpoint_callback = ModelCheckpoint(
            filepath=ckpt_full_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='max'
        )


        model.fit(x=[x_train], y=y_train, epochs=30, batch_size=128, validation_split=0.2, callbacks=[reduce_lr, checkpoint_callback])

        j_count += 1


def tt_function(args):


    # GPU Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # gpu번호
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)



    crops = 10
    # define dataset
    timeseries_list, label_list = torch.load('/home/jwlee/HMM/STCAL/data/test_hcp-task_roi-aal.pth')  # 전체데이터 5944 -> 59440
    task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
    task_list = list(task_timepoints.keys())
    task_list.sort()

    aug_timeseries, aug_label, sk_data, sk_label = data_augment(timeseries_list, label_list, task_list, crops)  # (59440, 90, 116) #(59440,)

    test_data = aug_timeseries
    test_label = aug_label




    CV_count = 10
    for i_count in range(CV_count):
        max_seq_len = test_data.shape[1]
        embedding_size = test_data.shape[2]
        print(max_seq_len, embedding_size)




        path = os.path.join(args.targetdir, 'model_weights', str(i_count), 'best')
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
        print(full_path)


        model = load_model(full_path, custom_objects={'BertEmbeddingsLayer':BertEmbeddingsLayer, 'Attention_SV':Attention_SV, 'MyAttention':MyAttention, 'ProjectionLayer':ProjectionLayer, 'gelu_erf':gelu_erf})
        test_loss, test_acc = model.evaluate(x=[test_data], y=test_label, batch_size=1)


        print('test_acc is : {}'.format(test_acc))
        # f = open(os.path.join(args.targetdir, 'model', str(i_count), 'best', 'test_acc.log'), 'a')
        # writelog(f, 'test_acc: %.4f' % (test_acc))
        # f.close()


        predict = model.predict(x=[test_data])
        pred = np.argmax(predict, 1)
        f1_score = metrics.f1_score(test_label, pred, average='macro')


        print('f1_score is : {}'.format(f1_score))
        # f = open(os.path.join(args.targetdir, 'model', str(i_count), 'best', 'f1_score.log'), 'a')
        # writelog(f, 'f1_score: %.4f' % (f1_score))
        # f.close()


        auc_score = metrics.roc_auc_score(test_label, predict, average='macro', multi_class='ovr')
        print('auc_score is : {}'.format(auc_score))
        print('---------------------------')











if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='MY-NETWORK')
        parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")
        parser.add_argument('-s', '--seed', type=int, default=1)
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_2')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-ds', '--sourcedir', type=str, default='/home/jwlee/HMM/STCAL/data')
        parser.add_argument('-dt', '--targetdir', type=str, default='/home/jwlee/HMM/STCAL/result')



        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--weights", default=True, help='pre-trained autoencoder weights')

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)
    print(args.exp_name)

    # training_function(args)

    if args.weights is not None:
        tt_function(args)
        # tt_analysis(args)