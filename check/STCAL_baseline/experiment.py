import os
import random
from dataload_craddock import dataloader, dataloader_hcp
from dataload import dataloader_v4
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
    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=embedding_size, vocab_size=1)(time_input) #TensorShape([None, 90, 200])
    ROI_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=max_seq_len, vocab_size=1)(ROI_input) #TensorShape([None, 200, 90])

    time_input_spliting1 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 30})(time_position_embedding)
    time_input_spliting2 = keras.layers.Lambda(slice, arguments={'h1': 15, 'h2': 45})(time_position_embedding)
    time_input_spliting3 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 60})(time_position_embedding)
    time_input_spliting4 = keras.layers.Lambda(slice, arguments={'h1': 45, 'h2': 75})(time_position_embedding)
    time_input_spliting5 = keras.layers.Lambda(slice, arguments={'h1': 60, 'h2': 90})(time_position_embedding)

    time_valueS1_l1, time_attentionS1_l1 = Attention_SV(num_heads=10, size_per_head=20)(time_input_spliting1) #TensorShape([None, 30, 116])
    time_valueS2_l1, time_attentionS2_l1 = Attention_SV(num_heads=10, size_per_head=20)(time_input_spliting2)
    time_valueS3_l1, time_attentionS3_l1 = Attention_SV(num_heads=10, size_per_head=20)(time_input_spliting3)
    time_valueS4_l1, time_attentionS4_l1 = Attention_SV(num_heads=10, size_per_head=20)(time_input_spliting4)
    time_valueS5_l1, time_attentionS5_l1 = Attention_SV(num_heads=10, size_per_head=20)(time_input_spliting5)

    time_attention_probS1_l1 = keras.layers.Softmax()(time_attentionS1_l1)
    time_attention_probS1_l1 = keras.layers.Dropout(0.1)(time_attention_probS1_l1)
    time_attention_scoreS1_l1 = MyAttention(num_heads=10, size_per_head=20)([time_valueS1_l1, time_attention_probS1_l1])
    projection_timeS1_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS1_l1, time_input_spliting1])
    intermediate_timeS1_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS1_l1)
    bert_out_timeS1_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS1_l1, projection_timeS1_l1])

    time_attention_probS2_l1 = keras.layers.Softmax()(time_attentionS2_l1)
    time_attention_probS2_l1 = keras.layers.Dropout(0.1)(time_attention_probS2_l1)
    time_attention_scoreS2_l1 = MyAttention(num_heads=10, size_per_head=20)([time_valueS2_l1, time_attention_probS2_l1])
    projection_timeS2_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS2_l1, time_input_spliting2])
    intermediate_timeS2_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS2_l1)
    bert_out_timeS2_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS2_l1, projection_timeS2_l1])

    time_attention_probS3_l1 = keras.layers.Softmax()(time_attentionS3_l1)
    time_attention_probS3_l1 = keras.layers.Dropout(0.1)(time_attention_probS3_l1)
    time_attention_scoreS3_l1 = MyAttention(num_heads=10, size_per_head=20)([time_valueS3_l1, time_attention_probS3_l1])
    projection_timeS3_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS3_l1, time_input_spliting3])
    intermediate_timeS3_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS3_l1)
    bert_out_timeS3_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS3_l1, projection_timeS3_l1])

    time_attention_probS4_l1 = keras.layers.Softmax()(time_attentionS4_l1)
    time_attention_probS4_l1 = keras.layers.Dropout(0.1)(time_attention_probS4_l1)
    time_attention_scoreS4_l1 = MyAttention(num_heads=10, size_per_head=20)([time_valueS4_l1, time_attention_probS4_l1])
    projection_timeS4_l1 = ProjectionLayer(hidden_size=embedding_size)([time_attention_scoreS4_l1, time_input_spliting4])
    intermediate_timeS4_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(projection_timeS4_l1)
    bert_out_timeS4_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS4_l1, projection_timeS4_l1])

    time_attention_probS5_l1 = keras.layers.Softmax()(time_attentionS5_l1)
    time_attention_probS5_l1 = keras.layers.Dropout(0.1)(time_attention_probS5_l1)
    time_attention_scoreS5_l1 = MyAttention(num_heads=10, size_per_head=20)([time_valueS5_l1, time_attention_probS5_l1])
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
    time_value_l2, time_attention_l2 = Attention_SV(num_heads=10, size_per_head=20)(bert_out_time_l1)
    ROI_value_l2, ROI_attention_l2 = Attention_SV(num_heads=10, size_per_head=9)(bert_out_ROI_l1)

    time_attention_prob_l2 = keras.layers.Softmax()(time_attention_l2)
    time_attention_prob_l2 = keras.layers.Dropout(0.1)(time_attention_prob_l2)
    time_attention_score_l2 = MyAttention(num_heads=10, size_per_head=20)([time_value_l2, time_attention_prob_l2])
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

    time_value_l3 = Attention_SV(num_heads=10, size_per_head=20, score=False)(bert_out_time_l2)
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
    output = keras.layers.Dense(1, activation='sigmoid')(dense02) #TensorShape([None, 7])

    model_ADHD = keras.Model(inputs=[time_input], outputs=output)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ADHD.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model_ADHD



def writelog(file, line):
    file.write(line + '\n')
    print(line)






def data_augment(timeseries, label, crops):
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
            augment_label.append(label[i])


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

    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id # gpu번호
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)




    for k in range(args.k_fold):
        K.clear_session()
        os.makedirs(os.path.join(args.targetdir, 'model', str(k)), exist_ok=True)
        os.makedirs(os.path.join(args.targetdir, 'model_weights', str(k)), exist_ok=True)



        # define dataset
        # tr_loader, vl_loader, ts_loader = dataloader(args, k)   # MDD
        tr_loader, vl_loader, ts_loader = dataloader_v4(args, k)  # ABIDE
        crops = 10


        num_trn = tr_loader[0].shape[0] #507
        trn_shf = np.random.permutation(num_trn) #shuffle
        data_shf, labl_shf = tr_loader[0][trn_shf], tr_loader[2][trn_shf]

        num_val = vl_loader[0].shape[0]
        val_shf = np.random.permutation(num_val) #shuffle
        val_data_shf, val_labl_shf = vl_loader[0][val_shf], vl_loader[2][val_shf]


        test_data, test_label = ts_loader[0], ts_loader[2]




        aug_train_data, aug_train_label, sk_train_data, sk_train_label = data_augment(data_shf, labl_shf, crops) #(5070, 90, 200) #(5070,) #(507,) #(507,)
        aug_val_data, aug_val_label, sk_val_data, sk_val_label = data_augment(val_data_shf, val_labl_shf, crops)  #(1690, 90, 200) #(1690,) #(169,) #(169,)
        aug_test_data, aug_test_label, sk_test_data, sk_test_label = data_augment(test_data, test_label, crops)  #(1700, 90, 200) #(1700,) #(170,) #(170,)


        max_seq_len = aug_train_data.shape[1]
        embedding_size = aug_train_data.shape[2]
        print(max_seq_len, embedding_size)

        model = STCAL(max_seq_len, embedding_size)
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2)

        ckpt_full_path = os.path.join(args.targetdir, 'model_weights', str(k), '{epoch:02d}-{val_accuracy:.5f}.h5')
        checkpoint_callback = ModelCheckpoint(
            filepath=ckpt_full_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )

        model.fit(x=[aug_train_data], y=aug_train_label, epochs=args.num_epochs, batch_size=args.minibatch_size, validation_data=([aug_val_data], aug_val_label), callbacks=[reduce_lr, checkpoint_callback])


        # test
        path = os.path.join(args.targetdir, 'model_weights', str(k))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getmtime)[-1]
        model = load_model(full_path, custom_objects={'BertEmbeddingsLayer': BertEmbeddingsLayer, 'Attention_SV': Attention_SV, 'MyAttention': MyAttention, 'ProjectionLayer': ProjectionLayer, 'gelu_erf': gelu_erf})

        ASD_predict = model.predict(x=[aug_test_data])
        binary_predictions = np.where(ASD_predict >= 0.5, 1, 0).flatten()
        test_acc2 = metrics.accuracy_score(aug_test_label, binary_predictions)
        test_loss, test_acc = model.evaluate(x=[aug_test_data], y=aug_test_label, batch_size=1)

        [[TN, FP], [FN, TP]] = confusion_matrix(aug_test_label, binary_predictions).astype(float)
        specificity = TN / (FP + TN)
        sensivity = recall = TP / (TP + FN)
        auc_score = metrics.roc_auc_score(aug_test_label, ASD_predict.flatten())


        print('---------------------------')
        f = open(os.path.join(args.targetdir, 'model', str(k), 'test_acc.log'), 'a')
        writelog(f, 'test_acc2: %.5f, test_acc: %.5f, sen: %.5f, spe: %.5f, auc: %.5f' % (test_acc2, test_acc, sensivity, specificity, auc_score))
        f.close()
        print()

















############################################################################################################################################################################################################################################
# import os
# import random
# import torch
# import numpy as np
# from BrainNetFormer import *
# from dataload_craddock import dataloader, dataloader_hcp
# from dataload import dataloader_v4
# import argparse
# from tqdm import tqdm
# from einops import rearrange, repeat
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid
# import glob
# import util
# import math
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score
# import torch.nn as nn
#
#
#
#
#
#
#
# def step(model, criterion, dyn_t, dyn_a, sampling_points, sampling_endpoints, t, a, label, train=True, clip_grad=0.0, device='cpu', optimizer=None):
#     if optimizer is None: model.eval()
#     else: model.train()
#
#
#     # @ main task prediction
#     logit, attention = model(dyn_t.to(device), dyn_a.to(device), t.to(device), a.to(device), sampling_endpoints)
#     logit = torch.sigmoid(logit)
#     loss = criterion(logit, label.unsqueeze(1).to(device))
#
#
#     # @ optimize model
#     if (train):
#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
#             if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
#             optimizer.step()
#
#
#     return logit, loss, attention
#
#
#
#
#
#
#
#
#
# def training_function(args):
#     # make directories
#     os.makedirs(os.path.join(args.targetdir, 'model'), exist_ok=True)
#     os.makedirs(os.path.join(args.targetdir, 'summary'), exist_ok=True)
#
#     # set seed and device
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#
#     # GPU Configuration
#     gpu_id = args.gpu_id
#     os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.manual_seed_all(args.seed)
#     print(device)
#
#
#     for k in range(args.k_fold):
#         os.makedirs(os.path.join(args.targetdir, 'model', str(k)), exist_ok=True)
#         os.makedirs(os.path.join(args.targetdir, 'model_weights', str(k)), exist_ok=True)
#
#
#
#         # define dataset
#         # tr_loader, vl_loader, ts_loader = dataloader(args, k)   # MDD
#         tr_loader, vl_loader, ts_loader = dataloader_v4(args, k)  # ABIDE
#
#
#
#         model = BrainNetFormer(
#             n_region=tr_loader[0].shape[2],  #n_region
#             hidden_dim=args.hidden_dim,
#             num_classes=args.num_classes,
#             num_heads=args.num_heads,
#             num_layers=args.num_layers,
#             sparsity=args.sparsity,
#             window_size=args.window_size,
#             dropout=args.dropout)
#         model.to(device)
#         criterion = torch.nn.BCELoss()
#
#         # define optimizer and learning rate scheduler
#         optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
#         # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=(  int(math.ceil(tr_loader[0].shape[0] / args.minibatch_size))  ), pct_start=0.2, div_factor=args.max_lr / args.lr, final_div_factor=1000)
#
#
#
#         # define logging objects
#         summary_writer = SummaryWriter(os.path.join(args.targetdir, 'summary',  str(k), 'train'), )
#         summary_writer_val = SummaryWriter(os.path.join(args.targetdir, 'summary',  str(k), 'val'), )
#
#
#
#         best_score = 0.0
#         for ep in range(args.num_epochs):
#             num_trn = tr_loader[0].shape[0] #507
#             num_val = vl_loader[0].shape[0]
#             trn_shf = torch.randperm(num_trn) #shuffle
#             data_shf, labl_shf = tr_loader[0][trn_shf].to(device), tr_loader[2][trn_shf].to(device)
#             val_shf = torch.randperm(num_val) #shuffle
#             val_data_shf, val_labl_shf = vl_loader[0][val_shf].to(device), vl_loader[2][val_shf].to(device)
#
#
#             test_data, test_label = ts_loader[0], ts_loader[2]
#
#
#             loss_accumulate = 0.0
#             acc_accumulate = 0.0
#             tr_predict_all = np.array([], dtype=int)
#             tr_prob_all = np.array([], dtype=int)
#             tr_labels_all = np.array([], dtype=int)
#
#             for i in tqdm(range(int(math.ceil(num_trn / args.minibatch_size)))):
#                 batchx, batchy = data_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)], labl_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)] #torch.Size([16, 115, 200]), torch.Size([16])
#
#                 dyn_t, sampling_points = util.bold.process_dynamic_t(batchx, args.window_size,
#                                                                      args.window_stride,
#                                                                      args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
#                 dyn_a, sampling_points = util.bold.process_dynamic_fc(batchx, args.window_size,
#                                                                       args.window_stride,
#                                                                       args.dynamic_length)  # torch.Size([16, 22, 200, 200])
#
#                 sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22
#
#                 a = util.bold.process_static_fc(batchx)  # torch.Size([16, 200, 200])
#                 t = batchx.permute(1, 0, 2)  # torch.Size([115, 16, 200])
#                 label = batchy
#
#
#
#
#                 logit, loss, attention = step(
#                     model=model,
#                     criterion=criterion,
#                     dyn_t=dyn_t,
#                     dyn_a=dyn_a,
#                     sampling_points=sampling_points,
#                     sampling_endpoints=sampling_endpoints,
#                     t=t,
#                     a=a,
#                     label=label,
#                     train=True,
#                     clip_grad=args.clip_grad,
#                     device=device,
#                     optimizer=optimizer)
#
#
#                 # @ main task
#                 loss_accumulate += loss.detach().cpu().numpy()
#                 probability = logit.detach().cpu().numpy().reshape(-1)
#                 prediction = (logit.cpu() > torch.FloatTensor([0.5])).float().view(-1)
#
#
#                 print(prediction)
#                 tr_predict_all = np.append(tr_predict_all, prediction)
#                 tr_prob_all = np.append(tr_prob_all, probability)
#                 tr_labels_all = np.append(tr_labels_all, label.cpu())
#
#
#
#
#             total_loss = loss_accumulate / int(math.ceil(num_trn / args.minibatch_size))
#             total_acc = metrics.accuracy_score(tr_labels_all, tr_predict_all)
#             tn, fp, fn, tp = confusion_matrix(tr_labels_all, tr_predict_all).ravel()
#             sensitivity = tp / (tp + fn)
#             specificity = tn / (tn + fp)
#             auc = roc_auc_score(tr_labels_all, tr_prob_all)
#
#
#             # @ main task
#             summary_writer.add_scalar('training loss', total_loss, ep)
#             summary_writer.add_scalar("training acc", total_acc, ep)
#             print(dict(epoch=ep, train_loss=total_loss, train_acc=total_acc, train_sensitivity=sensitivity, train_specificity=specificity, train_auc=auc))
#
#
#
#
#
#
#
#
#             predict_all = np.array([], dtype=int)
#             prob_all = np.array([], dtype=int)
#             labels_all = np.array([], dtype=int)
#             for i in tqdm(range(int(math.ceil(num_val / args.minibatch_size)))):
#                 with torch.no_grad():
#                     val_batchx, val_batchy = val_data_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)], val_labl_shf[i * args.minibatch_size:(i + 1) * (args.minibatch_size)]  # torch.Size([16, 115, 200]), torch.Size([16])
#
#                     dyn_t, sampling_points = util.bold.process_dynamic_t(val_batchx, args.window_size,
#                                                                          args.window_stride,
#                                                                          args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
#                     dyn_a, sampling_points = util.bold.process_dynamic_fc(val_batchx, args.window_size,
#                                                                           args.window_stride,
#                                                                           args.dynamic_length)  # torch.Size([16, 22, 200, 200])
#
#                     sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22
#
#                     a = util.bold.process_static_fc(val_batchx)  # torch.Size([16, 200, 200])
#                     t = val_batchx.permute(1, 0, 2)  # torch.Size([115, 16, 200])
#                     label = val_batchy
#
#                     logit, loss, attention = step(
#                         model=model,
#                         criterion=criterion,
#                         dyn_t=dyn_t,
#                         dyn_a=dyn_a,
#                         sampling_points=sampling_points,
#                         sampling_endpoints=sampling_endpoints,
#                         t=t,
#                         a=a,
#                         label=label,
#                         train=False,
#                         clip_grad=args.clip_grad,
#                         device=device,
#                         optimizer=None)
#
#
#                     loss_accumulate += loss.detach().cpu().numpy()
#                     probability = logit.detach().cpu().numpy().reshape(-1)
#                     prediction = (logit.cpu() > torch.FloatTensor([0.5])).float().view(-1)
#
#                     print(prediction)
#                     predict_all = np.append(predict_all, prediction)
#                     prob_all = np.append(prob_all, probability)
#                     labels_all = np.append(labels_all, label.cpu())
#
#
#
#
#             val_acc = metrics.accuracy_score(labels_all, predict_all)
#             tn, fp, fn, tp = confusion_matrix(labels_all, predict_all).ravel()
#             val_sensitivity = tp / (tp + fn)
#             val_specificity = tn / (tn + fp)
#             val_auc = roc_auc_score(labels_all, prob_all)
#
#
#
#
#
#             summary_writer_val.add_scalar('val acc', val_acc, ep)
#             print(dict(epoch=ep, val_acc=val_acc, val_sensitivity=val_sensitivity, val_specificity=val_specificity, val_auc=val_auc))
#
#
#             # test
#             tst_predict_all = np.array([], dtype=int)
#             tst_prob_all = np.array([], dtype=int)
#             tst_labels_all = np.array([], dtype=int)
#             for i in range(test_data.shape[0]):
#                 with torch.no_grad():
#                     batchx_t, batchy_t = test_data[i], test_label[i:i + 1]
#                     batchx_t = torch.tensor(batchx_t, dtype=torch.float32).unsqueeze(0).to(device)  # torch.Size([1, 200, 200])
#                     batchy_t = batchy_t.to(device)
#
#
#
#                     dyn_t, sampling_points = util.bold.process_dynamic_t(batchx_t, args.window_size,
#                                                                          args.window_stride,
#                                                                          args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
#                     dyn_a, sampling_points = util.bold.process_dynamic_fc(batchx_t, args.window_size,
#                                                                           args.window_stride,
#                                                                           args.dynamic_length)  # torch.Size([16, 22, 200, 200])
#
#                     sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22
#
#                     a = util.bold.process_static_fc(batchx_t)  # torch.Size([16, 200, 200])
#                     t = batchx_t.permute(1, 0, 2)  # torch.Size([115, 16, 200])
#                     label = batchy_t
#
#
#                     logit, loss, attention = step(
#                         model=model,
#                         criterion=criterion,
#                         dyn_t=dyn_t,
#                         dyn_a=dyn_a,
#                         sampling_points=sampling_points,
#                         sampling_endpoints=sampling_endpoints,
#                         t=t,
#                         a=a,
#                         label=label,
#                         train=False,
#                         clip_grad=args.clip_grad,
#                         device=device,
#                         optimizer=None)
#
#
#
#                     loss_accumulate += loss.detach().cpu().numpy()
#                     probability = logit.detach().cpu().numpy().reshape(-1)
#                     prediction = (logit.cpu() > torch.FloatTensor([0.5])).float().view(-1)
#
#                     tst_predict_all = np.append(tst_predict_all, prediction)
#                     tst_prob_all = np.append(tst_prob_all, probability)
#                     tst_labels_all = np.append(tst_labels_all, label.cpu())
#
#
#
#             test_acc = metrics.accuracy_score(tst_labels_all, tst_predict_all)
#             tn, fp, fn, tp = confusion_matrix(tst_labels_all, tst_predict_all).ravel()
#             test_sensitivity = tp / (tp + fn)
#             test_specificity = tn / (tn + fp)
#             test_auc = roc_auc_score(tst_labels_all, tst_prob_all)
#
#
#
#             if best_score < test_acc:
#                 best_score = test_acc
#                 best_epoch = ep
#
#                 # save checkpoint
#                 torch.save({
#                     'fold': k,
#                     'loss': total_loss,
#                     'epoch': ep,
#                     'model': model.state_dict(),
#                     'optimizer': optimizer.state_dict()},
#                     os.path.join(args.targetdir, 'model_weights', str(k), 'checkpoint_epoch_{}.pth'.format(ep)))
#
#                 f = open(os.path.join(args.targetdir, 'model', str(k), 'best_acc.log'), 'a')
#                 writelog(f, 'best_acc: %.5f for epoch: %d, sen: %.5f, spe: %.5f, auc: %.5f' % (best_score, best_epoch, test_sensitivity, test_specificity, test_auc))
#                 f.close()
#                 print('best score !!!')
#                 print()
#
#
#     summary_writer.close()
#     summary_writer_val.close()
#
#
#
#
#
#
# def tt_function(argv):
#     # GPU Configuration
#     gpu_id = args.gpu_id
#     os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  ##
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#
#     logger = util.logger.LoggerBrainNetFormer(argv.k_fold, args.num_classes)
#
#     for k in range(args.k_fold):
#
#         # define dataset
#         tr_loader, vl_loader, ts_loader = dataloader_v4(args, (k+1))
#
#         model = BrainNetFormer(
#             n_region=tr_loader[0].shape[2],
#             hidden_dim=args.hidden_dim,
#             num_classes=args.num_classes,
#             num_heads=args.num_heads,
#             num_layers=args.num_layers,
#             sparsity=args.sparsity,
#             window_size=args.window_size)
#
#         path = os.path.join(argv.targetdir, 'model_weights', str(k))
#         full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
#         checkpoint = torch.load(full_path)
#         model.load_state_dict(checkpoint['model'])
#         model.to(device)
#
#
#         criterion = torch.nn.BCEWithLogitsLoss()
#
#         # define logging objects
#         summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))
#
#         logger.initialize(k)
#         loss_accumulate = 0.0
#
#
#         test_data, test_label = ts_loader[0], ts_loader[2]
#         for i in range(test_data.shape[0]):
#             with torch.no_grad():
#                 batchx_t, batchy_t = test_data[i], test_label[i]
#
#
#                 dyn_t, sampling_points = util.bold.process_dynamic_t(batchx_t, args.window_size,
#                                                                      args.window_stride,
#                                                                      args.dynamic_length)  # torch.Size([16, 22, 200, 50]) len(sampling_points)=22
#                 dyn_a, sampling_points = util.bold.process_dynamic_fc(batchx_t, args.window_size,
#                                                                       args.window_stride,
#                                                                       args.dynamic_length)  # torch.Size([16, 22, 200, 200])
#
#                 sampling_endpoints = [p + args.window_size for p in sampling_points]  # len(sampling_endpoints)=22
#
#                 a = util.bold.process_static_fc(batchx_t)  # torch.Size([16, 200, 200])
#                 t = batchx_t.permute(1, 0, 2)  # torch.Size([115, 16, 200])
#                 label = batchy_t
#
#                 logit, loss, attention = step(
#                     model=model,
#                     criterion=criterion,
#                     dyn_t=dyn_t,
#                     dyn_a=dyn_a,
#                     sampling_points=sampling_points,
#                     sampling_endpoints=sampling_endpoints,
#                     t=t,
#                     a=a,
#                     label=label,
#                     train=False,
#                     clip_grad=args.clip_grad,
#                     device=device,
#                     optimizer=None)
#
#
#                 # @ main task
#                 pred = logit.argmax(1)
#                 prob = logit.softmax(1)
#                 logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
#                 loss_accumulate += loss.detach().cpu().numpy()
#
#         # summarize results
#         samples = logger.get(k)
#         metrics = logger.evaluate(k)
#         summary_writer.add_scalar('loss', loss_accumulate / int(math.ceil(test_data.shape[0] / args.minibatch_size)))
#         summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1])
#         [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key == 'fold']
#
#         summary_writer.flush()
#         print(metrics)
#         f = open(os.path.join(argv.targetdir, 'model', str(k), 'test_acc.log'), 'a')
#         writelog(f, 'test_acc: %.4f' % (metrics['accuracy']))
#         f.close()
#         print('test_acc is : {}'.format(metrics['accuracy']))



if __name__=='__main__':
    # parse options and make directories
    def get_arguments():
        parser = argparse.ArgumentParser(description='STCAL')
        parser.add_argument("--gpu_id", type=str, default="7", help="GPU id")
        parser.add_argument('-s', '--seed', type=int, default=1)
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_abide_v1')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-b', '--minibatch_size', type=int, default=128)
        parser.add_argument('-dt', '--targetdir', type=str, default='/DataCommon2/jwlee/check/STCAL/result')


        # dataset args
        parser.add_argument("--prec_glob", type=str, default='filt_noglobal', choices=['filt_global', 'filt_noglobal'], help="for ABIDE")
        # parser.add_argument("--prec_glob", type=str, default=False, choices=[True, False], help="for MDD")
        parser.add_argument("--prec_type", type=str, default='dparsf', choices=['cpac', 'dparsf'])
        parser.add_argument("--atlas", type=str, default='cc200', choices=['aal', 'ho', 'cc200'])
        parser.add_argument('--input_size', type=int, default=200)

        # training args
        # parser.add_argument('--window_size', type=int, default=40)
        # parser.add_argument('--window_stride', type=int, default=3)
        # parser.add_argument('--dynamic_length', type=int, default=115)
        # parser.add_argument('--lr', type=float, default=0.000001)
        # parser.add_argument('--max_lr', type=float, default=0.00001)
        # parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay for Adam optimizer", )
        # parser.add_argument('--reg_lambda', type=float, default=0.0001)
        # parser.add_argument('--reg_subtask', type=float, default=100)
        # parser.add_argument('--clip_grad', type=float, default=0.0)
        parser.add_argument('--num_epochs', type=int, default=30)
        # parser.add_argument('--num_heads', type=int, default=4)
        # parser.add_argument('--num_layers', type=int, default=2)
        # parser.add_argument('--hidden_dim', type=int, default=128)
        # parser.add_argument('--sparsity', type=int, default=40)
        # parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument("--rp", type=int, default=1)
        # parser.add_argument("--num_classes", type=int, default=1)

        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument("--weights", default=None, help='pre-trained ESTA weights')

        return parser


    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name, str(args.rp))
    print(args.exp_name)

    training_function(args)

    # if args.weights is not None:
    #     tt_function(args)  #test function