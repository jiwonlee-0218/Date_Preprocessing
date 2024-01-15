import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import backend as K
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sklearn.metrics as metrics
from tensorflow.keras.optimizers import Adam, Adadelta, Nadam
import os
from bert2.transformer import ProjectionLayer
from bert2.attention import Attention_SV
from params_flow.activations import gelu
from utils import MyAttention, BertEmbeddingsLayer
from sklearn.metrics import accuracy_score, confusion_matrix


os.environ['PYTHONHASHSEED']=str(42)
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)



def slice(x, h1, h2):
 """ Define a tensor slice function
 """
 return x[:,h1:h2, :]

def slice2parts(x, h1, h2, h3):
 """ Define a tensor slice function
 """
 return x[:,h1:h2, :], x[:,h2:h3,:]
def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))

def data_augment(data, label, pheno, crops):
    augment_label = []
    augment_data = []
    augmented_pheno = []

    for i in range(len(data)):
        max = data[i].shape[0]
        if max >= 100:

            range_list = range(90 + 1, int(max))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                # r = random.randint(90,int(max))
                augment_data.append(data[i][r - 90:r])
                augment_label.append(label[i])
                augmented_pheno.append(pheno[i])

    return np.array(augment_data), np.array(augment_label), np.array(augmented_pheno)


def STCAL(max_seq_len, embedding_size):
    time_input = keras.layers.Input(shape=(max_seq_len, embedding_size), dtype='float32', name="time_input")
    ROI_input = keras.layers.Permute((2, 1), name='ROI_input')(time_input)
    pheno_input = keras.layers.Input((4,), name='pheno_input')

    # position embedding
    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=embedding_size,
                                                  vocab_size=1)(time_input)
    ROI_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=max_seq_len, vocab_size=1)(
        ROI_input)

    time_input_spliting1 = keras.layers.Lambda(slice, arguments={'h1': 0, 'h2': 30})(time_position_embedding)
    time_input_spliting2 = keras.layers.Lambda(slice, arguments={'h1': 15, 'h2': 45})(time_position_embedding)
    time_input_spliting3 = keras.layers.Lambda(slice, arguments={'h1': 30, 'h2': 60})(time_position_embedding)
    time_input_spliting4 = keras.layers.Lambda(slice, arguments={'h1': 45, 'h2': 75})(time_position_embedding)
    time_input_spliting5 = keras.layers.Lambda(slice, arguments={'h1': 60, 'h2': 90})(time_position_embedding)

    time_valueS1_l1, time_attentionS1_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting1)
    time_valueS2_l1, time_attentionS2_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting2)
    time_valueS3_l1, time_attentionS3_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting3)
    time_valueS4_l1, time_attentionS4_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting4)
    time_valueS5_l1, time_attentionS5_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_input_spliting5)

    time_attention_probS1_l1 = keras.layers.Softmax()(time_attentionS1_l1)
    time_attention_probS1_l1 = keras.layers.Dropout(0.1)(time_attention_probS1_l1)
    time_attention_scoreS1_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS1_l1, time_attention_probS1_l1])
    projection_timeS1_l1 = ProjectionLayer(hidden_size=embedding_size)(
        [time_attention_scoreS1_l1, time_input_spliting1])
    intermediate_timeS1_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_timeS1_l1)
    bert_out_timeS1_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS1_l1, projection_timeS1_l1])

    time_attention_probS2_l1 = keras.layers.Softmax()(time_attentionS2_l1)
    time_attention_probS2_l1 = keras.layers.Dropout(0.1)(time_attention_probS2_l1)
    time_attention_scoreS2_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS2_l1, time_attention_probS2_l1])
    projection_timeS2_l1 = ProjectionLayer(hidden_size=embedding_size)(
        [time_attention_scoreS2_l1, time_input_spliting2])
    intermediate_timeS2_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_timeS2_l1)
    bert_out_timeS2_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS2_l1, projection_timeS2_l1])

    time_attention_probS3_l1 = keras.layers.Softmax()(time_attentionS3_l1)
    time_attention_probS3_l1 = keras.layers.Dropout(0.1)(time_attention_probS3_l1)
    time_attention_scoreS3_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS3_l1, time_attention_probS3_l1])
    projection_timeS3_l1 = ProjectionLayer(hidden_size=embedding_size)(
        [time_attention_scoreS3_l1, time_input_spliting3])
    intermediate_timeS3_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_timeS3_l1)
    bert_out_timeS3_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS3_l1, projection_timeS3_l1])

    time_attention_probS4_l1 = keras.layers.Softmax()(time_attentionS4_l1)
    time_attention_probS4_l1 = keras.layers.Dropout(0.1)(time_attention_probS4_l1)
    time_attention_scoreS4_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS4_l1, time_attention_probS4_l1])
    projection_timeS4_l1 = ProjectionLayer(hidden_size=embedding_size)(
        [time_attention_scoreS4_l1, time_input_spliting4])
    intermediate_timeS4_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_timeS4_l1)
    bert_out_timeS4_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS4_l1, projection_timeS4_l1])

    time_attention_probS5_l1 = keras.layers.Softmax()(time_attentionS5_l1)
    time_attention_probS5_l1 = keras.layers.Dropout(0.1)(time_attention_probS5_l1)
    time_attention_scoreS5_l1 = MyAttention(num_heads=10, size_per_head=19)([time_valueS5_l1, time_attention_probS5_l1])
    projection_timeS5_l1 = ProjectionLayer(hidden_size=embedding_size)(
        [time_attention_scoreS5_l1, time_input_spliting5])
    intermediate_timeS5_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_timeS5_l1)
    bert_out_timeS5_l1 = ProjectionLayer(hidden_size=embedding_size)([intermediate_timeS5_l1, projection_timeS5_l1])

    bert_out_timeS1_l1_part1, bert_out_timeS1_l1_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 15, 'h3': 30})(
        bert_out_timeS1_l1)
    bert_out_timeS2_l1_part1, bert_out_timeS2_l1_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 15, 'h3': 30})(
        bert_out_timeS2_l1)
    bert_out_timeS3_l1_part1, bert_out_timeS3_l1_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 15, 'h3': 30})(
        bert_out_timeS3_l1)
    bert_out_timeS4_l1_part1, bert_out_timeS4_l1_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 15, 'h3': 30})(
        bert_out_timeS4_l1)
    bert_out_timeS5_l1_part1, bert_out_timeS5_l1_part2 = keras.layers.Lambda(slice2parts,
                                                                             arguments={'h1': 0, 'h2': 15, 'h3': 30})(
        bert_out_timeS5_l1)

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

    bert_out_time_l1 = keras.layers.LayerNormalization()(bert_out_time_l1)

    ROI_value_l1, ROI_attention_l1 = Attention_SV(num_heads=10, size_per_head=9)(ROI_position_embedding)
    ROI_attention_prob_l1 = keras.layers.Softmax()(ROI_attention_l1)
    ROI_attention_prob_l1 = keras.layers.Dropout(0.1)(ROI_attention_prob_l1)
    ROI_attention_score_l1 = MyAttention(num_heads=10, size_per_head=9)([ROI_value_l1, ROI_attention_prob_l1])
    projection_ROI_l1 = ProjectionLayer(hidden_size=max_seq_len)([ROI_attention_score_l1, ROI_position_embedding])
    intermediate_ROI_l1 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_ROI_l1)
    bert_out_ROI_l1 = ProjectionLayer(hidden_size=max_seq_len)([intermediate_ROI_l1, projection_ROI_l1])

    # -----------------------------------------------------------------------------------------------------------------
    # layer two resbert
    time_value_l2, time_attention_l2 = Attention_SV(num_heads=10, size_per_head=19)(bert_out_time_l1)
    ROI_value_l2, ROI_attention_l2 = Attention_SV(num_heads=10, size_per_head=9)(bert_out_ROI_l1)

    time_attention_prob_l2 = keras.layers.Softmax(name='attention_score')(time_attention_l2)
    time_attention_prob_l2 = keras.layers.Dropout(0.1)(time_attention_prob_l2)
    time_attention_score_l2 = MyAttention(num_heads=10, size_per_head=19)([time_value_l2, time_attention_prob_l2])
    projection_time_l2 = ProjectionLayer(hidden_size=embedding_size)([time_attention_score_l2, bert_out_time_l1])
    intermediate_time_l2 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_time_l2)
    bert_out_time_l2 = ProjectionLayer(hidden_size=embedding_size)([intermediate_time_l2, projection_time_l2])

    ROI_attention_prob_l2 = keras.layers.Softmax()(ROI_attention_l2)
    ROI_attention_prob_l2 = keras.layers.Dropout(0.1)(ROI_attention_prob_l2)
    ROI_attention_score_l2 = MyAttention(num_heads=10, size_per_head=9)([ROI_value_l2, ROI_attention_prob_l2])
    projection_ROI_l2 = ProjectionLayer(hidden_size=max_seq_len)([ROI_attention_score_l2, bert_out_ROI_l1])
    intermediate_ROI_l2 = keras.layers.Dense(units=4 * embedding_size, activation=gelu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        projection_ROI_l2)
    bert_out_ROI_l2 = ProjectionLayer(hidden_size=max_seq_len)([intermediate_ROI_l2, projection_ROI_l2])

    # -------------------------------------------------------------------------------
    # layer three co-attention and resbert

    time_value_l3 = Attention_SV(num_heads=10, size_per_head=19, score=False)(bert_out_time_l2)
    ROI_value_l3, ROI_attention_l3 = Attention_SV(num_heads=10, size_per_head=9)(bert_out_ROI_l2)

    ROI_attention_reshape = keras.layers.Reshape((-1, embedding_size, embedding_size))(ROI_attention_l3)
    mean_ROI_score = keras.layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(ROI_attention_reshape)
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

    bert_out_time = keras.layers.Dropout(0.5)(bert_out_time)
    bert_out_ROI = keras.layers.Dropout(0.5)(bert_out_ROI)

    bert_out_time = keras.layers.GlobalAveragePooling1D()(bert_out_time)
    bert_out_ROI = keras.layers.GlobalAveragePooling1D()(bert_out_ROI)

    bert_out_concate = keras.layers.Concatenate()([bert_out_time, bert_out_ROI])
    merge = keras.layers.Concatenate()([bert_out_concate, pheno_input])
    dense = keras.layers.Dense(64, activation='linear')(merge)
    dense = keras.layers.Dropout(0.5)(dense)
    dense02 = keras.layers.Dense(10, activation='linear')(dense)
    output = keras.layers.Dense(1, activation='sigmoid')(dense02)

    model_ADHD = keras.Model(inputs=[time_input, pheno_input], outputs=output)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ADHD.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model_ADHD


if __name__ == '__main__':
    # Preparing data
    crops = 10

    ASD_train_feats = np.load('Data/ADHD_IS/ADHD_IS_data_train.npy',allow_pickle=True)
    ASD_train_labels = np.load('Data/ADHD_IS/ADHD_IS_labels_train.npy')
    ASD_train_phenos = np.load('Data/ADHD_IS/ADHD_IS_phenos_train.npy',allow_pickle=True)

    ASD_test_feats = np.load('Data/ADHD_IS/ADHD_IS_data_test.npy',allow_pickle=True)
    ASD_test_labels = np.load('Data/ADHD_IS/ADHD_IS_labels_test.npy')
    ASD_test_phenos = np.load('Data/ADHD_IS/ADHD_IS_phenos_test.npy',allow_pickle=True)



    CV_count = 10
    ADHD_bert_mean = []
    ADHD_sub_bert_mean = []
    ADHD_sub2_bert_mean = []

    ADHD_sensitive_mean = []
    ADHD_specificity_mean = []
    ADHD_auc_mean = []

    for i_count in range(CV_count):
        K.clear_session()

        ADHD_bert = []
        ADHD_sub_bert = []
        ADHD_sub_bert2 = []

        ADHD_sensitive = []
        ADHD_specificity = []
        ADHD_auc = []

        # training data
        x_train, y_train, x_train_pheno = data_augment(ASD_train_feats,ASD_train_labels,ASD_train_phenos,crops)
        x_test, y_test, x_test_pheno= data_augment(ASD_test_feats,ASD_test_labels,ASD_test_phenos,crops)

        print(x_train.shape)

        # shuffle
        index = [i for i in range(x_train.shape[0])]
        random.shuffle(index)

        x_train = x_train[index]
        x_train_pheno = x_train_pheno[index]
        y_train = y_train[index]

        # np.save('model_interpretation_labels',y_train)
        # np.save('shuffle_index',index)

        print(x_train.shape)
        print(x_train_pheno.shape)
        print(y_train.shape)

        x_test_pheno = x_test_pheno.reshape(-1,4)

        max_seq_len = x_train.shape[1]
        embedding_size = x_train.shape[2]

        model = STCAL(max_seq_len, embedding_size)
        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                        patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2)

        model.fit(x=[x_train, x_train_pheno], y=y_train, epochs=30, batch_size=128,
                    validation_split=0.05)

        ADHD_predict = model.predict(x=[x_test, x_test_pheno])
        ADHD_pred = [1 if prob >= 0.5 else 0 for prob in ADHD_predict]

        pred = []
        pred2 = []
        true_labels = []

        true_score = []
        pred_score = []
        for i in range(len(ASD_test_feats)):
            ADHD_subject = ADHD_pred[i * crops:i * crops + crops]
            if np.count_nonzero(ADHD_subject) >= crops/2:
                pred.append(1)
            else:
                pred.append(0)

            ADHD_subject_score = ADHD_predict[i * crops:i * crops + crops]
            pred_score.append(np.sum(ADHD_subject_score) / crops)
            if np.sum(ADHD_subject_score) / crops >= 0.5:
                pred2.append(1)
            else:
                pred2.append(0)

            ADHD_true_subject = y_test[i * crops:i * crops + crops]
            true_score.append(np.sum(ADHD_true_subject) / crops)
            if np.count_nonzero(ADHD_true_subject) >= crops/2:
                true_labels.append(1)
            else:
                true_labels.append(0)


        [[TN, FP], [FN, TP]] = confusion_matrix(true_labels, pred2).astype(float)
        specificity = TN / (FP + TN)
        sensivity = recall = TP / (TP + FN)
        auc = metrics.roc_auc_score(true_score, pred_score)

        ADHD_bert.append(metrics.accuracy_score(y_test, ADHD_pred))
        ADHD_sub_bert.append(metrics.accuracy_score(true_labels, pred))
        ADHD_sub_bert2.append(metrics.accuracy_score(true_labels, pred2))
        ADHD_auc.append(auc)
        ADHD_specificity.append(specificity)
        ADHD_sensitive.append(sensivity)

        print("ADHD accuracy: " + str(metrics.accuracy_score(y_test, ADHD_pred)))
        print("ADHD sub accuracy: " + str(metrics.accuracy_score(true_labels, pred)))
        print("ADHD sub2 accuracy: " + str(metrics.accuracy_score(true_labels, pred2)))
        print("ADHD sub2 auc: " + str(auc))
        print("ADHD sub2 specificity: " + str(specificity))
        print("ADHD sub2 sensivity: " + str(sensivity))

        print(str(i_count) + " th CV done!")
        ADHD_bert_mean.append(np.mean(ADHD_bert))
        ADHD_sub_bert_mean.append(np.mean(ADHD_sub_bert))
        ADHD_sub2_bert_mean.append(np.mean(ADHD_sub_bert2))
        ADHD_auc_mean.append(np.mean(ADHD_auc))
        ADHD_specificity_mean.append(np.mean(ADHD_specificity))
        ADHD_sensitive_mean.append(np.mean(ADHD_sensitive))

    print("ADHD_bert_mean:" + str(ADHD_bert_mean))
    print("ADHD_sub_bert_mean:" + str(ADHD_sub_bert_mean))
    print("ADHD_sub2 mean: " + str(ADHD_sub2_bert_mean))
    print("ADHD_sub2 auc_mean: " + str(ADHD_auc_mean))
    print("ADHD_sub2 specificity_mean: " + str(ADHD_specificity_mean))
    print("ADHD_sub2 sensitive_meann: " + str(ADHD_sensitive_mean))
