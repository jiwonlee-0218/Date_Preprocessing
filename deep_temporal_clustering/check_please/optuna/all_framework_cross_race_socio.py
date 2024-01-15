import numpy as np
import pandas as pd
import optuna
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score


def normalization(data, norm_type):

    if norm_type in ['mm_f', 'st_f']:
        if norm_type == 'mm_f':
            scaler = MinMaxScaler()
        elif norm_type == 'st_f':
            scaler = StandardScaler()
        scaler.fit(data)
        norm_data = scaler.transform(data)

    elif norm_type in ['mm_s', 'st_s']:
        norm_data = (data - np.expand_dims(data.min(axis=1), -1)) / np.expand_dims((data.max(axis=1) - data.min(axis=1)), -1)

    elif norm_type in ['mm_a', 'st_a']:
        norm_data = (data - data.min()) / (data.max() - data.min())

    else:
        raise ValueError

    return norm_data


def normalization2(train, valid, test, norm_type):

    if norm_type == 'feature':
        scaler = StandardScaler()
        scaler.fit(train) # 교차검증을 위해 train-test로 분리하였을 경우 전체 데이터가 아닌 훈련 데이터에 대해서만 fit()을 적용해야한다.
        norm_train = scaler.transform(train)
        norm_valid = scaler.transform(valid)
        norm_test = scaler.transform(test)

    elif norm_type == 'all':
        norm_train = (train - train.min()) / (train.max() - train.min())
        norm_valid = (valid - train.min()) / (train.max() - train.min())
        norm_test = (test - train.min()) / (train.max() - train.min())

    else:
        raise ValueError

    return norm_train, norm_valid, norm_test



def trnvaltst(data, seed, fold, race, socio):
    # Load index
    # idxg_path = main_path + 'race_threshold25_cv/rp%s_r%d_god_%s.npz' % (seed, race, socio)
    # idxb_path = main_path + 'race_threshold25_cv/rp%s_r%d_bad_%s.npz' % (seed, race, socio)
    idxg_path = main_path + 'race_threshold10_cv/rp%s_r%d_god_%s_%s.npz' % (seed, race, socio[0], socio[1])
    idxb_path = main_path + 'race_threshold10_cv/rp%s_r%d_bad_%s_%s.npz' % (seed, race, socio[0], socio[1])

    idxg_1 = np.load(idxg_path, allow_pickle=True)
    idxb_1 = np.load(idxb_path, allow_pickle=True)

    # Balancing
    trnidx_g, trnidx_b = idxg_1['trn_idx'][fold], idxb_1['trn_idx'][fold]
    if trnidx_g.shape[0] > trnidx_b.shape[0]:
        trnidx_g = trnidx_g[:trnidx_b.shape[0]]
    elif trnidx_g.shape[0] < trnidx_b.shape[0]:
        trnidx_b = trnidx_b[:trnidx_g.shape[0]]
    validx_g, validx_b = idxg_1['val_idx'][fold], idxb_1['val_idx'][fold]
    if validx_g.shape[0] > validx_b.shape[0]:
        validx_g = validx_g[:validx_b.shape[0]]
    elif validx_g.shape[0] < validx_b.shape[0]:
        validx_b = validx_b[:validx_g.shape[0]]
    tstidx_g, tstidx_b = idxg_1['tst_idx'][fold], idxb_1['tst_idx'][fold]
    if tstidx_g.shape[0] > tstidx_b.shape[0]:
        tstidx_g = tstidx_g[:tstidx_b.shape[0]]
    elif tstidx_g.shape[0] < tstidx_b.shape[0]:
        tstidx_b = tstidx_b[:tstidx_g.shape[0]]

    trnidx = np.concatenate([trnidx_g, trnidx_b])
    validx = np.concatenate([validx_g, validx_b])
    tstidx = np.concatenate([tstidx_g, tstidx_b])

    y_train = np.concatenate([-np.ones(trnidx_g.shape[0]), np.ones(trnidx_b.shape[0])])
    y_valid = np.concatenate([-np.ones(validx_g.shape[0]), np.ones(validx_b.shape[0])])
    y_test = np.concatenate([-np.ones(tstidx_g.shape[0]), np.ones(tstidx_b.shape[0])])

    X_train = data[trnidx]
    X_valid = data[validx]
    X_test = data[tstidx]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

class Objective:

    def __init__(self, args, data, socio):
        # Hold this implementation specific arguments as the fields of the class.
        self.args = args
        self.conn_data = data
        self.socio_set = socio

    def __call__(self, trial):

        args = self.args
        conn_data = self.conn_data
        socio_set = self.socio_set
        classifier_name = args.classifier
        X_train, X_valid, X_test, y_train, y_valid, y_test = trnvaltst(conn_data, args.seed, args.fold, args.race, socio_set)
        X_train, X_valid, _ = normalization2(X_train, X_valid, X_test, args.norm2)

        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-5, 1e3, log=True)
            svc_k = trial.suggest_categorical("kernel", ['linear'])
            classifier_obj = sklearn.svm.SVC(C=svc_c, kernel=svc_k, gamma='auto')
        elif classifier_name == "LinearSVC":
            # svc_c = trial.suggest_float("svc_c", 1e-5, 1e3, log=True)
            svc_c = trial.suggest_float("svc_c", 1e-2, 1e2, log=True)
            svc_l = trial.suggest_categorical("loss", ['squared_hinge']) # Hinge
            classifier_obj = sklearn.svm.LinearSVC(C=svc_c, loss=svc_l)
        elif classifier_name == "LogisticRegression":
            lr_c = trial.suggest_float("svc_c", 1e-5, 1e3, log=True)
            classifier_obj = sklearn.linear_model.LogisticRegression(C=lr_c, solver='saga', penalty='l1')
        elif classifier_name == "RandomForest":
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)

        classifier_obj.fit(X_train, y_train)
        acc = classifier_obj.score(X_valid, y_valid)
        return acc

def test(data, args, best_param, socio_set):
    X_train, X_valid, X_test, y_train, _, y_test = trnvaltst(data, args.seed, args.fold, args.race, socio_set)
    X_train, _, X_test = normalization2(X_train, X_valid, X_test, args.norm2)

    classifier_name = args.classifier
    if classifier_name == "svc_c":
        svc_c = best_param['svc_c']
        svc_k = best_param['kernel']
        classifier_obj = sklearn.svm.SVC(C=svc_c, kernel=svc_k, gamma='auto')
    elif classifier_name == "LinearSVC":
        svc_c = best_param['svc_c']
        svc_l = best_param['loss']
        classifier_obj = sklearn.svm.LinearSVC(C=svc_c, loss=svc_l, max_iter=3000)
    elif classifier_name == "LogisticRegression":
        lr_c = best_param['svc_c']
        classifier_obj = sklearn.linear_model.LogisticRegression(C=lr_c, solver='saga', penalty='l1')
    elif classifier_name == "RandomForest":
        rf_max_depth = best_param['rf_max_depth']
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)

    classifier_obj.fit(X_train, y_train)
    pred_t = classifier_obj.predict(X_test)
    # prob_t = classifier_obj.predict_proba(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_t).ravel()
    sen = tp / (tp + fn)
    spc = tn / (tn + fp)
    acc = accuracy_score(y_test, pred_t)
    prc = precision_score(y_test, pred_t)
    rec = recall_score(y_test, pred_t)
    f1s = f1_score(y_test, pred_t)
    if classifier_name =='RandomForest':
        auc = roc_auc_score(y_test, classifier_obj.predict_proba(X_test)[:, 1])
        auprc = average_precision_score(y_test, classifier_obj.predict_proba(X_test)[:, 1])
    else:
        auc = roc_auc_score(y_test, classifier_obj.decision_function(X_test))
        auprc = average_precision_score(y_test, classifier_obj.decision_function(X_test))

    # return print(args.socio, args.race, args.seed, best_param['svc_c'], auc, acc, sen, spc, prc, rec, f1s, auprc)
    return [args.socio, args.race, args.seed, best_param['svc_c'], round(auc, 4), round(acc, 4), round(sen, 4), round(spc, 4),
            round(prc, 4), round(rec, 4), round(f1s, 4),  round(auprc, 4)]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default='d', choices=['s', 'd', 'q'])
    parser.add_argument("--race", type=int, default=1, choices=[1, 2, 3], help='1: white, 2: black, 3: hispanic')
    parser.add_argument("--norm", type=str, default='mm_s', choices=['mm_f', 'mm_s', 'mm_a', 'st_f', 'st_s', 'st_a'],
                        help='mm: minmax, st: standard, f: feature-wise, s: subject-wise, a: all mean, std')
    parser.add_argument("--socio", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--norm2", type=str, default='feature', choices=['feature', 'all'])
    parser.add_argument("--classifier", type=str, default='LinearSVC', choices=["SVC", "LinearSVC", "RandomForest", "LogisticRegression"])
    parser.add_argument("--seed", type=int, default=1210)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--nroi", type=int, default=84)
    args = parser.parse_args()

    np.random.seed(args.seed)
    main_path = '/home/jwlee/HMM/deep_temporal_clustering/check_please/Data/'
    # Load connectivity dataset
    conn_path = main_path + 'con_aparc_count_nanx.csv'
    conn = pd.read_csv(conn_path).to_numpy()[:, 2:]
    # conn[conn < 500] = 0
    conn = normalization(conn, args.norm)

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if args.socio == 1 :
        socio_set = ['i_educ', 'i_income']
    elif args.socio == 2:
        socio_set = ['i_educ', 'n_edu_h']
    elif args.socio == 3:
        socio_set = ['i_educ', 'n_pov']
    elif args.socio == 4:
        socio_set = ['i_income', 'n_edu_h']
    elif args.socio == 5:
        socio_set = ['i_income', 'n_pov']
    elif args.socio == 6:
        socio_set = ['n_edu_h', 'n_pov']

    study = optuna.create_study(direction="maximize")
    study.optimize(Objective(args=args, data=conn, socio=socio_set), n_trials=50)
    result = test(conn, args, study.best_params, socio_set)
    print(result)
    with open("/home/jwlee/HMM/deep_temporal_clustering/check_please/Data/result_1/race%d_%s_%s.txt" % (args.race, socio_set[0], socio_set[1]), 'a') as f:
        for r in result:
            f.write(str(r) + ' ')
        f.write('\n')
        f.close()
