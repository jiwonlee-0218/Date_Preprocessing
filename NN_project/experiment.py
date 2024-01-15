import torch
import torch.nn as nn
import os
import numpy as np
from model import TAE, Prediction_Model
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import argparse
from dataset import dataloader
import random
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics





def writelog(file, line):
    file.write(line + '\n')
    print(line)




def pretrained_autoencoder(args):
    """
    function for the autoencoder pretraining
    """


    print("Pretraining autoencoder... \n")

    # set seed and device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(args.seed)
    print(device)



    # dataloader
    train_data, train_label, test_data, test_label, all_subtask_dict, all_task_list = dataloader(args)



    skf = StratifiedKFold(args.k_fold, shuffle=True, random_state=0)

    for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_label)):
        print("--------------------- fold: {} -----------------------".format(fold))
        os.makedirs(os.path.join(args.targetdir, 'pretrain_model_weights', str(fold)), exist_ok=True)



        x_train, y_train = train_data[train_index], train_label[train_index]
        x_val, y_val = train_data[val_index], train_label[val_index]

        train_dataset = list(zip(torch.tensor(x_train), torch.tensor(y_train)))
        valid_dataset = list(zip(torch.tensor(x_val), torch.tensor(y_val)))
        train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.minibatch_size, shuffle=True)

        tae = TAE(args)
        tae = tae.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)



        best_score = 0.0
        for epoch in range(args.pretrain_num_epochs):

            loss_accumulate = 0.0
            for batch_idx, (inputs, _) in enumerate(tqdm(train_loader)):
                tae.train()
                inputs = inputs.to(device)

                optimizer.zero_grad()  # 기울기에 대한 정보 초기화
                features, x_reconstr = tae(inputs)
                loss_mse = criterion(inputs, x_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
                loss_mse.backward()  # 기울기 구함


                optimizer.step()  # 최적화 진행

                loss_accumulate += loss_mse.detach().cpu().numpy()



            train_loss = loss_accumulate / (batch_idx + 1)
            print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch, train_loss))


            # validation
            tae.eval()
            with torch.no_grad():
                val_loss_accumulate = 0.0
                for j, (val_x, val_y) in enumerate(valid_loader):
                    val_x = val_x.to(device)

                    v_features, val_reconstr = tae(val_x)
                    val_loss = criterion(val_x, val_reconstr)

                    val_loss_accumulate += val_loss.detach().cpu().numpy()

                validation_loss = val_loss_accumulate / (j + 1)
                print("Pretraining autoencoder validation loss for epoch {} is : {}".format(epoch, validation_loss))


            if epoch == 0:
                min_val_loss = validation_loss

            if validation_loss < min_val_loss:
                torch.save({
                    'model_state_dict': tae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': validation_loss},
                    os.path.join(args.targetdir, 'pretrain_model_weights', str(fold), 'checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, validation_loss)))
                min_val_loss = validation_loss
                print("save weights !!")

    print("Ending pretraining autoencoder. \n")




def downstream_task(args):


    print("Downstream_task... for prediction\n")

    # set seed and device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(args.seed)
    print(device)


    # dataloader
    train_data, train_label, test_data, test_label, all_subtask_dict, all_task_list = dataloader(args)


    skf = StratifiedKFold(args.k_fold, shuffle=True, random_state=0) ##########위랑 똑같이 나눠지는지 확인해보기


    for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_label)):
        print("--------------------- fold: {} -----------------------".format(fold))
        os.makedirs(os.path.join(args.targetdir, 'prediction_model_weights', str(fold)), exist_ok=True)
        os.makedirs(os.path.join(args.targetdir, 'prediction_model_result', str(fold)), exist_ok=True)


        x_train, y_train = train_data[train_index], train_label[train_index]
        x_val, y_val = train_data[val_index], train_label[val_index]

        train_dataset = list(zip(torch.tensor(x_train), torch.tensor(y_train)))
        valid_dataset = list(zip(torch.tensor(x_val), torch.tensor(y_val)))
        train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.minibatch_size, shuffle=True)



        tae = TAE(args)
        path = os.path.join(args.targetdir, 'pretrain_model_weights', str(fold))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        checkpoint = torch.load(full_path)
        tae.load_state_dict(checkpoint['model_state_dict'])
        tae = tae.to(device)



        hidden_dim = args.hidden_dim
        num_classes = args.num_classes
        dropout = args.dropout
        inference_model = Prediction_Model(args, hidden_dim, dropout, num_classes)
        inference_model.to(device)


        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(inference_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)



        best_score = 0.0
        for epoch in range(args.inference_num_epochs):


            tr_predict_all = np.array([], dtype=int)
            tr_labels_all = np.array([], dtype=int)
            loss_accumulate = 0.0
            inference_model.train()


            for batch_idx, (inputs, _) in enumerate(tqdm(train_loader)):
                inputs = inputs.to(device)

                optimizer.zero_grad()  # 기울기에 대한 정보 초기화

                features = tae.tae_encoder(inputs)
                out = inference_model(features)

                subtask_list = []
                for i in _:
                    for task_idx, _task in enumerate(all_task_list):
                        if i == task_idx:
                            subtask_list.append(all_subtask_dict[_task])
                subtask_label = torch.tensor(np.array(subtask_list)).flatten()


                loss_subtask = criterion(out, subtask_label.cuda())   # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
                loss_subtask.backward()  # 기울기 구함
                optimizer.step()  # 최적화 진행

                loss_accumulate += loss_subtask.detach().cpu().numpy()
                pred = out.argmax(1).cpu().numpy()
                tr_predict_all = np.append(tr_predict_all, pred)
                tr_labels_all = np.append(tr_labels_all, subtask_label.cpu().numpy())


            train_loss = loss_accumulate / (batch_idx + 1)
            train_acc = metrics.accuracy_score(tr_labels_all, tr_predict_all)
            print("Prediction loss: {}, acc: {} for epoch {}".format(train_loss, train_acc, epoch))


            # validation
            inference_model.eval()
            with torch.no_grad():


                val_predict_all = np.array([], dtype=int)
                val_labels_all = np.array([], dtype=int)
                val_loss_accumulate = 0.0
                for j, (val_x, val_y) in enumerate(valid_loader):
                    val_x = val_x.to(device)


                    v_features = tae.tae_encoder(val_x)
                    v_out = inference_model(v_features)


                    val_subtask_list = []
                    for i in val_y:
                        for task_idx, _task in enumerate(all_task_list):
                            if i == task_idx:
                                val_subtask_list.append(all_subtask_dict[_task])
                    val_subtask_label = torch.tensor(np.array(val_subtask_list)).flatten()


                    val_loss_subtask = criterion(v_out, val_subtask_label.cuda())

                    val_loss_accumulate += val_loss_subtask.detach().cpu().numpy()
                    val_pred = v_out.argmax(1).cpu().numpy()
                    val_predict_all = np.append(val_predict_all, val_pred)
                    val_labels_all = np.append(val_labels_all, val_subtask_label.cpu().numpy())


                validation_loss = val_loss_accumulate / (j + 1)
                val_acc = metrics.accuracy_score(val_labels_all, val_predict_all)
                print("Prediction val loss: {}, val acc: {} for epoch {}".format(validation_loss, val_acc, epoch))




                if best_score < val_acc:
                    best_score = val_acc
                    best_epoch = epoch

                    torch.save({
                        'fold': fold,
                        'model_state_dict': inference_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': train_loss},
                        os.path.join(args.targetdir, 'prediction_model_weights', str(fold), 'checkpoint_epoch_{}.pt'.format(epoch)))

                    print("save weights !!")






        # test
        tae = TAE(args)
        path = os.path.join(args.targetdir, 'pretrain_model_weights', str(fold))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        tae.load_state_dict(checkpoint['model_state_dict'])
        tae = tae.to(device)


        path = os.path.join(args.targetdir, 'prediction_model_weights', str(fold))
        full_path = sorted(glob.glob(path + '/*'), key=os.path.getctime)[-1]
        print(full_path)
        checkpoint = torch.load(full_path)
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        num_classes = args.num_classes
        inference_model = Prediction_Model(args, hidden_dim, dropout, num_classes)
        inference_model.to(device)
        inference_model.load_state_dict(checkpoint['model_state_dict'])
        inference_model.to(device)


        inference_model.eval()
        with torch.no_grad():
            tst_predict_all = np.array([], dtype=int)
            tst_labels_all = np.array([], dtype=int)

            test_dataset = list(zip(torch.tensor(test_data), torch.tensor(test_label)))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


            for batch_idx, (x_test, y_test) in enumerate(test_loader):
                x_test = x_test.to(device)
                features = tae.tae_encoder(x_test)
                tst_out = inference_model(features)

                for task_idx, _task in enumerate(all_task_list):
                    if y_test == task_idx:
                        test_subtask_label = torch.Tensor(all_subtask_dict[_task])


                test_pred = tst_out.argmax(1).cpu().numpy()
                tst_predict_all = np.append(tst_predict_all, test_pred)
                tst_labels_all = np.append(tst_labels_all, test_subtask_label.cpu().numpy())

        test_acc = metrics.accuracy_score(tst_labels_all, tst_predict_all)
        print()
        print("Prediction test acc: {}, fold: {}".format(test_acc, fold))
        f = open(os.path.join(args.targetdir, 'prediction_model_result', str(fold), 'test_acc.log'), 'a')
        writelog(f, 'test_acc: %.5f' % (test_acc))
        f.close()

    print("Ending Prediction. \n")



if __name__ == "__main__":

    def get_arguments():
        parser = argparse.ArgumentParser(description='Transformer-based Autoencoder')
        parser.add_argument('-s', '--seed', type=int, default=24)
        parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
        parser.add_argument('-n', '--exp_name', type=str, default='experiment_1_sangmin')
        parser.add_argument('-k', '--k_fold', type=int, default=5)
        parser.add_argument('-b', '--minibatch_size', type=int, default=16)
        parser.add_argument('-dt', '--targetdir', type=str, default='/DataCommon2/jwlee/NN_project/result')



        # data args
        parser.add_argument('--input_size', type=int, default=116)
        parser.add_argument('--input_length', type=int, default=176)



        # training args
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument("--weight_decay", type=float, default=5e-6, help="Weight decay for Adam optimizer", )
        parser.add_argument('--pretrain_num_epochs', type=int, default=100)
        parser.add_argument('--inference_num_epochs', type=int, default=100)


        # model args
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument("--num_classes", type=int, default=25)


        parser.add_argument("--weights", default=True, help='pretrained autoencoder weights')

        return parser

    parser = get_arguments()
    args = parser.parse_args()
    args.targetdir = os.path.join(args.targetdir, args.exp_name)


    train_data, train_label, test_data, test_label, all_subtask_dict, all_task_list = dataloader(args)



    # pretraining
    # if args.weights is None and args.pretrain_num_epochs > 0:
    # pretrained_autoencoder(args)


    # downstream task
    if args.weights is not None:
        downstream_task(args)

