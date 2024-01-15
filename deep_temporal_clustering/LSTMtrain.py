import torch
import torch.nn as nn
import os
import numpy as np
from LSTMmain_config import get_arguments
from LSTMmodels import ClusterNet, TAE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import glob
import matplotlib.pyplot as plt


def writelog(file, line):
    file.write(line + '\n')
    print(line)



def pretrain_autoencoder(args, verbose=True, directory='.'):
    """
    function for the autoencoder pretraining
    """


    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(os.path.join(directory, 'models_logs')):
        os.makedirs(os.path.join(directory, 'models_logs'))

    if not os.path.exists(os.path.join(directory, 'models_weights')):
        os.makedirs(os.path.join(directory, 'models_weights'))

    # Text Logging
    f = open(os.path.join(directory, 'setting.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'GPU ID: %s' % (args.gpu_id))
    writelog(f, 'Dataset: %s' % (args.dataset_name))
    writelog(f, 'Dataset Path: %s' % (args.path_data))
    writelog(f, '----------------------')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Max Patience: %d (10 percent of the epoch size)' % args.max_patience)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    f.close()



    print("Pretraining autoencoder... \n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))

    ## define TAE architecture
    tae = TAE(args)
    tae = tae.to(args.device)
    print(tae)

    ## MSE loss
    loss_ae = nn.MSELoss()
    ## Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    for epoch in range(args.epochs_ae):

        # training
        tae.train()
        all_loss = 0


        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)

            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            features, x_reconstr = tae(inputs)
            loss_mse = loss_ae(inputs, x_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
            loss_mse.backward()  # 기울기 구함


            optimizer.step()  # 최적화 진행

            all_loss += loss_mse.item()

        train_loss = all_loss / (batch_idx + 1)

        writer.add_scalar("training loss", train_loss, epoch+1)
        if verbose:
            print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

        # validation
        tae.eval()
        with torch.no_grad():
            all_val_loss = 0
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)
                v_features, val_reconstr = tae(val_x)
                val_loss = loss_ae(val_x, val_reconstr)

                all_val_loss += val_loss.item()

            validation_loss = all_val_loss / (j + 1)

            writer.add_scalar("validation loss", validation_loss, epoch+1)
            print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))

        if epoch == 0:
            min_val_loss = validation_loss

        if validation_loss < min_val_loss:
            torch.save({
                'model_state_dict': tae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': validation_loss
            }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                        validation_loss))
            min_val_loss = validation_loss
            print("save weights !!")

    writer.close()
    print("Ending pretraining autoencoder. \n")




def train_ClusterNET(epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    mse_loss = 0
    total_train_loss = 0
    kl_loss = 0
    all_gt = []

    for batch_idx, (inputs, labels) in enumerate(train_dl):  # epoch 1에 모든 training_data batch만큼
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        all_gt.append(labels.cpu().detach()) # all_gt = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])


        optimizer_clu.zero_grad()

        z, x_reconstr, loss_KL, ARI = model(inputs, all_gt)  # ClusterNet의 forward

        loss_mse = loss_clu(inputs, x_reconstr)


        total_loss = loss_mse + loss_KL
        total_loss.backward()

        optimizer_clu.step()

        total_train_loss += total_loss.item()
        mse_loss += loss_mse.item()
        kl_loss += loss_KL



    return (total_train_loss / (batch_idx + 1)), (mse_loss / (batch_idx + 1)),  (kl_loss / (batch_idx + 1))


def training_function(args, verbose=True):

    """
    function for training the DTC network.
    """


    ## train clustering model

    print("Training full model ...")
    if not os.path.exists(os.path.join(directory, 'full_models')):
        os.makedirs(os.path.join(directory, 'full_models'))  ### for model save directory
    if not os.path.exists(os.path.join(directory, 'full_models_logs')):
        os.makedirs(os.path.join(directory, 'full_models_logs'))
    writer = SummaryWriter(log_dir=os.path.join(directory, 'full_models_logs'))

    f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'If the validation loss decreases compared to the previous epoch...')
    f.close()


    for epoch in range(args.max_epochs):
        train_loss, mse_loss, kl_loss = train_ClusterNET(epoch, args, verbose=verbose) # 1 epoch training
        print("For epoch ", epoch + 1, "Total Loss is : %.4f" % (train_loss), "MSE Loss is : %.4f" % ( mse_loss), "KL Loss is : %.4f" % (kl_loss))

        writer.add_scalar("training total loss", train_loss, (epoch+1))
        writer.add_scalar("training MSE loss", mse_loss, (epoch + 1))
        writer.add_scalar("training KL loss", kl_loss, (epoch + 1))


        model.eval()
        with torch.no_grad():
            all_val_loss = 0
            all_val_mse_loss = 0
            all_val_kl_loss = 0
            total_ari = 0  # epoch = 0
            all_gt = []
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)
                all_gt.append(val_y.cpu().detach())


                z, x_reconstr, loss_KL, ARI = model(val_x, all_gt)

                loss_mse = loss_clu(val_x, x_reconstr)
                total_loss = loss_mse + loss_KL


                all_val_mse_loss += loss_mse.item()
                all_val_loss += total_loss.item()
                all_val_kl_loss += loss_KL
                total_ari += ARI.item()





            all_val_mse_loss = all_val_mse_loss / (j+1)
            all_val_loss = all_val_loss / (j+1)
            all_val_kl_loss = all_val_kl_loss / (j+1)
            total_ari = total_ari / (j+1)
            print("For epoch ", epoch + 1, "val_Total Loss is : %.4f" % (all_val_loss), "val_MSE Loss is : %.4f" % (all_val_mse_loss), "val_KL Loss is : %.4f" % (all_val_kl_loss))




            writer.add_scalar("validation total loss", all_val_loss, (epoch + 1))
            writer.add_scalar("validation MSE loss", all_val_mse_loss, (epoch + 1))
            writer.add_scalar("validation KL loss", all_val_kl_loss, (epoch + 1))

        if epoch == 0:
            min_val_loss = all_val_loss
            print("ARI is : %.4f" % total_ari)

            f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
            writelog(f, 'Epoch: %d, ARI is: %.4f' % ((epoch + 1), total_ari))
            f.close()

            torch.save(
                model,
                os.path.join(directory, 'full_models') + '/checkpoint_epoch_{}_loss_{:.5f}_model.pt'.format(epoch + 1, all_val_loss)
            )


        if all_val_loss < min_val_loss:
            print( "ARI is : %.4f" % total_ari )

            f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
            writelog(f, 'Epoch: %d, ARI is: %.4f' % ((epoch+1), total_ari))
            f.close()




            torch.save(
                model, os.path.join(directory, 'full_models')+'/checkpoint_epoch_{}_loss_{:.5f}_model.pt'.format(epoch+1, all_val_loss)
            )
            min_val_loss = all_val_loss



    writer.close()
    print("Ending Training full model... \n")





def test_function(args):

    """
    function for training the DTC network.
    """


    ## test clustering model

    print("Test pretraining model ...")
    if not os.path.exists(os.path.join(directory, 'pretrain_models_pictures')):
        os.makedirs(os.path.join(directory, 'pretrain_models_pictures'))

    path = os.path.join(directory, args.ae_weights)
    full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
    full_path = full_path[-1]
    print("I got: " + full_path + " weights")

    test_tae = TAE(args)
    test_tae = test_tae.to(args.device)
    print(test_tae)
    loss_ae = nn.MSELoss()


    checkpoint = torch.load(full_path, map_location=args.device)
    test_tae.load_state_dict(checkpoint['model_state_dict'])



    test_tae.eval()
    with torch.no_grad():
        X_tests = X_test.type(torch.FloatTensor).to(args.device)

        features, x_reconstr = test_tae(X_tests)


        loss_mse = loss_ae(X_tests, x_reconstr)

        print("Test Pretrained autoencoder loss is : %.4f" % loss_mse)

    f = open(os.path.join(os.path.join(directory, 'pretrain_models_pictures/'), 'Pretrained autoencoder\'s test loss.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'Test Pretrained Autoencoder loss is : %.4f' % loss_mse)
    f.close()


    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(X_tests[1].cpu().T)
    plt.title("Original_X")
    plt.colorbar()
    plt.savefig(os.path.join(directory, 'pretrain_models_pictures/') + 'orig.png')

    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(x_reconstr[1].cpu().detach().T)
    plt.title("Reconstruction")
    plt.savefig(os.path.join(directory, 'pretrain_models_pictures/') + 'recon.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(features[1].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'pretrain_models_pictures/') + 'represented.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(features[1][:150, :].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'pretrain_models_pictures/') + 'represented_shorts.png')








if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()
    args.path_data = args.path_data.format(args.dataset_name)

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    data = np.load('/DataCommon/jwlee/MOTOR_LR/hcp_motor.npz')
    samples = data['tfMRI_MOTOR_LR']
    samples = samples[:1080]  # (1080, 284, 116)



    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1080):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)


    # train, validation, test
    label = data['label_MOTOR_LR']
    label = label[:1080]  # (1080, 284)

    X_train, X_test, y_train, y_test = train_test_split(sample, label, random_state=42, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)  ###
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)  ###

    args.serie_size = X_train.shape[1] ### timepoint


    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name,
                             'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(
                                 args.lr_ae) + '_wdcay_' + str(args.weight_decay))

    # number of clusters
    args.n_clusters = len(np.unique(y_train))

    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain

        pretrain_autoencoder(args, directory=directory)




    if args.ae_weights is not None and args.ae_models is None and args.autoencoder_test is not None:   ########### pretrain test using saved weights

        test_function(args)




    if args.ae_weights is not None and args.ae_models is None and args.autoencoder_test is None:  #### weight ok, but model is none



        path = os.path.join(directory, args.ae_weights)
        model = ClusterNet(args, path=path)  ## 모델 초기화 with the pretrained autoencoder model


        model = model.to(args.device)
        loss_clu = nn.MSELoss()
        optimizer_clu = torch.optim.Adam(model.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        training_function(args)





    if args.ae_weights is not None and args.ae_models is not None:  ### weight ok, model ok

        # Evaluate
        print("Test for DTC Model...")
        if not os.path.exists(os.path.join(directory, 'models_pictures')):
            os.makedirs(os.path.join(directory, 'models_pictures'))

        E_path = os.path.join(directory, args.ae_models)
        E_full_path = sorted(glob.glob(E_path + '*.pt'), key=os.path.getctime)
        E_full_path = E_full_path[-1]
        print("I got: " + E_full_path+ " embedding")

        dtc_model = torch.load(E_full_path)
        dtc_model = dtc_model.to(args.device)
        loss_clu = nn.MSELoss()


        all_gt = []
        dtc_model.eval()
        with torch.no_grad():
            X_test = X_test.type(torch.FloatTensor).to(args.device)
            all_gt.append(y_test.cpu().detach())

            z, x_reconstr, loss_KL, ARI = dtc_model(X_test, all_gt)

            loss_mse = loss_clu(X_test, x_reconstr)
            total_loss = loss_mse + loss_KL

            print("Total Loss is : %.4f" % (total_loss), "MSE Loss is : %.4f" % (loss_mse), "KL Loss is : %.4f" % (loss_KL), "ARI : %.4f" % (ARI))

        f = open(os.path.join(os.path.join(directory, 'models_pictures/'), 'DTC Model\'s test loss and ARI.log'), 'a')
        writelog(f, '======================')
        writelog(f, 'Model Name: %s' % args.model_name)
        writelog(f, '----------------------')
        writelog(f, 'Epoch: %d' % args.epochs_ae)
        writelog(f, 'Batch Size: %d' % args.batch_size)
        writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
        writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
        writelog(f, '======================')
        writelog(f, 'DTC Model\'s test total loss is : %.4f,  test MSE loss is : %.4f,  test KL loss is : %.4f' % (total_loss, loss_mse, loss_KL))
        writelog(f, 'DTC Model\'s ARI is : %.4f' % ARI)
        f.close()


        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(X_test[1].cpu().T)
        plt.title("Original_X")
        plt.colorbar()
        plt.savefig(os.path.join(directory, 'models_pictures/') + 'orig.png')

        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(x_reconstr[1].cpu().detach().T)
        plt.title("Reconstruction")
        plt.savefig(os.path.join(directory, 'models_pictures/') + 'recon.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(z[1].cpu().detach().T)
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'models_pictures/') + 'represented.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(z[1][:150, :].cpu().detach().T)
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'models_pictures/') + 'represented_shorts.png')