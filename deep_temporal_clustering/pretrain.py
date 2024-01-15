import torch
import datetime
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import glob
from config import get_arguments
from pretrain_models import TAE
from load_data import get_loader
import numpy as np
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def writelog(file, line):
    file.write(line + '\n')
    print(line)




def pretrain_autoencoder(args, verbose=True):
    """
    function for the autoencoder pretraining
    """
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

            optimizer.zero_grad() # 기울기에 대한 정보 초기화
            features, x_reconstr = tae(inputs)
            loss_mse = loss_ae(inputs, x_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
            loss_mse.backward() #기울기 구함

            optimizer.step() #최적화 진행

            all_loss += loss_mse.item()

        train_loss = all_loss / (batch_idx + 1)
        writer.add_scalar("training loss", train_loss, epoch+1)
        if verbose:
            print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch+1, train_loss))


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

            writer.add_scalar("validation loss", validation_loss, epoch + 1)
            print("val_loss for epoch {} is : {}".format(epoch+1, validation_loss ))

        if epoch == 0:
            min_val_loss = validation_loss

        if validation_loss < min_val_loss:
            torch.save({
                'model_state_dict': tae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': validation_loss
            }, os.path.join(directory, 'models_weights')+'/checkpoint_epoch_{}_loss_{:.4f}.pt'.format(epoch+1, validation_loss))

            min_val_loss = validation_loss
            print("save weights !!")


    writer.close()
    print("Ending pretraining autoencoder. \n")














if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()
    args.path_data = args.path_data.format(args.dataset_name)

    # args.path_weights_main = os.path.join(path_weights, "full_model_weigths.pth")

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # data load
    data = np.load('/DataCommon/jwlee/MOTOR_LR/hcp_motor.npz')
    samples = data['tfMRI_MOTOR_LR']
    samples = samples[:1080]  # (1080, 284, 116)

    mm = MinMaxScaler()
    results = []
    for ss in range(1080):
        results.append(mm.fit_transform(samples[ss]))

    sample = np.array(results)

    label = data['label_MOTOR_LR']
    label = label[:1080]  # (1080, 284)

    X_train, X_test, y_train, y_test = train_test_split(sample, label, random_state=42, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_val), torch.FloatTensor(y_val)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size) ###
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size) ###

    args.serie_size = X_train.shape[1]

    # date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name,
                             'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(
                                 args.lr_ae) + '_wdcay_' + str(args.weight_decay) )








    if args.ae_weights is None and args.epochs_ae > 0: ########### pretrain
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

        pretrain_autoencoder(args)

    if args.ae_weights is not None:
        if not os.path.exists(os.path.join(directory, 'models_pictures')):
            os.makedirs(os.path.join(directory, 'models_pictures'))

        path = os.path.join(directory, args.ae_weights)
        full_path = sorted(glob.glob(path + '*.pt'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")



        ## define TAE architecture
        tae = TAE(args)
        tae = tae.to(args.device)
        print(tae)
        loss_ae = nn.MSELoss()


        checkpoint = torch.load(full_path)
        tae.load_state_dict(checkpoint['model_state_dict'])

        tae.eval()
        with torch.no_grad():
            X_test = X_test.type(torch.FloatTensor).to(args.device)
            repre, output = tae(X_test) # recon: torch.Size([108, 284, 10]), output: torch.Size([108, 284, 116])
            loss = loss_ae(X_test[1], output[1])
            print("sub1's loss: ", loss)

        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(X_test[1].cpu().T)
        plt.title("Original_X")
        plt.colorbar()
        # plt.show()
        plt.savefig( os.path.join(directory, 'models_pictures/') + 'orig.png')

        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(output[1].cpu().detach().T)
        plt.title("Reconstruction")
        # plt.colorbar()
        # plt.clim(-6, 6)
        # plt.show()
        plt.savefig(  os.path.join(directory, 'models_pictures/') + 'recon.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(repre[1].cpu().detach().T)
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'models_pictures/') + 'represented.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(repre[1][:150, :].cpu().detach().T)
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'models_pictures/') + 'represented_shorts.png')









