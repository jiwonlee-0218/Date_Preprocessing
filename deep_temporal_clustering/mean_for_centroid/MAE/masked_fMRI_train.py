from torch.utils.tensorboard import SummaryWriter



import os
from tqdm import tqdm
import torch



import masked_fMRI_vision_transformer as vits




def writelog(file, line):
    file.write(line + '\n')
    print(line)



def train_SiTv2(args, train_dl, valid_dl, directory):


    if not os.path.exists(directory):
        os.makedirs(directory)

        # Text Logging
        f = open(os.path.join(directory, 'setting.log'), 'a')
        writelog(f, '======================')
        writelog(f, 'GPU ID: %s' % (args.gpu_id))
        writelog(f, '----------------------')
        writelog(f, 'Model Name: %s' % args.model_name)
        writelog(f, '----------------------')
        writelog(f, 'Epoch: %d' % args.epochs)
        writelog(f, 'Batch Size: %d' % args.batch_size)
        writelog(f, 'Learning Rate: %s' % str(args.lr))
        writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
        writelog(f, '======================')
        f.close()

    if not os.path.exists(os.path.join(directory, 'models_logs')):
        os.makedirs(os.path.join(directory, 'models_logs'))

    if not os.path.exists(os.path.join(directory, 'models_weights')):
        os.makedirs(os.path.join(directory, 'models_weights'))





    ################ Create Transformer

    Full_SiT_model = vits.FullpiplineSiT(args)
    Full_SiT_model = Full_SiT_model.to(args.device)
    print(f"==> {args.model} model is created.")


    ################ optimization
    # Create Optimizer
    optimizer = torch.optim.Adam(Full_SiT_model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)

    ################ MSE loss
    loss_ae = torch.nn.MSELoss()




    ################ Training

    print("Pretraining autoencoder... with ViT\n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))
    # early_stopping = EarlyStopping(patience=15, verbose=True)

    for epoch in range(args.epochs):


        # Train an epoch
        Full_SiT_model.train()
        all_loss = 0


        for batch_idx, (inputs, _) in enumerate(train_dl):

            inputs = inputs.type(torch.FloatTensor).to(args.device)

            optimizer.zero_grad()
            represent, recons = Full_SiT_model(inputs)
            loss_mse = loss_ae(inputs, recons)
            loss_mse.backward()

            optimizer.step()


            all_loss += loss_mse.item()

        train_loss = all_loss / (batch_idx + 1)

        writer.add_scalar("training loss", train_loss, epoch + 1)
        print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

        # validation
        Full_SiT_model.eval()
        with torch.no_grad():
            all_val_loss = 0
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)

                val_represent, val_recons = Full_SiT_model(val_x)
                val_loss = loss_ae(val_x, val_recons)

                all_val_loss += val_loss.item()

            validation_loss = all_val_loss / (j + 1)

            writer.add_scalar("validation loss", validation_loss, epoch + 1)
            print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))

        # early_stopping(validation_loss, SiT_model)
        #
        # if early_stopping.early_stop:
        #     break

        if epoch == 0:
            min_val_loss = validation_loss

        if validation_loss < min_val_loss:
            torch.save({
                'model_state_dict': Full_SiT_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': validation_loss
            }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                        validation_loss))
            min_val_loss = validation_loss
            print("save weights !!")



    writer.close()


    print("Ending training \n")






#--------------------------------------------------------------------------------------------------------------------------------------------------------

def train_ClusterNET(args, train_dl, DTC_model, loss_ae, optimizer):
    """
    Function for training one epoch of the DTC
    """
    DTC_model.train()
    mse_loss = 0
    total_train_loss = 0
    kl_loss_Q = 0
    all_gt = []
    total_ari = 0

    for batch_idx, (inputs, labels) in enumerate(train_dl):  # epoch 1에 모든 training_data batch만큼
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        all_gt.append(labels.cpu().detach()) # all_gt = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])


        optimizer.zero_grad()

        z, x_reconstr, kl_loss, ARI = DTC_model(inputs, all_gt)  # ClusterNet의 forward

        loss_mse = loss_ae(inputs, x_reconstr)
        total_loss = loss_mse + kl_loss

        total_loss.backward()
        optimizer.step()




        total_train_loss += total_loss.item()
        mse_loss += loss_mse.item()
        kl_loss_Q += kl_loss.item()
        total_ari += ARI

    return (total_train_loss / (batch_idx + 1)), (mse_loss / (batch_idx + 1)),  (kl_loss_Q / (batch_idx + 1)), (total_ari / (batch_idx + 1))





def training_function(args, train_dl, valid_dl, directory, DTC_model, loss_ae, optimizer):

    """
    function for training the DTC network.
    """



    print("Training full DTC model ...")
    if not os.path.exists(os.path.join(directory, 'full_models')):
        os.makedirs(os.path.join(directory, 'full_models'))  ### for model save directory
    if not os.path.exists(os.path.join(directory, 'full_models_logs')):
        os.makedirs(os.path.join(directory, 'full_models_logs'))
    writer = SummaryWriter(log_dir=os.path.join(directory, 'full_models_logs'))

    f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'If the validation loss decreases compared to the previous epoch...')
    f.close()


    for epoch in tqdm(range(args.max_epochs)):
        train_loss, mse_loss, kl_loss, ARI = train_ClusterNET(args, train_dl, DTC_model, loss_ae, optimizer) # 1 epoch training
        print("For epoch ", epoch + 1, "Total Loss is : %.4f" % (train_loss), "MSE Loss is : %.4f" % ( mse_loss), "KL Loss is : %.4f" % (kl_loss), "ARI is : %.4f" % (ARI))

        writer.add_scalar("training total loss", train_loss, (epoch+1))
        writer.add_scalar("training MSE loss", mse_loss, (epoch + 1))
        writer.add_scalar("training KL loss", kl_loss, (epoch + 1))
        writer.add_scalar("training ARI", ARI, (epoch + 1))


        DTC_model.eval()
        with torch.no_grad():
            all_val_loss = 0
            all_val_mse_loss = 0
            all_val_kl_loss = 0
            total_ari = 0  # epoch = 0
            all_gt = []
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)
                all_gt.append(val_y.cpu().detach())


                z, x_reconstr, kl_loss, ARI = DTC_model(val_x, all_gt)

                V_loss_mse = loss_ae(val_x, x_reconstr)
                V_loss_KL = kl_loss
                V_total_loss = V_loss_mse + V_loss_KL

                all_val_mse_loss += V_loss_mse.item()
                all_val_loss += V_total_loss.item()
                all_val_kl_loss += V_loss_KL
                total_ari += ARI.item()





            all_val_mse_loss = all_val_mse_loss / (j+1)
            all_val_loss = all_val_loss / (j+1)
            all_val_kl_loss = all_val_kl_loss / (j+1)
            total_ari = total_ari / (j+1)
            print("For epoch ", epoch + 1, "val_Total Loss is : %.4f" % (all_val_loss), "val_MSE Loss is : %.4f" % (all_val_mse_loss), "val_KL Loss is : %.4f" % (all_val_kl_loss), "ARI is : %.4f" % total_ari)


            writer.add_scalar("validation total loss", all_val_loss, (epoch + 1))
            writer.add_scalar("validation MSE loss", all_val_mse_loss, (epoch + 1))
            writer.add_scalar("validation KL loss", all_val_kl_loss, (epoch + 1))
            writer.add_scalar("validation ARI", total_ari, (epoch + 1))


        f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
        writelog(f, 'Epoch: %d, valid total loss is: %.4f, valid mse loss is: %.4f, valid kl loss is: %.4f, ARI is: %.4f' % ((epoch + 1), all_val_loss, all_val_mse_loss, all_val_kl_loss, total_ari))
        f.close()


        torch.save(
            DTC_model,
            os.path.join(directory, 'full_models') + '/checkpoint_epoch_{}_loss_{:.5f}_model.pt'.format(epoch + 1, all_val_loss)
        )



    writer.close()
    print("Ending Training full model... \n")










