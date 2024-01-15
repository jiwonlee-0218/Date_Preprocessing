from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping


import os
import datetime
import time
import torch



import utils
import fMRI_main
import fMRI_vision_transformer_SiT as vits
import clusternet



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
    SiT_model = vits.__dict__[args.model](img_size=[args.img_size])  ### ENCODER ###


    SiT_model = fMRI_main.FullpiplineSiT(args, SiT_model)
    SiT_model = SiT_model.cuda()
    print(f"==> {args.model} model is created.")


    ################ optimization ...
    # Create Optimizer

    optimizer = torch.optim.AdamW(SiT_model.parameters())  # to use with ViTs



    to_restore = {"epoch": 0} # dict
    start_epoch = to_restore["epoch"]





    ################ Training
    start_time = time.time()
    print(f"==> Start training from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))
    early_stopping = EarlyStopping(patience=15, verbose=True)

    for epoch in range(args.epochs):


        # Train an epoch
        SiT_model.train()
        all_loss = 0
        for it, (im, _) in enumerate(train_dl):

            im = im.type(torch.FloatTensor).to(args.device)
            recons_l, rec_imgs, orig_imgs, represent = SiT_model(im)

            loss = recons_l

            # model update
            optimizer.zero_grad()


            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(SiT_model.parameters(), args.clip_grad)
            optimizer.step()

            all_loss += loss.item()

        train_loss = all_loss / (it + 1)

        writer.add_scalar("training loss", train_loss, epoch + 1)
        print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

        # validation
        SiT_model.eval()
        with torch.no_grad():
            all_val_loss = 0
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)
                val_recons_l, val_rec_imgs, val_orig_imgs, val_represent = SiT_model(val_x)

                val_loss = val_recons_l

                all_val_loss += val_loss.item()

            validation_loss = all_val_loss / (j + 1)

            writer.add_scalar("validation loss", validation_loss, epoch + 1)
            print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))

        early_stopping(validation_loss, SiT_model)

        if early_stopping.early_stop:
            break

        if epoch == 0:
            min_val_loss = validation_loss

        if validation_loss < min_val_loss:
            torch.save({
                'model_state_dict': SiT_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': validation_loss
            }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                        validation_loss))
            min_val_loss = validation_loss
            print("save weights !!")



    writer.close()


    print("Ending training \n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



#--------------------------------------------------------------------------------------------------------------------------------------------------------

def training_function(args, directory, train_dl, valid_dl, verbose=True):

    """
    function for training the DTC network.
    """

    path = os.path.join(directory, args.weights)
    DTC_model = clusternet.ClusterNet(args, path=path)  ## 모델 초기화 with the pretrained autoencoder model 정의해놓고 입력넣으면 바로 쓸 수 있게
    DTC_model = DTC_model.to(args.device)
    optimizer_clu = torch.optim.Adam(DTC_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)


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
    writelog(f, 'Epoch: %d' % args.epochs)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'If the validation loss decreases compared to the previous epoch...')
    f.close()


    for epoch in range(args.max_epochs):
        train_loss, mse_loss, kl_loss = train_ClusterNET(args, train_dl, DTC_model, optimizer_clu) # 1 epoch training

        print("For epoch ", epoch + 1, "Total Loss is : %.4f" % (train_loss), "MSE Loss is : %.4f" % ( mse_loss), "KL Loss is : %.4f" % (kl_loss))

        writer.add_scalar("training total loss", train_loss, (epoch+1))
        writer.add_scalar("training MSE loss", mse_loss, (epoch + 1))
        writer.add_scalar("training KL loss", kl_loss, (epoch + 1))


        DTC_model.eval()
        with torch.no_grad():
            all_val_loss = 0
            all_val_SiT_loss = 0
            all_val_kl_loss = 0
            total_ari = 0  # epoch = 0
            all_gt = []
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)
                all_gt.append(val_y.cpu().detach())


                z, x_reconstr, loss_KL, ARI, SiT_loss = DTC_model(val_x, all_gt)


                total_loss = args.gamma * SiT_loss + (1-args.gamma) * loss_KL


                all_val_SiT_loss += SiT_loss.item() * args.gamma
                all_val_loss += total_loss.item()
                all_val_kl_loss += loss_KL * (1-args.gamma)
                total_ari += ARI.item()





            all_val_SiT_loss = all_val_SiT_loss / (j+1)
            all_val_loss = all_val_loss / (j+1)
            all_val_kl_loss = all_val_kl_loss / (j+1)
            total_ari = total_ari / (j+1)
            print("For epoch ", epoch + 1, "val_Total Loss is : %.4f" % (all_val_loss), "val_SiT Loss is : %.4f" % (all_val_SiT_loss), "val_KL Loss is : %.4f" % (all_val_kl_loss))



            writer.add_scalar("validation total loss", all_val_loss, (epoch + 1))
            writer.add_scalar("validation MSE loss", all_val_SiT_loss, (epoch + 1))
            writer.add_scalar("validation KL loss", all_val_kl_loss, (epoch + 1))

        if epoch == 0:
            min_val_loss = all_val_loss
            print("ARI is : %.4f" % total_ari)

            f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
            writelog(f, 'Epoch: %d, ARI is: %.4f' % ((epoch + 1), total_ari))
            f.close()

            torch.save(
                DTC_model,
                os.path.join(directory, 'full_models') + '/checkpoint_epoch_{}_loss_{:.5f}_model.pt'.format(epoch + 1, all_val_loss)
            )


        if all_val_loss < min_val_loss:
            print( "ARI is : %.4f" % total_ari )

            f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
            writelog(f, 'Epoch: %d, ARI is: %.4f' % ((epoch+1), total_ari))
            f.close()




            torch.save(
                DTC_model, os.path.join(directory, 'full_models')+'/checkpoint_epoch_{}_loss_{:.5f}_model.pt'.format(epoch+1, all_val_loss)
            )
            min_val_loss = all_val_loss



    writer.close()
    print("Ending Training full model... \n")



def train_ClusterNET(args, train_dl, DTC_model, optimizer_clu):
    """
    Function for training one epoch of the DTC
    """
    DTC_model.train()
    total_SiT_loss = 0
    total_train_loss = 0
    kl_loss = 0
    all_gt = []

    for batch_idx, (inputs, labels) in enumerate(train_dl):  # epoch 1에 모든 training_data batch만큼
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        all_gt.append(labels.cpu().detach()) # all_gt = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])


        optimizer_clu.zero_grad()

        z, x_reconstr, loss_KL, ARI, SiT_loss = DTC_model(inputs, all_gt)  # ClusterNet의 forward




        total_loss = args.gamma * SiT_loss + (1-args.gamma) * loss_KL
        total_loss.backward() # backpropagation, compute gradient
        optimizer_clu.step() # update


        total_train_loss += total_loss.item()
        total_SiT_loss += SiT_loss.item() * args.gamma
        kl_loss += loss_KL * (1-args.gamma)

    return (total_train_loss / (batch_idx + 1)), (total_SiT_loss / (batch_idx + 1)), (kl_loss / (batch_idx + 1))



