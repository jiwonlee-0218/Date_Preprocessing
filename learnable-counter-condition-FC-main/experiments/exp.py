import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utilss import metric as mt




def train_pfc(args, device, models, optimizer2, dataloader, criterions):
    criterion_cls, criterion_rec = criterions[0], criterions[1]
    model_df, model_tf, model_class, model_dc = models[0], models[1], models[2], models[3]
    inds = torch.triu_indices(args.input_size, args.input_size, 1).to(device)

    num_trn = dataloader[0].shape[0]
    trn_shf = torch.randperm(num_trn)
    data_shf, labl_shf = dataloader[0][trn_shf].to(device), dataloader[1][trn_shf].to(device)
    for i in range(int(num_trn/args.bs)):

        "To extract prototype-based FC"
        with torch.no_grad():
            model_df.eval(), model_tf.eval()
            model_class.eval(), model_dc.eval()

            data, targets = data_shf[i*args.bs:(i+1)*(args.bs)], labl_shf[i*args.bs:(i+1)*(args.bs)]
            output_df, _ = model_df(data)
            output_tf = model_tf(output_df)
            cls_token, roi_token = output_tf[:, 0, :], output_tf[:, 1:, :]
            output = model_class(cls_token.unsqueeze(1))

            _, pred = torch.max(F.softmax(output, -1).data, -1)
            correct_normal = torch.where((pred == targets) & (targets == 0))[0].to(device)
            correct_patient = torch.where((pred == targets) & (targets == 1))[0].to(device)

            if correct_patient.shape[0] < correct_normal.shape[0]:
                correct_normal = correct_normal[:correct_patient.shape[0]]
            elif correct_patient.shape[0] > correct_normal.shape[0]:
                correct_patient = correct_patient[:correct_normal.shape[0]]
            else:
                pass

        "Train decoder"
        model_df.eval(), model_tf.eval()
        model_class.eval(), model_dc.train()

        model_df.set_require_grad(False)
        model_tf.set_require_grad(False)
        model_class.set_require_grad(False)

        optimizer2.zero_grad()
        "Prototype-based FC"
        correct_idx_baln = torch.cat([correct_normal, correct_patient])
        correct_idx_perm = torch.flip(correct_idx_baln, dims=(0,))

        origin_norm = torch.norm(cls_token[correct_idx_baln], dim=1).unsqueeze(-1)
        permut_norm = torch.norm(cls_token[correct_idx_perm], dim=1).unsqueeze(-1)
        integrate = roi_token.mean(1)[correct_idx_baln] + (cls_token[correct_idx_perm] / permut_norm * origin_norm)

        # Shuffle
        second_perm = torch.randperm(integrate.shape[0])
        recon = model_dc(integrate[second_perm])
        label = targets[correct_idx_perm][second_perm]

        protofc = torch.zeros(recon.shape[0], args.input_size, args.input_size).to(device)
        protofc[:, inds[0], inds[1]] = recon
        protofc = protofc + torch.transpose(protofc, 2, 1)

        output_tf_p = model_tf(protofc)
        cls_token_p = output_tf_p[:, 0, :]
        output_p = model_class(cls_token_p.unsqueeze(1))
        loss_cls = args.h_cls2 * criterion_cls(output_p, label.to(device))

        # Original
        integrate_o = roi_token.mean(1) + cls_token.squeeze(1)
        recon_o = model_dc(integrate_o)
        loss_rec = args.h_rec2 * criterion_rec(recon_o, output_df[:, inds[0], inds[1]])

        (loss_cls + loss_rec).backward()
        optimizer2.step()


def train_fc(args, device, models, optimizer, dataloader, criterions):
    criterion_cls, criterion_rec = criterions[0], criterions[1]
    model_df, model_tf, model_class, model_dc = models[0], models[1], models[2], models[3]
    inds = torch.triu_indices(args.input_size, args.input_size, 1)

    model_df.train(), model_tf.train()
    model_class.train(), model_dc.train()

    model_df.set_require_grad(True)
    model_tf.set_require_grad(True)
    model_class.set_require_grad(True)

    num_trn = dataloader[0].shape[0]
    trn_shf = torch.randperm(num_trn)
    data_shf, labl_shf = dataloader[0][trn_shf].to(device), dataloader[1][trn_shf].to(device)
    for i in range(int(num_trn/args.bs)):
        optimizer.zero_grad()

        data, targets = data_shf[i*args.bs:(i+1)*(args.bs)], labl_shf[i*args.bs:(i+1)*(args.bs)]
        output_df, _ = model_df(data)
        output_tf = model_tf(output_df)
        cls_token, roi_token = output_tf[:, 0, :], output_tf[:, 1:, :]
        output, roi_reg = model_class(cls_token.unsqueeze(1), roi_token.mean(1).unsqueeze(1))
        loss_cls = args.h_cls * criterion_cls(output, targets)

        integrate = roi_token.mean(1) + cls_token.squeeze(1)
        recon = model_dc(integrate)
        loss_rec = args.h_rec * criterion_rec(recon, output_df[:, inds[0], inds[1]])
        loss_roi = args.h_roi * torch.mean(torch.abs(roi_reg))

        (loss_cls + loss_rec + loss_roi).backward()
        optimizer.step()






def evaluate(args, device, epoch, models, dataloader, criterions, phase):
    criterion_cls, criterion_rec = criterions[0], criterions[1]
    model_df, model_tf, model_class, model_dc = models[0], models[1], models[2], models[3]
    inds = torch.triu_indices(args.input_size, args.input_size, 1)

    model_df.eval(), model_tf.eval()
    model_class.eval(), model_dc.eval()
    with torch.no_grad():
        total_rloss, total_closs, total_iloss = 0., 0., 0.
        logits_p, labels = [], []
        num_data = dataloader[0].shape[0]
        data_shf, labl_shf = dataloader[0].to(device), dataloader[1].to(device)
        for i in range(int(num_data / args.bs)):
            data, targets = data_shf[i * args.bs:(i + 1) * (args.bs)], labl_shf[i * args.bs:(i + 1) * (args.bs)]
            output_df, _ = model_df(data.to(device))
            output_tf = model_tf(output_df)
            cls_token = output_tf[:, 0, :]
            roi_token = output_tf[:, 1:, :]
            output, roi_reg = model_class(cls_token.unsqueeze(1), roi_token.mean(1).unsqueeze(1))
            loss_cls = args.h_cls * criterion_cls(output, targets.to(device))

            integrate = roi_token.mean(1) + cls_token.squeeze(1)
            recon = model_dc(integrate)
            loss_rec = args.h_rec * criterion_rec(recon, output_df[:, inds[0], inds[1]].to(device))
            loss_roi = args.h_roi * torch.mean(torch.abs(roi_reg))

            logits_p.append(output.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())

            total_rloss += loss_rec.item()
            total_closs += loss_cls.item()
            total_iloss += loss_roi.item()

    total_aloss = total_rloss + total_closs + total_iloss
    result_p = mt.cal_all_metric(logits_p, labels)

    if epoch % 10 == 0:
        print('|E{:3d}| {} | loss {:5.4f}/{:5.4f}/{:5.4f}/{:5.4f}/ | {:5.4f}/{:5.4f}/{:5.4f}/{:5.4f}'.format(
                epoch, phase, total_aloss/(i+1), total_rloss/(i+1), total_closs/(i+1), total_iloss/(i+1),
                result_p[1], result_p[0], result_p[2], result_p[3]))

    return [total_aloss/(i+1), total_rloss/(i+1), total_closs/(i+1)], [result_p]
