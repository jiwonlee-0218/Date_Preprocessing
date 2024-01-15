import torch


def load_optimizer(args, model_list):
    model_df, model_tf, model_cls, model_dec = model_list[0], model_list[1], model_list[2], model_list[3]
    optimizer = torch.optim.AdamW([{"params": model_df.parameters(), "lr": args.lr_df},
                                   {"params": model_tf.intra_net.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.encoding_block_sa.parameters(), "lr": args.lr_tf},
                                   {"params": model_tf.cls_token, 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_cls.parameters(), 'lr': args.lr_pr, "weight_decay": 0},
                                   {"params": model_dec.parameters(), "lr": args.lr_dc},
                                   ], lr=args.lr_tf, weight_decay=args.l2)
    optimizer2 = torch.optim.AdamW([{"params": model_dec.parameters(), "lr": args.lr_dc2}], lr=args.lr_dc2, weight_decay=args.l2)
    return [model_df, model_tf, model_cls, model_dec], optimizer, optimizer2
