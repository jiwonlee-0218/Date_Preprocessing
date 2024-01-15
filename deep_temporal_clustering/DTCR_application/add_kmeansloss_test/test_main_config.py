import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--dataset_name", default="EMOTION_7cluster_include_cue", help="dataset name")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID"], default="EUC", help="The similarity type")

    # model args
    parser.add_argument("--model_name", default="DTCR_bidirection_dilated_RNN2",help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")

    # parser.add_argument('--clip_grad', type=float, default=5.0, help="Gradient clipping: Maximal parameter gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=351, help="Epochs number of the autoencoder training",)
    parser.add_argument("--max_patience", type=int, default=15, help="The maximum patience for pre-training, above which we stop training.",)

    parser.add_argument("--lr_ae", type=float, default=0.005, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_test',)
    # parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')
    # parser.add_argument("--ae_models", default='full_models/', help='full autoencoder weights')
    parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
    parser.add_argument("--ae_models", default=None, help='full autoencoder weights')
    parser.add_argument("--autoencoder_test", default=None, help='full autoencoder weights')




    return parser


# def get_arguments():
#     parser = argparse.ArgumentParser()
#
#
#
# #     # data args
#     parser.add_argument(
#         "--dataset_name",
#         default="FingerMovements",
#         help="dataset name"
#     )
#     parser.add_argument(
#         "--path_data",
#         default="data/{}",
#         help="dataset name"
#     )
#
#     # model args
#     parser.add_argument(
#         "--pool",
#         required=False,
#         default=1,
#         help="pooling hyperparameter. Refer to the paper for each dataset's corresponding value",
#     )
#     parser.add_argument(
#         "--similarity",
#         required=False,
#         choices=["COR", "EUC", "CID"],
#         default="EUC",
#         help="The similarity type",
#     )
#     parser.add_argument(
#         "--path_weights",
#         default="models_weights/{}/",
#         help="embedding weights",
#     )
#     parser.add_argument(
#         "--path_logs",
#         default="models_logs/{}/",
#         help="embedding logs",
#     )
#     parser.add_argument(
#         "--n_clusters",
#         type=int,
#         default=2,
#         help="Number of clusters , corresponding to the labels number",
#     )
#
#     parser.add_argument(
#         "--alpha",
#         type=int,
#         default=1,
#         help="alpha hyperparameter for DTC model",
#     )
#     # training args
#
#     parser.add_argument(
#         "--batch_size",
#         default=32,
#         help="batch size"
#     )
#     parser.add_argument(
#         "--epochs_ae",
#         type=int,
#         default=10,
#         help="Epochs number of the autoencoder training",
#     )
#     parser.add_argument(
#         "--max_epochs",
#         type=int,
#         default=50,
#         help="Maximum epochs numer of the full model training",
#     )
#
#     parser.add_argument(
#         "--max_patience",
#         type=int,
#         default=5,
#         help="The maximum patience for DTC training , above which we stop training.",
#     )
#
#     parser.add_argument(
#         "--lr_ae",
#         type=float,
#         default=1e-2,
#         help="Learning rate of the autoencoder training",
#     )
#     parser.add_argument(
#         "--lr_cluster",
#         type=float,
#         default=1e-2,
#         help="Learning rate of the full model training",
#     )
#     parser.add_argument(
#         "--momentum",
#         type=float,
#         default=0.9,
#         help="SGD momentum for the full model training",
#     )
#
#     return parser


