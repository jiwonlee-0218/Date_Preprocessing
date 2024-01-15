import torch.nn as nn
import torch

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import glob
import os
import test_fMRI_vision_transformer_SiT as vits
import test_fMRI_main


def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))


def compute_similarity(z, centroids, similarity="EUC"):   # z : (284, 10)
    """
    Function that compute distance between a latent vector z and the clusters centroids.
    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """
    n_clusters, n_hidden = centroids.shape[0], centroids.shape[1]  # 8, 10
    bs = z.shape[0] # (1080 * 284)

    if similarity == "CID":
        CE_z = compute_CE(z).unsqueeze(1)  # shape (batch_size , 1)
        CE_cen = compute_CE(centroids).unsqueeze(0)  ## shape (1 , n_clusters )
        z = z.unsqueeze(0).expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        CE_z = CE_z.expand((bs, n_clusters))  # (bs , n_clusters)
        CE_cen = CE_cen.expand((bs, n_clusters))  # (bs , n_clusters)
        CF = torch.max(CE_z, CE_cen) / torch.min(CE_z, CE_cen)
        return torch.transpose(mse, 0, 1) * CF

    elif similarity == "EUC":
        distance = torch.sqrt(torch.sum(((z.unsqueeze(1) - centroids) ** 2), dim=2)) # (Batch * Time, 8)
        return distance

    elif similarity == "COR":
        std_z = (
            torch.std(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_z = (
            torch.mean(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        std_cen = (
            torch.std(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_cen = (
            torch.mean(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        ## covariance
        z_expand = z.unsqueeze(1).expand((bs, n_clusters, n_hidden))
        cen_expand = centroids.unsqueeze(0).expand((bs, n_clusters, n_hidden))
        prod_expec = torch.mean(
            z_expand * cen_expand, dim=2
        )  ## (bs , n_clusters)
        pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)
        return torch.sqrt(2 * (1 - pearson_corr))














class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args, path='.'):
        super(ClusterNet, self).__init__()

        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")





        SiT_model = vits.__dict__[args.model](img_size=[args.img_size])  ### ENCODER ###
        SiT_model = test_fMRI_main.FullpiplineSiT(args, SiT_model)
        self.SiT_model = SiT_model.cuda()

        checkpoint = torch.load(full_path, map_location=args.device)
        self.SiT_model.load_state_dict(checkpoint['model_state_dict'])



        ## clustering model
        self.alpha_ = 1.0
        self.centr_size = 80
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity

    def kl_loss_function(self, input, pred):
        out = input * torch.log((input) / (pred))
        return torch.mean(torch.sum(out, dim=1))


    def init_centroids(self, z): # 여기서 _는 사용x
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        z = z.unsqueeze(dim=0)  # torch.Size([284, 116]) -> torch.Size([1, 284, 116])


        z_np = z.detach().cpu()  # z_np: (1, 284, 10)
        features = z_np.reshape(-1, z_np.shape[2])  # features: (1 * 284, 10)=(284, 10)


        assignements = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="complete", affinity="euclidean"
        ).fit(features)
        # assignements.labels_ = (sub * time) = (284,)

        centroids_ = torch.zeros(
            (self.n_clusters, self.centr_size), device=self.device
        )  # centroids_ : torch.Size([8, 10])

        for cluster_ in range(self.n_clusters):
            centroids_[cluster_] = features[assignements.labels_ == cluster_].mean(axis=0)


        self.centroids = nn.Parameter(centroids_)   # centroids_, self.centroids : torch.Size([8, 10]),

    def forward(self, x, all_gt):

        recons_l, recons_l_w, rec_imgs, orig_imgs, represent = self.SiT_model(x)
        z_np = represent.detach().cpu() # z_np: (64, 284, 10)


        ARI = 0
        all_kl_loss = 0
        pred_label = all_gt[0][0]




        for i in range(x.size(0)):  # x.size(0): 64 == batch_size, x: (64, 284, 116)
            X_tensor = z_np[i].type(torch.FloatTensor).to(self.device)  # z_np[i] : torch.Size([284, 10]) represented

            ## initialize clusters centroids
            self.init_centroids(X_tensor)

            similarity = compute_similarity(
                represent[i], self.centroids, similarity=self.similarity   # centroid와 representation간의 거리 계산
            )

            ## Q (batch_size , n_clusters)
            Q = torch.pow(      (1 + ( similarity / self.alpha_ )), -(self.alpha_ + 1) / 2      ) # (284, 8)
            sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)  # (284, 1)
            final_Q = Q / sum_columns_Q  # torch.Size([284, 8])

            ## P : ground truth distribution == target distribution
            P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)  #  torch.sum = torch.Size([8]) - .view = torch.Size([1, 8]), P = torch.Size([306720, 8])
            sum_columns_P = torch.sum(P, dim=1).view(-1, 1)  # torch.Size([306720, 1])
            final_P = P / sum_columns_P   # torch.Size([284, 8])

            loss_KL = self.kl_loss_function(final_P, final_Q)
            all_kl_loss += loss_KL.item()


            preds = torch.max(final_Q, dim=1)[1]   # preds: torch.Size([284])
            a = preds.cpu().numpy()
            b =  pred_label.cpu().numpy()
            ARI += adjusted_rand_score(a,b)



        LOSS_KL = all_kl_loss / (x.size(0)) # batch_size에 대한 loss
        all_batch_ARI = (ARI / (x.size(0))) # batch_size에 대한 ARI




        return represent, rec_imgs, LOSS_KL, all_batch_ARI, recons_l   # kl loss, all_preds = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])

