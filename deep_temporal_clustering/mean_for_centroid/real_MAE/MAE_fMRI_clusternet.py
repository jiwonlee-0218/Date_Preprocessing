import torch.nn as nn
import torch

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score






def compute_similarity(z, centroids, similarity="EUC"):   # z : (284, 10)
    """
    Function that compute distance between a latent vector z and the clusters centroids.
    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """


    if similarity == "COS":
        distance = torch.nn.functional.cosine_similarity(z.unsqueeze(1), centroids, dim=2)
        # cos_sim = torch.nn.CosineSimilarity(dim=2)
        # distance = cos_sim(z.unsqueeze(1), centroids)

    elif similarity == "EUC":
        distance = torch.sqrt(torch.sum(((z.unsqueeze(1) - centroids) ** 2), dim=2)) # (Batch * Time, 8)


    return distance















class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args, Full_SiT_model: torch.nn.Module):
        super(ClusterNet, self).__init__()





        self.Full_SiT_model = Full_SiT_model





        ## clustering model
        self.alpha_ = 1.0
        self.centr_size = 64
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity
        self.loss_function = nn.KLDivLoss(reduction='batchmean')






    def forward(self, x, all_gt):

        z, x_reconstr = self.Full_SiT_model(x)

        ARI = 0
        all_kl_loss = 0
        pred_label = all_gt[0][0]

        z_np = z.detach().cpu()
        for i in range(z_np.size(0)):
            assignements = KMeans(n_clusters=self.n_clusters).fit(z_np[i])

            centroids_ = torch.zeros(
                (self.n_clusters, z_np[i].shape[1]), device=self.device
            )

            for cluster_ in range(self.n_clusters):
                centroids_[cluster_] = z_np[i][assignements.labels_ == cluster_].mean(axis=0)

            cluster_centers = centroids_

            similarity = compute_similarity(
                z[i], cluster_centers, similarity=self.similarity  # centroid와 representation간의 거리 계산
            )



            ## Q (batch_size , n_clusters)
            Q = torch.pow(      (1 + ( similarity / self.alpha_ )), -(self.alpha_ + 1) / 2      ) # (284, 8)
            sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)  # (284, 1)
            final_Q = Q / sum_columns_Q  # torch.Size([284, 8])

            ## P : ground truth distribution == target distribution
            P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)  #  torch.sum = torch.Size([8]) - .view = torch.Size([1, 8]), P = torch.Size([306720, 8])
            sum_columns_P = torch.sum(P, dim=1).view(-1, 1)  # torch.Size([306720, 1])
            final_P = P / sum_columns_P   # torch.Size([284, 8])

            kl_loss = self.loss_function(final_Q.log(), final_P.detach())
            all_kl_loss += kl_loss


            preds = torch.max(final_Q, dim=1)[1]   # preds: torch.Size([284])
            a = preds.cpu().numpy()
            b =  pred_label.cpu().numpy()
            ARI += adjusted_rand_score(a,b)



        LOSS_KL = all_kl_loss / (z_np.size(0)) # batch_size에 대한 loss
        all_batch_ARI = (ARI / (x.size(0))) # batch_size에 대한 ARI




        return  z, x_reconstr, LOSS_KL, all_batch_ARI   # kl loss, all_preds = list이므로 batch_idx=0일 때, len(list)=1, list[0].shape = torch.Size([64, 284])

