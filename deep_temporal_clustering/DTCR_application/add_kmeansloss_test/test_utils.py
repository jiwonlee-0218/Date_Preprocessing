import torch
import numpy as np
from scipy.special import comb
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score
from sklearn.cluster import KMeans
# from torchmetrics.functional import pairwise_euclidean_distance


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

    cos_sim = torch.nn.CosineSimilarity(dim=2)
    distance = cos_sim(z.unsqueeze(1), centroids)
    # distance = pairwise_euclidean_distance(z, centroids) # (Batch * Time, 8)
    # distance = torch.sqrt(torch.sum(((z.unsqueeze(1) - centroids) ** 2), dim=2) + 1.e-12) # (Batch * Time, 8)
    return distance



def ri_score(y_true, y_pred):
    return rand_score(y_true, y_pred)

def nmi_score(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')

def cluster_using_kmeans(embeddings, K):
    kmc = KMeans(n_clusters=K).fit(embeddings)
    pmc = kmc.predict(embeddings)
    return pmc