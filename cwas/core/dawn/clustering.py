from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from cwas.utils.log import print_progress

class kmeans_cluster:
    def __init__(self, tsne_out: pd.DataFrame, seed: int) -> None:

        self._tsne_out = tsne_out
        self._seed = seed
        self._distance_matrix = None  # cached distance matrix
        self._best_km = None  # cached best KMeans model from optimal_k

    @property
    def tsne_out(self) -> pd.DataFrame:
        return self._tsne_out

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def distance_matrix(self):
        if self._distance_matrix is None:
            self._distance_matrix = squareform(pdist(self.tsne_out))
        return self._distance_matrix

    def optimal_k(self, k_range, output_name, parsimonious=True):
        start, end = list(map(int, k_range.replace(" ", "").split(",")))
        if start < 2 or start > end:
            raise ValueError(f"Invalid k_range: start={start}, end={end}. Require 2 <= start <= end.")
        k_values = list(range(start, end+1))

        avg_sil_values = list(map(self._avg_sil, k_values))
        df = pd.DataFrame({"k_values": k_values, "avg_sil_values": avg_sil_values})
        self.avg_sil_df = df

        opt_k = df.loc[df.avg_sil_values==max(df.avg_sil_values), "k_values"].values[0]
        parsimonious_k = None

        if parsimonious:
            max_sil = df.avg_sil_values.max()
            min_sil = df.avg_sil_values.min()
            sil_range = max_sil - min_sil
            if sil_range > 0:
                sil_threshold = max_sil - 0.05 * sil_range
            else:
                sil_threshold = max_sil
            candidates = df.loc[df.avg_sil_values >= sil_threshold, "k_values"].values

            if len(candidates) == 0:
                print_progress(f"Parsimonious K: no candidates in plateau; using silhouette-max K={opt_k}")
                parsimonious_k = opt_k
            else:
                parsimonious_k = int(candidates.min())
                print_progress(
                    f"Parsimonious K: {len(candidates)} candidate K values in plateau "
                    f"(silhouette >= {sil_threshold:.4f}), selected smallest K={parsimonious_k}"
                )

            opt_k = parsimonious_k

        self._print_silhouett_plot(output_name, parsimonious_k=parsimonious_k)

        # Fit final KMeans with full n_init=300 for the optimal k and cache it
        km = KMeans(n_clusters=opt_k, n_init=300, max_iter=100, random_state=self.seed)
        km.fit(np.array(self.tsne_out))
        self._best_km = km

        return opt_k

    def _avg_sil(self, k):
        km = KMeans(n_clusters=k, n_init=50, max_iter=100, random_state=self.seed)
        labels = km.fit_predict(np.array(self.tsne_out))
        # Reuse cached distance matrix
        ss = silhouette_score(self.distance_matrix, labels, metric='precomputed')

        return ss

    def _print_silhouett_plot(self, output_name, parsimonious_k=None):
        k_start = min(self.avg_sil_df['k_values'])
        k_end = max(self.avg_sil_df['k_values'])
        max_ss = max(self.avg_sil_df.avg_sil_values)
        k = self.avg_sil_df.loc[self.avg_sil_df.avg_sil_values==max_ss, "k_values"].values[0]

        plt.figure(figsize=(20, 10))
        plt.title('Range of K: {} to {}'.format(k_start, k_end), fontsize=15, pad=10)
        plt.plot(self.avg_sil_df['k_values'], self.avg_sil_df['avg_sil_values'], '-o', c="black", linewidth=1, markersize=4)
        plt.vlines(k, 0, max_ss, color='red', linestyles='--')
        plt.text(k, max_ss+0.01, "Optimal K = {}".format(str(k)), fontdict={'color': 'red', 'weight': 'bold', 'fontsize': 15})
        if parsimonious_k is not None and parsimonious_k != k:
            pars_ss = self.avg_sil_df.loc[self.avg_sil_df.k_values == parsimonious_k, "avg_sil_values"].values[0]
            plt.vlines(parsimonious_k, 0, pars_ss, color='blue', linestyles='--')
            plt.text(parsimonious_k, pars_ss+0.01, "Parsimonious K = {}".format(str(parsimonious_k)), fontdict={'color': 'blue', 'weight': 'bold', 'fontsize': 15})
        plt.xticks(np.arange(0, int(np.ceil(k_end/10)*10)+1, 10), fontsize=10)
        plt.yticks(np.arange(0, max_ss+0.1, 0.1), fontsize=10)
        plt.xlim(k_start-5, k_end+5)
        plt.ylim(0, max_ss+0.1)
        plt.xlabel("Number of clusters K", fontsize=12, labelpad=10)
        plt.ylabel("Average Silhouettes", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(output_name) # final output
        plt.close()


    def center_init(self, k):
        # Reuse cached KMeans model if available (same k), otherwise fit new one
        if self._best_km is not None and self._best_km.n_clusters == k:
            km = self._best_km
        else:
            km = KMeans(n_clusters=k, n_init=300, max_iter=100, random_state=self.seed)
            km.fit(self.tsne_out)

        km_tsne_cluster = km.labels_.tolist()  # already 0-indexed
        km_tsne_centers = pd.DataFrame(km.cluster_centers_[km_tsne_cluster], columns=["t-SNE1", "t-SNE2"])
        dist_to_center_tsne = np.sqrt(np.sum((self.tsne_out - km_tsne_centers)**2, axis=1))

        self.km_tsne_cluster_ = km_tsne_cluster
        self.dist_to_center_tsne_ = dist_to_center_tsne

        i_init_pt = list(map(self._init_pt_kmeans, list(range(k))))

        return i_init_pt

    def _init_pt_kmeans(self, k):
        i_check = list(np.where([x == k for x in self.km_tsne_cluster_])[0])
        i_check_min = self.dist_to_center_tsne_[i_check].idxmin()

        return i_check_min
