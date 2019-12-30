from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#TODO think about the comment of first using PCA then tsne of dimensions are very big


class DimensionReducer:
    def __init__(self, mode="tsne"):
        self.mode = mode

    def reduce(self, matrix):
        if self.mode == "tsne-big":
            first_reduced_matrix = PCA(n_components=2).fit_transform(matrix)
            return TSNE(n_components=2).fit_transform(first_reduced_matrix)

        elif self.mode == "tsne":
            return TSNE(n_components=2).fit_transform(matrix)

        elif self.mode == "pca":
            return PCA(n_components=2).fit_transform(matrix)
        else:
            raise(Exception("please select either 'pca' or 'tsne' as mode"))

