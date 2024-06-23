import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def perform_lda(X, y, num_components=2):
    try:
        # Try the original LDA method first
        mean_vectors = [np.mean(X[y == cl], axis=0) for cl in np.unique(y)]
        
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for cl, mv in zip(np.unique(y), mean_vectors):
            class_sc_mat = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cl]:
                row, mv = row.reshape(X.shape[1], 1), mv.reshape(X.shape[1], 1)
                class_sc_mat += (row - mv).dot((row - mv).T)
            S_W += class_sc_mat
        
        overall_mean = np.mean(X, axis=0).reshape(X.shape[1], 1)
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == i, :].shape[0]
            mean_vec = mean_vec.reshape(X.shape[1], 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
        # Add a small constant to the diagonal of S_W to avoid singularity
        epsilon = 1e-6
        S_W += epsilon * np.eye(S_W.shape[0])
        
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        W = np.hstack([eigen_pairs[i][1].reshape(X.shape[1], 1) for i in range(min(num_components, len(eigen_pairs)))])
        
        X_lda = X.dot(W)
        return X_lda, W
    
    except np.linalg.LinAlgError:
        print("LDA failed. Falling back to TruncatedSVD.")
        # Fallback to TruncatedSVD (similar to LSA) if LDA fails
        svd = TruncatedSVD(n_components=num_components, random_state=42)
        X_svd = svd.fit_transform(X)
        return X_svd, svd.components_.T

def visualize_lda(X_lda, titles):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(titles)))
    
    for i, (title, color) in enumerate(zip(titles, colors)):
        plt.scatter(x=X_lda[i, 0].real,
                    y=X_lda[i, 1].real,
                    color=color,
                    alpha=0.7,
                    s=100)  # Increase point size
        plt.annotate(title[:30] + '...', (X_lda[i, 0].real, X_lda[i, 1].real),
                     xytext=(5, 2), textcoords='offset points', fontsize=8,
                     alpha=0.8)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Document projections onto the first 2 components')
    plt.tight_layout()
    plt.grid(True)

    return plt.gcf()