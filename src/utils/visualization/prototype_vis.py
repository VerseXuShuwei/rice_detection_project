"""
Prototype Cluster Visualization

Recent Updates:
    - [2026-01-14] Initial: t-SNE visualization for background prototypes

Key Features:
    - t-SNE dimensionality reduction for high-dimensional features
    - Interactive Plotly scatter plots with cluster coloring
    - Prototype center highlighting
    - Cluster distribution statistics

Usage:
    >>> from src.utils.visualization import visualize_clusters
    >>>
    >>> # After building prototypes
    >>> data = torch.load('outputs/prototypes/background_prototypes.pth')
    >>> visualize_clusters(
    ...     prototypes=data['prototypes'],
    ...     features=features_array,
    ...     labels=data['cluster_labels'],
    ...     output_path='outputs/prototypes/cluster_vis.html'
    ... )

Configuration:
    Automatically called by build_prototypes.py if feature_critic.construction.vis_output_path is set
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")


def visualize_clusters(
    prototypes: torch.Tensor,
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    sample_size: Optional[int] = 5000
) -> Dict[str, Any]:
    """
    Generate interactive t-SNE visualization of cluster distribution.

    Args:
        prototypes: (n_clusters, D) - Cluster centers
        features: (N, D) - All tile features
        labels: (N,) - Cluster assignment for each tile
        output_path: Path to save HTML visualization
        perplexity: t-SNE perplexity parameter (5-50, default 30)
        max_iter: t-SNE iterations (default 1000)
        random_state: Random seed for reproducibility
        sample_size: Subsample features if N > sample_size (for speed)
                     None = use all features

    Returns:
        stats: Dict with visualization statistics
            - n_samples: Number of samples visualized
            - n_clusters: Number of clusters
            - explained_variance_pca: PCA variance before t-SNE
    """
    if not SKLEARN_AVAILABLE or not PLOTLY_AVAILABLE:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install scikit-learn plotly"
        )

    # [新增] 强制 L2 归一化：把所有向量投影到单位球面上
    # 这样 t-SNE 的欧氏距离就等价于余弦距离了
    print("[VIS] Applying L2 Normalization for Cosine Similarity visualization...")
    features_norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    features = features / features_norm

    prototypes_norm = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-10
    prototypes = prototypes / prototypes_norm

    print(f"[VIS] Generating cluster visualization...")
    print(f"[VIS] Features: {features.shape}, Prototypes: {prototypes.shape}")

    # Convert to numpy
    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n_samples, n_features = features.shape
    n_clusters = len(prototypes)

    # Subsample if needed (t-SNE is slow for large datasets)
    if sample_size is not None and n_samples > sample_size:
        print(f"[VIS] Subsampling {sample_size}/{n_samples} tiles for visualization...")
        indices = np.random.RandomState(random_state).choice(
            n_samples, size=sample_size, replace=False
        )
        features_vis = features[indices]
        labels_vis = labels[indices]
    else:
        features_vis = features
        labels_vis = labels
        sample_size = n_samples

    # Step 1: PCA pre-processing (reduce to 50D if needed)
    if n_features > 50:
        print(f"[VIS] Applying PCA: {n_features}D → 50D...")
        pca = PCA(n_components=50, random_state=random_state)
        features_pca = pca.fit_transform(features_vis)
        prototypes_pca = pca.transform(prototypes)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"[VIS] PCA explained variance: {explained_var:.2%}")
    else:
        features_pca = features_vis
        prototypes_pca = prototypes
        explained_var = 1.0

    # Step 2: t-SNE dimensionality reduction
    print(f"[VIS] Running t-SNE (perplexity={perplexity}, n_iter={max_iter})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, sample_size - 1),  # Ensure perplexity < n_samples
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )

    # Combine tiles + prototypes for joint embedding
    combined = np.vstack([features_pca, prototypes_pca])
    embedded = tsne.fit_transform(combined)

    # Split back
    tiles_2d = embedded[:sample_size]
    prototypes_2d = embedded[sample_size:]

    print(f"[VIS] t-SNE completed. Embedded shape: {tiles_2d.shape}")

    # Step 3: Generate interactive Plotly visualization
    print(f"[VIS] Generating Plotly figure...")

    # Create color palette
    colors = px.colors.qualitative.Set3[:n_clusters]

    fig = go.Figure()

    # Plot tiles (scatter points)
    for cluster_id in range(n_clusters):
        mask = labels_vis == cluster_id
        n_tiles_in_cluster = mask.sum()
        pct = n_tiles_in_cluster / sample_size * 100

        fig.add_trace(go.Scatter(
            x=tiles_2d[mask, 0],
            y=tiles_2d[mask, 1],
            mode='markers',
            name=f'Cluster {cluster_id} ({n_tiles_in_cluster}, {pct:.1f}%)',
            marker=dict(
                size=3,
                color=colors[cluster_id],
                opacity=0.5,
                line=dict(width=0)
            ),
            hovertemplate=f'<b>Cluster {cluster_id}</b><br>' +
                          'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

    # Plot prototypes (large stars)
    fig.add_trace(go.Scatter(
        x=prototypes_2d[:, 0],
        y=prototypes_2d[:, 1],
        mode='markers+text',
        name='Prototypes',
        marker=dict(
            size=20,
            color='black',
            symbol='star',
            line=dict(color='white', width=2)
        ),
        text=[f'P{i}' for i in range(n_clusters)],
        textposition='top center',
        textfont=dict(size=12, color='black'),
        hovertemplate='<b>Prototype %{text}</b><br>' +
                      'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f't-SNE Visualization of Background Prototypes (n={sample_size}, k={n_clusters})',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        width=1200,
        height=800,
        hovermode='closest',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.02
        ),
        template='plotly_white'
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"[VIS] Saved interactive visualization to: {output_path}")

    # Return statistics
    stats = {
        'n_samples': sample_size,
        'n_clusters': n_clusters,
        'explained_variance_pca': float(explained_var),
        'output_path': str(output_path)
    }

    return stats


def generate_cluster_report(
    cluster_labels: np.ndarray,
    cluster_info: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate text report with cluster statistics.

    Args:
        cluster_labels: (N,) - Cluster assignments
        cluster_info: Dict from build_prototypes() with keys:
            - n_clusters
            - cluster_sizes
            - silhouette_score
            - inertia
        output_path: Path to save report (*.txt or *.md)
    """
    n_clusters = cluster_info['n_clusters']
    cluster_sizes = cluster_info['cluster_sizes']
    silhouette = cluster_info['silhouette_score']
    inertia = cluster_info['inertia']

    total_samples = len(cluster_labels)

    # Generate report
    lines = [
        "# Background Prototype Cluster Report",
        "",
        "## Summary Statistics",
        f"- **Total Samples:** {total_samples:,}",
        f"- **Number of Clusters:** {n_clusters}",
        f"- **Silhouette Score:** {silhouette:.4f} (Quality: {'Excellent' if silhouette > 0.7 else 'Good' if silhouette > 0.5 else 'Fair' if silhouette > 0.3 else 'Poor'})",
        f"- **K-Means Inertia:** {inertia:.2f}",
        "",
        "## Cluster Distribution",
        ""
    ]

    lines.append("| Cluster ID | Size | Percentage |")
    lines.append("|------------|------|------------|")
    for i, size in enumerate(cluster_sizes):
        pct = size / total_samples * 100
        lines.append(f"| Cluster {i} | {size:>6,} | {pct:>5.1f}% |")

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("**Silhouette Score Meaning:**")
    lines.append("- **> 0.7:** Excellent separation - clusters are well-defined")
    lines.append("- **0.5-0.7:** Good separation - clusters are distinct")
    lines.append("- **0.3-0.5:** Fair separation - some overlap exists")
    lines.append("- **< 0.3:** Poor separation - consider adjusting k")
    lines.append("")
    lines.append("**Cluster Size Balance:**")
    lines.append("- Ideally, no single cluster should dominate (>50% of samples)")
    lines.append("- Tiny clusters (<5% of samples) may indicate outliers")
    lines.append("")

    # Check for imbalance warnings
    max_pct = max(size / total_samples for size in cluster_sizes) * 100
    min_pct = min(size / total_samples for size in cluster_sizes) * 100

    if max_pct > 50:
        lines.append("⚠️ **WARNING:** One cluster contains >50% of samples. Consider increasing k.")
        lines.append("")
    if min_pct < 5:
        lines.append("⚠️ **WARNING:** One cluster contains <5% of samples (possible outliers).")
        lines.append("")

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[REPORT] Saved cluster report to: {output_path}")
