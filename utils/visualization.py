"""
Visualization utilities for Decision Trees Dashboard.
Creates interactive and static visualizations using Plotly and Matplotlib.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from typing import List, Dict, Any
import io
import base64


def plot_tree_matplotlib(tree_model: DecisionTreeClassifier,
                        feature_names: List[str],
                        class_names: List[str],
                        figsize: tuple = (20, 12)) -> plt.Figure:
    """
    Create matplotlib visualization of decision tree.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=[str(cls) for cls in class_names],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    
    ax.set_title("Decision Tree Visualization", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str]) -> go.Figure:
    """
    Create interactive heatmap of confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        
    Returns:
        Plotly Figure object
    """
    # Normalize for percentage display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create hover text
    hover_text = []
    for i in range(len(class_names)):
        hover_row = []
        for j in range(len(class_names)):
            hover_row.append(
                f"True: {class_names[i]}<br>"
                f"Predicted: {class_names[j]}<br>"
                f"Count: {cm[i][j]}<br>"
                f"Percentage: {cm_normalized[i][j]:.1f}%"
            )
        hover_text.append(hover_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertext=hover_text,
        hoverinfo='text',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        height=500,
        font=dict(size=12)
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 15) -> go.Figure:
    """
    Create bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        
    Returns:
        Plotly Figure object
    """
    df_plot = importance_df.head(top_n).sort_values('importance')
    
    fig = go.Figure(go.Bar(
        x=df_plot['importance'],
        y=df_plot['feature'],
        orientation='h',
        marker=dict(
            color=df_plot['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=df_plot['importance'].apply(lambda x: f'{x:.4f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {min(top_n, len(importance_df))} Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=max(400, top_n * 25),
        showlegend=False,
        font=dict(size=11)
    )
    
    return fig


def plot_metrics_comparison(metrics: Dict[str, float]) -> go.Figure:
    """
    Create bar chart comparing train/test metrics.
    
    Args:
        metrics: Dictionary with metric names and values
        
    Returns:
        Plotly Figure object
    """
    metric_types = ['accuracy', 'precision', 'recall', 'f1']
    train_values = [metrics.get(f'train_{m}', 0) for m in metric_types]
    test_values = [metrics.get(f'test_{m}', 0) for m in metric_types]
    
    fig = go.Figure(data=[
        go.Bar(
            name='Training',
            x=metric_types,
            y=train_values,
            text=[f'{v:.3f}' for v in train_values],
            textposition='outside',
            marker_color='#3b82f6'
        ),
        go.Bar(
            name='Test',
            x=metric_types,
            y=test_values,
            text=[f'{v:.3f}' for v in test_values],
            textposition='outside',
            marker_color='#10b981'
        )
    ])
    
    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.1]),
        barmode='group',
        height=400,
        font=dict(size=12),
        hovermode='x unified'
    )
    
    return fig


def plot_class_distribution(y: pd.Series,
                           title: str = "Class Distribution") -> go.Figure:
    """
    Create bar chart of class distribution.
    
    Args:
        y: Target Series
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    class_counts = y.value_counts().sort_index()
    
    fig = go.Figure(go.Bar(
        x=class_counts.index.astype(str),
        y=class_counts.values,
        text=class_counts.values,
        textposition='outside',
        marker=dict(
            color=class_counts.values,
            colorscale='Portland',
            showscale=False
        ),
        hovertemplate='<b>Class %{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=(class_counts.values / class_counts.sum() * 100)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Class',
        yaxis_title='Number of Samples',
        height=400,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig


def plot_tree_depth_analysis(tree_model: DecisionTreeClassifier) -> go.Figure:
    """
    Visualize tree structure statistics.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        
    Returns:
        Plotly Figure object with tree statistics
    """
    tree_ = tree_model.tree_
    
    # Calculate statistics
    n_nodes = tree_.node_count
    n_leaves = tree_.n_leaves
    max_depth = tree_.max_depth
    
    # Node distribution by depth
    node_depth = np.zeros(n_nodes)
    is_leaves = np.zeros(n_nodes, dtype=bool)
    stack = [(0, 0)]  # (node_id, depth)
    
    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        
        is_split_node = tree_.feature[node_id] != _tree.TREE_UNDEFINED
        if is_split_node:
            stack.append((tree_.children_left[node_id], depth + 1))
            stack.append((tree_.children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    
    # Count nodes at each depth
    depth_counts = pd.Series(node_depth).value_counts().sort_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Tree Structure Overview', 'Nodes per Depth Level'),
        specs=[[{'type': 'domain'}, {'type': 'bar'}]]
    )
    
    # Pie chart: overview
    fig.add_trace(
        go.Pie(
            labels=['Internal Nodes', 'Leaf Nodes'],
            values=[n_nodes - n_leaves, n_leaves],
            hole=0.4,
            marker=dict(colors=['#3b82f6', '#10b981']),
            textinfo='label+value+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Bar chart: nodes per depth
    fig.add_trace(
        go.Bar(
            x=depth_counts.index,
            y=depth_counts.values,
            marker_color='#8b5cf6',
            text=depth_counts.values,
            textposition='outside',
            hovertemplate='<b>Depth %{x}</b><br>Nodes: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Depth Level", row=1, col=2)
    fig.update_yaxes(title_text="Number of Nodes", row=1, col=2)
    
    fig.update_layout(
        title_text=f"Decision Tree Analysis (Total Nodes: {n_nodes}, Max Depth: {max_depth})",
        height=400,
        showlegend=False,
        font=dict(size=11)
    )
    
    return fig


from sklearn.tree import _tree

def create_tree_summary_table(tree_model: DecisionTreeClassifier) -> pd.DataFrame:
    """
    Create summary table with tree statistics.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        
    Returns:
        DataFrame with tree statistics
    """
    tree_ = tree_model.tree_
    
    summary = pd.DataFrame({
        'Metric': [
            'Total Nodes',
            'Leaf Nodes',
            'Internal Nodes',
            'Maximum Depth',
            'Number of Features',
            'Number of Classes'
        ],
        'Value': [
            tree_.node_count,
            tree_.n_leaves,
            tree_.node_count - tree_.n_leaves,
            tree_.max_depth,
            tree_.n_features,
            tree_.n_classes[0]
        ]
    })
    
    return summary
