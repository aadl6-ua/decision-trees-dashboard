"""
Decision Tree modeling utilities.
Handles training, rule extraction, and evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import List, Tuple, Dict, Any, Optional


def train_decision_tree(X: pd.DataFrame,
                        y: pd.Series,
                        test_size: float = 0.3,
                        criterion: str = 'gini',
                        max_depth: Optional[int] = None,
                        min_samples_split: int = 2,
                        min_samples_leaf: int = 1,
                        random_state: int = 42) -> Dict[str, Any]:
    """
    Train a decision tree classifier and return model + metrics.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of dataset for testing
        criterion: Split criterion ('gini' or 'entropy')
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf node
        random_state: Random seed
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
    )
    
    # Train model
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'classification_report': classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    }
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics,
        'feature_names': list(X.columns),
        'target_name': y.name
    }


def extract_rules(tree_model: DecisionTreeClassifier,
                 feature_names: List[str],
                 class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extract decision rules from trained tree.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        
    Returns:
        List of dictionaries, each representing a rule
    """
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, depth, conditions):
        """Recursively traverse tree to extract rules."""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # Not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child (<=)
            left_conditions = conditions + [f"{name} <= {threshold:.3f}"]
            recurse(tree_.children_left[node], depth + 1, left_conditions)
            
            # Right child (>)
            right_conditions = conditions + [f"{name} > {threshold:.3f}"]
            recurse(tree_.children_right[node], depth + 1, right_conditions)
        else:  # Leaf node
            # Get class distribution
            value = tree_.value[node][0]
            total_samples = tree_.n_node_samples[node]
            class_idx = np.argmax(value)
            
            # Class distribution
            class_dist = {
                class_names[i]: int(value[i])
                for i in range(len(class_names))
                if value[i] > 0
            }
            
            rule = {
                'conditions': conditions,
                'predicted_class': class_names[class_idx],
                'samples': total_samples,
                'class_distribution': class_dist,
                'purity': value[class_idx] / total_samples,
                'depth': depth
            }
            rules.append(rule)
    
    recurse(0, 0, [])
    return rules


def format_rules_text(rules: List[Dict[str, Any]]) -> str:
    """
    Format rules as human-readable text.
    
    Args:
        rules: List of rule dictionaries
        
    Returns:
        Formatted string with all rules
    """
    text_lines = []
    for i, rule in enumerate(rules, 1):
        conditions_str = " AND ".join(rule['conditions']) if rule['conditions'] else "True"
        dist_str = ", ".join([f"{cls}: {count}" for cls, count in rule['class_distribution'].items()])
        
        text_lines.append(f"Rule {i}:")
        text_lines.append(f"  IF {conditions_str}")
        text_lines.append(f"  THEN class = {rule['predicted_class']}")
        text_lines.append(f"  Support: {rule['samples']} samples")
        text_lines.append(f"  Purity: {rule['purity']:.2%}")
        text_lines.append(f"  Distribution: {dist_str}")
        text_lines.append("")
    
    return "\n".join(text_lines)


def get_feature_importance(tree_model: DecisionTreeClassifier,
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance scores.
    
    Args:
        tree_model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and importance scores
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': tree_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df
