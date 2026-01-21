"""
Interactive Decision Trees Dashboard
A Streamlit application for training, visualizing, and analyzing decision tree classifiers.

Developed by Álvaro Andrés De Lamo
University of Silesia - Decision and Association Rules Course
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Import custom utilities
from utils.preprocessing import load_data, preprocess_data, get_data_summary
from utils.decision_trees import (
    train_decision_tree, extract_rules, format_rules_text, get_feature_importance
)
from utils.visualization import (
    plot_tree_matplotlib, plot_confusion_matrix, plot_feature_importance,
    plot_metrics_comparison, plot_class_distribution, plot_tree_depth_analysis,
    create_tree_summary_table
)

# Page configuration
st.set_page_config(
    page_title="Decision Trees Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #555555 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        font-weight: 600;
        margin-top: 1.5rem;
    }
    h3 {
        font-weight: 600;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Header
st.title("🌳 Interactive Decision Trees Dashboard")
st.markdown("**Train, visualize, and analyze decision tree classifiers with automatic rule extraction**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Example Dataset"],
        help="Upload your own CSV or use the included Cars dataset"
    )
    
    df = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with features and a target column"
        )
        
        if uploaded_file is not None:
            df, error = load_data(uploaded_file)
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ File loaded successfully!")
                st.session_state.data_loaded = True
    
    else:  # Use example dataset
        example_path = Path(__file__).parent / "data" / "cars.csv"
        if example_path.exists():
            df, error = load_data(str(example_path))
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Example dataset loaded!")
                st.session_state.data_loaded = True
        else:
            st.error("❌ Example dataset not found")
    
    # Data preprocessing settings
    if st.session_state.data_loaded and df is not None:
        st.markdown("---")
        st.subheader("🔧 Preprocessing")
        
        # Target column selection
        target_column = st.selectbox(
            "Select target column:",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,
            help="Choose the column to predict"
        )
        
        # Missing values strategy
        missing_strategy = st.selectbox(
            "Handle missing values:",
            options=['mean', 'median', 'mode', 'drop'],
            help="Strategy for handling missing values"
        )
        
        # Model hyperparameters
        st.markdown("---")
        st.subheader("🎯 Model Parameters")
        
        test_size = st.slider(
            "Test set size:",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Proportion of dataset for testing"
        )
        
        criterion = st.selectbox(
            "Split criterion:",
            options=['gini', 'entropy'],
            help="Measure of split quality"
        )
        
        max_depth = st.slider(
            "Maximum depth:",
            min_value=1,
            max_value=20,
            value=6,
            help="Maximum depth of the tree (None = unlimited)"
        )
        max_depth = None if max_depth == 20 else max_depth
        
        min_samples_split = st.slider(
            "Min samples to split:",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum samples required to split a node"
        )
        
        min_samples_leaf = st.slider(
            "Min samples in leaf:",
            min_value=1,
            max_value=20,
            value=5,
            help="Minimum samples required in a leaf node"
        )
        
        # Train button
        st.markdown("---")
        train_button = st.button("🚀 Train Decision Tree", type="primary", use_container_width=True)

# Main content
if not st.session_state.data_loaded or df is None:
    st.info("👈 Please upload a CSV file or select the example dataset from the sidebar to get started.")
    
    # Display example instructions
    with st.expander("ℹ️ How to use this dashboard"):
        st.markdown("""
        ### Getting Started
        1. **Upload Data**: Use the sidebar to upload your CSV file or use the example dataset
        2. **Configure**: Select target column and preprocessing options
        3. **Set Parameters**: Adjust decision tree hyperparameters
        4. **Train**: Click "Train Decision Tree" to build the model
        5. **Analyze**: Explore visualizations, metrics, and extracted rules
        
        ### Data Requirements
        - CSV format with header row
        - At least one feature column and one target column
        - Supports both numeric and categorical features
        - Missing values will be handled automatically
        
        ### Features
        - 📊 Interactive tree visualization
        - 📈 Performance metrics and confusion matrix
        - 🔍 Feature importance analysis
        - 📝 Automatic rule extraction
        - 💾 Export results to CSV
        """)

else:
    # Display data info
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    summary = get_data_summary(df)
    
    with col1:
        st.metric("Total Rows", f"{summary['n_rows']:,}")
    with col2:
        st.metric("Total Columns", summary['n_columns'])
    with col3:
        st.metric("Missing Values", summary['missing_values'])
    with col4:
        st.metric("Memory Usage", f"{summary['memory_usage']:.2f} MB")
    
    # Data preview
    with st.expander("👀 Preview Data", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(df)} rows")
    
    # Train model when button is clicked
    if train_button:
        with st.spinner("🔄 Training decision tree..."):
            try:
                # Preprocess data
                X, y, preprocessing_info = preprocess_data(
                    df, 
                    target_column=target_column,
                    handle_missing=missing_strategy
                )
                
                # Train model
                results = train_decision_tree(
                    X, y,
                    test_size=test_size,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )
                
                # Store in session state
                st.session_state.results = results
                st.session_state.preprocessing_info = preprocessing_info
                st.session_state.model_trained = True
                
                st.success("✅ Model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.session_state.model_trained = False
    
    # Display results if model is trained
    if st.session_state.model_trained:
        results = st.session_state.results
        preprocessing_info = st.session_state.preprocessing_info
        
        st.markdown("---")
        st.header("📈 Model Results")
        
        # Performance metrics
        st.subheader("🎯 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        metrics = results['metrics']
        
        with col1:
            st.metric(
                "Test Accuracy",
                f"{metrics['test_accuracy']:.3f}",
                delta=f"{metrics['test_accuracy'] - metrics['train_accuracy']:.3f}"
            )
        with col2:
            st.metric(
                "Test Precision",
                f"{metrics['test_precision']:.3f}",
                delta=f"{metrics['test_precision'] - metrics['train_precision']:.3f}"
            )
        with col3:
            st.metric(
                "Test Recall",
                f"{metrics['test_recall']:.3f}",
                delta=f"{metrics['test_recall'] - metrics['train_recall']:.3f}"
            )
        with col4:
            st.metric(
                "Test F1 Score",
                f"{metrics['test_f1']:.3f}",
                delta=f"{metrics['test_f1'] - metrics['train_f1']:.3f}"
            )
        
        # Metrics comparison chart
        st.plotly_chart(
            plot_metrics_comparison(metrics),
            use_container_width=True
        )
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Confusion Matrix")
            class_names = [str(c) for c in sorted(results['y_test'].unique())]
            st.plotly_chart(
                plot_confusion_matrix(metrics['confusion_matrix'], class_names),
                use_container_width=True
            )
        
        with col2:
            st.subheader("🎲 Class Distribution")
            st.plotly_chart(
                plot_class_distribution(results['y_test'], "Test Set Class Distribution"),
                use_container_width=True
            )
        
        # Decision Tree Visualization
        st.markdown("---")
        st.subheader("🌳 Decision Tree Visualization")
        
        # Tree summary
        tree_summary = create_tree_summary_table(results['model'])
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(tree_summary, use_container_width=True, hide_index=True)
        
        with col2:
            st.plotly_chart(
                plot_tree_depth_analysis(results['model']),
                use_container_width=True
            )
        
        # Tree plot
        with st.expander("🔍 View Full Tree Diagram", expanded=False):
            fig = plot_tree_matplotlib(
                results['model'],
                results['feature_names'],
                class_names
            )
            st.pyplot(fig)
        
        # Feature Importance
        st.markdown("---")
        st.subheader("⭐ Feature Importance")
        
        importance_df = get_feature_importance(results['model'], results['feature_names'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(
                plot_feature_importance(importance_df, top_n=15),
                use_container_width=True
            )
        
        with col2:
            st.dataframe(
                importance_df.head(15),
                use_container_width=True,
                hide_index=True
            )
        
        # Decision Rules
        st.markdown("---")
        st.subheader("📝 Extracted Decision Rules")
        
        rules = extract_rules(results['model'], results['feature_names'], class_names)
        rules_text = format_rules_text(rules)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.text_area(
                f"Decision Rules ({len(rules)} rules extracted)",
                rules_text,
                height=400
            )
        
        with col2:
            st.info(f"""
            **Rule Statistics:**
            - Total Rules: {len(rules)}
            - Average Depth: {np.mean([r['depth'] for r in rules]):.1f}
            - Average Purity: {np.mean([r['purity'] for r in rules]):.2%}
            - Average Support: {np.mean([r['samples'] for r in rules]):.0f} samples
            """)
            
            # Download rules
            st.download_button(
                label="📥 Download Rules (TXT)",
                data=rules_text,
                file_name="decision_rules.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Export Results
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export predictions
            predictions_df = pd.DataFrame({
                'True_Label': results['y_test'],
                'Predicted_Label': results['y_test_pred']
            })
            csv_predictions = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv_predictions,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export feature importance
            csv_importance = importance_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Feature Importance (CSV)",
                data=csv_importance,
                file_name="feature_importance.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Export model summary
            summary_text = f"""Decision Tree Model Summary
================================

Dataset Information:
- Total Samples: {preprocessing_info['original_shape'][0]}
- Total Features: {preprocessing_info['original_shape'][1] - 1}
- Target Column: {target_column}

Model Parameters:
- Criterion: {criterion}
- Max Depth: {max_depth if max_depth else 'Unlimited'}
- Min Samples Split: {min_samples_split}
- Min Samples Leaf: {min_samples_leaf}
- Test Size: {test_size:.0%}

Performance Metrics:
- Test Accuracy: {metrics['test_accuracy']:.4f}
- Test Precision: {metrics['test_precision']:.4f}
- Test Recall: {metrics['test_recall']:.4f}
- Test F1 Score: {metrics['test_f1']:.4f}

Tree Structure:
- Total Nodes: {results['model'].tree_.node_count}
- Leaf Nodes: {results['model'].tree_.n_leaves}
- Maximum Depth: {results['model'].tree_.max_depth}
"""
            st.download_button(
                label="📥 Download Summary (TXT)",
                data=summary_text,
                file_name="model_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p><strong>Decision Trees Dashboard</strong> | Developed by Álvaro Andrés De Lamo</p>
    <p>University of Silesia - Decision and Association Rules Course 🇵🇱</p>
</div>
""", unsafe_allow_html=True)
