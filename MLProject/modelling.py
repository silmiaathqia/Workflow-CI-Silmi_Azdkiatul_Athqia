import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
import argparse
import os
import dagshub
from datetime import datetime

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments for MLflow Project"""
    parser = argparse.ArgumentParser(description='Train Worker Productivity MLP Model')
    parser.add_argument('--max_iter', type=int, default=500, help='Maximum iterations for training')
    parser.add_argument('--alpha', type=float, default=0.001, help='Alpha regularization parameter')
    parser.add_argument('--hidden_layer_sizes', type=str, default='128,64,32', help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--learning_rate_init', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=str, default='auto', help='Batch size for training')
    parser.add_argument('--solver', type=str, default='adam', help='Solver for optimization')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    return parser.parse_args()

def parse_hidden_layers(hidden_str):
    """Convert comma-separated string to tuple of integers"""
    return tuple(map(int, hidden_str.split(',')))

def setup_dagshub():
    """Setup DagsHub connection for MLflow tracking"""
    try:
        dagshub.init(
            repo_owner=os.getenv('DAGSHUB_REPO_OWNER', 'silmiaathqia'),
            repo_name=os.getenv('DAGSHUB_REPO_NAME', 'Worker-Productivity-MLflow'),
            mlflow=True
        )
        print("DagsHub connection established successfully!")
    except Exception as e:
        print(f"Warning: Could not connect to DagsHub: {e}")
        print("Falling back to local MLflow tracking...")

def load_processed_data():
    """Load preprocessed data from kriteria 1"""
    try:
        # Try different possible paths
        data_paths = [
            'processed_data/',
            'data/',
            './',
            '../processed_data/',
        ]
        
        train_data = None
        val_data = None
        test_data = None
        
        for path in data_paths:
            try:
                train_data = pd.read_csv(f'{path}data_train.csv')
                val_data = pd.read_csv(f'{path}data_validation.csv')
                test_data = pd.read_csv(f'{path}data_test.csv')
                print(f"Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if train_data is None:
            raise FileNotFoundError("Could not find data files in any expected location")
        
        # Separate features and target
        X_train = train_data.drop('productivity_label_encoded', axis=1)
        y_train = train_data['productivity_label_encoded']
        
        X_val = val_data.drop('productivity_label_encoded', axis=1)
        y_val = val_data['productivity_label_encoded']
        
        X_test = test_data.drop('productivity_label_encoded', axis=1)
        y_test = test_data['productivity_label_encoded']
        
        print(f"Data berhasil dimuat!")
        print(f"Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Kelas target: {sorted(y_train.unique())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_mlp_model(args):
    """Create MLP model using sklearn for worker productivity classification"""
    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=args.activation,
        solver=args.solver,
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate='constant',
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        early_stopping=True,
        verbose=True
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names=['High', 'Low', 'Medium']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def plot_metrics_comparison(metrics_dict):
    """Plot metrics comparison"""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def plot_training_history(model):
    """Plot training loss history"""
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, 'b-', linewidth=2)
        plt.title('Training Loss History')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    return None

def calculate_additional_metrics(y_true, y_pred, y_pred_proba):
    """Calculate additional metrics beyond autolog"""
    from sklearn.metrics import roc_auc_score, log_loss, matthews_corrcoef
    from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
    
    # Multi-class metrics
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_score = None
    
    try:
        logloss = log_loss(y_true, y_pred_proba)
    except:
        logloss = None
    
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Class-wise metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    additional_metrics = {
        'matthews_corrcoef': mcc,
        'cohen_kappa_score': kappa,
        'balanced_accuracy': balanced_acc,
        'precision_class_0': precision_per_class[0],
        'precision_class_1': precision_per_class[1],
        'precision_class_2': precision_per_class[2],
        'recall_class_0': recall_per_class[0],
        'recall_class_1': recall_per_class[1],
        'recall_class_2': recall_per_class[2],
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'f1_class_2': f1_per_class[2]
    }
    
    if auc_score is not None:
        additional_metrics['roc_auc_weighted'] = auc_score
    
    if logloss is not None:
        additional_metrics['log_loss'] = logloss
    
    return additional_metrics

def train_model_with_mlflow(args):
    """Train MLP model with enhanced MLflow tracking"""
    
    # Setup DagsHub if possible
    setup_dagshub()
    
    # Set MLflow experiment
    experiment_name = "Worker_Productivity_Classification_MLProject"
    mlflow.set_experiment(experiment_name)
    
    # Load data
    data = load_processed_data()
    if data is None:
        raise ValueError("Could not load data")
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Combine train and validation for sklearn MLPClassifier
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data scaling completed!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"MLProject_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters (MANUAL LOGGING - not autolog)
        model_params = {
            'hidden_layer_sizes': args.hidden_layer_sizes,
            'activation': args.activation,
            'solver': args.solver,
            'alpha': args.alpha,
            'max_iter': args.max_iter,
            'learning_rate_init': args.learning_rate_init,
            'batch_size': args.batch_size,
            'early_stopping': True,
            'n_iter_no_change': 10,
            'random_state': 42,
            'train_size': X_train_scaled.shape[0],
            'test_size': X_test_scaled.shape[0],
            'n_features': X_train_scaled.shape[1],
            'n_classes': len(np.unique(y_train_full))
        }
        
        mlflow.log_params(model_params)
        
        # Create and train model
        model = create_mlp_model(args)
        
        print("\nStarting training...")
        print("Model Configuration:")
        print(f"Hidden layers: {parse_hidden_layers(args.hidden_layer_sizes)}")
        print(f"Activation: {args.activation}")
        print(f"Solver: {args.solver}")
        print(f"Alpha (L2 reg): {args.alpha}")
        print(f"Max iterations: {args.max_iter}")
        
        # Train model
        model.fit(X_train_scaled, y_train_full)
        
        print(f"\nTraining completed!")
        print(f"Number of iterations: {model.n_iter_}")
        print(f"Final loss: {model.loss_:.6f}")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate standard metrics (MANUAL LOGGING)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate additional metrics (ADVANCE requirement)
        additional_metrics = calculate_additional_metrics(y_test, y_pred, y_pred_proba)
        
        # Log all metrics manually
        all_metrics = {
            'test_accuracy': accuracy,
            'test_precision_weighted': precision,
            'test_recall_weighted': recall,
            'test_f1_score_weighted': f1,
            'training_loss': model.loss_,
            'n_iterations': model.n_iter_,
            'convergence_status': 'converged' if model.n_iter_ < args.max_iter else 'max_iter_reached'
        }
        
        # Add additional metrics
        all_metrics.update(additional_metrics)
        
        mlflow.log_metrics(all_metrics)
        
        print(f"\nTest Results:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Matthews Correlation Coefficient: {additional_metrics['matthews_corrcoef']:.4f}")
        print(f"Cohen's Kappa: {additional_metrics['cohen_kappa_score']:.4f}")
        print(f"Balanced Accuracy: {additional_metrics['balanced_accuracy']:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        class_names = ['High', 'Low', 'Medium']
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Create and log visualizations
        # 1. Confusion Matrix
        cm_plot = plot_confusion_matrix(y_test, y_pred, class_names)
        cm_plot.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        # 2. Metrics Comparison
        metrics_plot = plot_metrics_comparison({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Balanced Acc': additional_metrics['balanced_accuracy']
        })
        metrics_plot.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('metrics_comparison.png')
        plt.close()
        
        # 3. Training History
        training_plot = plot_training_history(model)
        if training_plot:
            training_plot.savefig('training_history.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('training_history.png')
            plt.close()
        
        # 4. Class-wise Performance
        class_metrics = {
            'High Productivity': [additional_metrics['precision_class_0'], 
                                additional_metrics['recall_class_0'], 
                                additional_metrics['f1_class_0']],
            'Low Productivity': [additional_metrics['precision_class_1'], 
                               additional_metrics['recall_class_1'], 
                               additional_metrics['f1_class_1']],
            'Medium Productivity': [additional_metrics['precision_class_2'], 
                                  additional_metrics['recall_class_2'], 
                                  additional_metrics['f1_class_2']]
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(class_metrics))
        width = 0.25
        
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        for i, metric in enumerate(metrics_names):
            values = [class_metrics[cls][i] for cls in class_metrics.keys()]
            ax.bar(x + i*width, values, width, label=metric, color=colors[i])
        
        ax.set_xlabel('Productivity Classes')
        ax.set_ylabel('Score')
        ax.set_title('Class-wise Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_metrics.keys())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('class_wise_performance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('class_wise_performance.png')
        plt.close()
        
        # Log model with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train_scaled, y_pred)
        
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            registered_model_name="WorkerProductivityMLP",
            pip_requirements=['scikit-learn', 'pandas', 'numpy']
        )
        
        # Save and log scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact('scaler.pkl')
        
        # Save model info
        model_info = {
            'model_type': 'MLPClassifier',
            'sklearn_version': mlflow.sklearn.__version__,
            'training_timestamp': datetime.now().isoformat(),
            'feature_names': list(X_train.columns),
            'target_classes': class_names,
            'model_params': model_params,
            'performance_metrics': all_metrics,
            'docker_image': 'silmiathqia/worker-productivity-mlp:latest'
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact('model_info.json')
        
        print("\nModel artifacts saved:")
        print("- Model: Logged to MLflow with signature")
        print("- Scaler: scaler.pkl")
        print("- Visualizations: confusion_matrix.png, metrics_comparison.png, etc.")
        print("- Model Info: model_info.json")
        
        # Log additional tags
        mlflow.set_tags({
            'model_type': 'MLPClassifier',
            'use_case': 'worker_productivity_classification',
            'environment': 'MLProject',
            'deployment_ready': 'true',
            'docker_compatible': 'true'
        })
        
        print(f"\nMLProject training completed successfully!")
        print(f"Final Model Performance:")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- F1-Score: {f1:.4f}")
        print(f"- Balanced Accuracy: {additional_metrics['balanced_accuracy']:.4f}")
        print(f"- Matthews Correlation: {additional_metrics['matthews_corrcoef']:.4f}")
        print(f"- Training Iterations: {model.n_iter_}")
        
        return model, scaler, all_metrics

def make_prediction_sample():
    """Make sample prediction to test the model"""
    try:
        # Load the latest model from MLflow
        model_uri = "models:/WorkerProductivityMLP/latest"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Create sample data (you should replace this with actual feature values)
        sample_data = pd.DataFrame({
            'feature_1': [0.5],
            'feature_2': [0.3],
            # Add all your features here based on your actual dataset
        })
        
        # Scale and predict
        sample_scaled = scaler.transform(sample_data)
        prediction = model.predict(sample_scaled)
        prediction_proba = model.predict_proba(sample_scaled)
        
        class_names = ['High', 'Low', 'Medium']
        
        print(f"\nSample Prediction:")
        print(f"Predicted Class: {class_names[prediction[0]]}")
        print("Class Probabilities:")
        for i, prob in enumerate(prediction_proba[0]):
            print(f"  {class_names[i]}: {prob:.4f}")
            
    except Exception as e:
        print(f"Could not make sample prediction: {e}")

if __name__ == "__main__":
    print("WORKER PRODUCTIVITY CLASSIFICATION - MLflow PROJECT")
    print("=" * 65)
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"Training with parameters:")
    print(f"- Hidden Layers: {args.hidden_layer_sizes}")
    print(f"- Max Iterations: {args.max_iter}")
    print(f"- Alpha: {args.alpha}")
    print(f"- Learning Rate: {args.learning_rate_init}")
    print(f"- Solver: {args.solver}")
    print(f"- Activation: {args.activation}")
    print("-" * 65)
    
    # Train model
    try:
        results = train_model_with_mlflow(args)
        
        if results:
            model, scaler, metrics = results
            print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"📊 Final Performance Summary:")
            print(f"   - Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"   - F1-Score: {metrics['test_f1_score_weighted']:.4f}")
            print(f"   - Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"   - Training Iterations: {metrics['n_iterations']}")
            
            # Test sample prediction
            make_prediction_sample()
            
            print(f"\n✅ Model ready for Docker deployment!")
            print(f"🐳 Docker command: mlflow models build-docker -m 'models:/WorkerProductivityMLP/latest' -n 'worker-productivity-mlp'")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise e