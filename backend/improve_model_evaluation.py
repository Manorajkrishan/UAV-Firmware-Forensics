"""
Improved Model Evaluation to Prevent Overfitting
Adds cross-validation, confusion matrix, and better metrics
"""
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model_with_cv(model, X, y, cv_folds=5):
    """
    Evaluate model with cross-validation to detect overfitting
    
    Returns:
        dict with cv_scores, mean_accuracy, std_accuracy, and overfitting_warning
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    
    # Detect overfitting: if accuracy is too high (>0.99) or std is very low
    overfitting_warning = False
    if mean_accuracy > 0.99:
        overfitting_warning = True
    if std_accuracy < 0.01 and mean_accuracy > 0.95:
        overfitting_warning = True
    
    return {
        'cv_scores': cv_scores.tolist(),
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'min_accuracy': cv_scores.min(),
        'max_accuracy': cv_scores.max(),
        'overfitting_warning': overfitting_warning
    }

def comprehensive_model_evaluation(model, X_train, y_train, X_test, y_test, model_name='Model'):
    """
    Comprehensive evaluation with multiple metrics
    """
    # Train predictions
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Get probabilities if available
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_test_proba = None
    
    # Calculate metrics
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Overfitting check: large gap between train and test accuracy
    accuracy_gap = train_accuracy - test_accuracy
    overfitting_warning = accuracy_gap > 0.15  # More than 15% gap
    
    # Cross-validation
    cv_results = evaluate_model_with_cv(model, X_train, y_train)
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'accuracy_gap': accuracy_gap,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'overfitting_warning': overfitting_warning or cv_results['overfitting_warning'],
        'cv_mean_accuracy': cv_results['mean_accuracy'],
        'cv_std_accuracy': cv_results['std_accuracy'],
        'recommended_accuracy': min(test_accuracy, cv_results['mean_accuracy'])  # Use lower of test or CV
    }
    
    # Print report
    print(f"\n{'='*50}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*50}")
    print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Accuracy Gap:   {accuracy_gap:.4f} ({'⚠ OVERFITTING' if overfitting_warning else '✓ OK'})")
    print(f"\nCross-Validation:")
    print(f"  Mean: {cv_results['mean_accuracy']:.4f} ({cv_results['mean_accuracy']*100:.2f}%)")
    print(f"  Std:  {cv_results['std_accuracy']:.4f}")
    print(f"  Range: [{cv_results['min_accuracy']:.4f}, {cv_results['max_accuracy']:.4f}]")
    print(f"\nOther Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    if overfitting_warning or cv_results['overfitting_warning']:
        print(f"\n⚠ WARNING: Model may be overfitting!")
        print(f"  Recommended accuracy: {results['recommended_accuracy']:.4f} ({results['recommended_accuracy']*100:.2f}%)")
    print(f"{'='*50}\n")
    
    return results

