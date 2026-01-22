"""
SCIKIT-LEARN BASICS - PHASE 3
===============================
A comprehensive guide to scikit-learn fundamentals with practical examples
"""

# ============================================================================
# 1. API CONVENTIONS - THE SCIKIT-LEARN PATTERN
# ============================================================================

"""
CORE PRINCIPLE: All scikit-learn estimators follow a consistent API pattern

The Three Main Methods:
-----------------------
1. .fit(X, y)        - Train/learn from data
2. .predict(X)       - Make predictions on new data
3. .transform(X)     - Transform data (for preprocessors)

This consistency means once you learn one algorithm, you know them all!
"""

# Example: The Universal Pattern
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Every model follows the same pattern:
# Step 1: Create the model object
model = LinearRegression()

# Step 2: Fit (train) on data
# model.fit(X_train, y_train)

# Step 3: Predict on new data
# predictions = model.predict(X_test)

# This works EXACTLY the same for ANY algorithm:
# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)


# ============================================================================
# 2. DATA LOADING - BUILT-IN DATASETS
# ============================================================================

"""
Scikit-learn provides built-in datasets for practice without data cleaning
"""

from sklearn.datasets import (
    load_iris,           # Classic classification dataset (flower species)
    load_diabetes,       # Regression dataset (diabetes progression)
    load_wine,           # Classification (wine types)
    load_breast_cancer,  # Binary classification (cancer detection)
    make_classification, # Generate synthetic classification data
    make_regression      # Generate synthetic regression data
)

# Example 1: Loading a real dataset (Iris)
print("=" * 60)
print("LOADING IRIS DATASET")
print("=" * 60)

iris = load_iris()

# Understanding the dataset structure
print(f"Features (X): {iris.data.shape}")  # (150 samples, 4 features)
print(f"Target (y): {iris.target.shape}")  # (150 samples,)
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"\nFirst 5 samples:\n{iris.data[:5]}")
print(f"First 5 targets: {iris.target[:5]}")

# Example 2: Creating synthetic data
print("\n" + "=" * 60)
print("GENERATING SYNTHETIC DATA")
print("=" * 60)

from sklearn.datasets import make_classification

# Create custom dataset for experimentation
X_synthetic, y_synthetic = make_classification(
    n_samples=1000,      # Number of samples
    n_features=20,       # Number of features
    n_informative=15,    # Number of useful features
    n_redundant=5,       # Number of redundant features
    n_classes=2,         # Binary classification
    random_state=42      # For reproducibility
)

print(f"Synthetic data shape: {X_synthetic.shape}")
print(f"Class distribution: {sum(y_synthetic == 0)} vs {sum(y_synthetic == 1)}")


# ============================================================================
# 3. TRAIN-TEST SPLITTING
# ============================================================================

"""
train_test_split() is your best friend for creating training and test sets
"""

from sklearn.model_selection import train_test_split

print("\n" + "=" * 60)
print("TRAIN-TEST SPLITTING")
print("=" * 60)

# Basic split (default 75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target,
    random_state=42  # Always set this for reproducibility!
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Custom split ratio
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target,
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42
)

print(f"\nWith test_size=0.2:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Stratified split (maintains class proportions)
# VERY IMPORTANT for imbalanced datasets!
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target,
    test_size=0.2,
    stratify=iris.target,  # Keeps same class ratio in train and test
    random_state=42
)

print(f"\nStratified split - Class distribution:")
print(f"Original: {[sum(iris.target == i) for i in range(3)]}")
print(f"Training: {[sum(y_train == i) for i in range(3)]}")
print(f"Test: {[sum(y_test == i) for i in range(3)]}")


# ============================================================================
# 4. YOUR FIRST MODELS - LINEAR & LOGISTIC REGRESSION
# ============================================================================

"""
Focus on understanding the WORKFLOW, not the math behind algorithms
The workflow is the same for ALL models in scikit-learn!
"""

print("\n" + "=" * 60)
print("LINEAR REGRESSION - REGRESSION EXAMPLE")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load regression dataset
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, 
    diabetes.target,
    test_size=0.2,
    random_state=42
)

# THE STANDARD WORKFLOW:
# Step 1: Create the model
model = LinearRegression()

# Step 2: Train the model
model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"\nInterpretation: Model explains {r2*100:.1f}% of variance in diabetes progression")


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - CLASSIFICATION EXAMPLE")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Use Iris dataset for classification
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target,
    test_size=0.2,
    stratify=iris.target,
    random_state=42
)

# THE SAME WORKFLOW (notice the pattern!):
# Step 1: Create the model
classifier = LogisticRegression(max_iter=200)

# Step 2: Train the model
classifier.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = classifier.predict(X_test)

# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"\nConfusion Matrix:")
print(conf_matrix)
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# ============================================================================
# COMPLETE EXAMPLE - PUTTING IT ALL TOGETHER
# ============================================================================

print("\n" + "=" * 60)
print("COMPLETE WORKFLOW EXAMPLE")
print("=" * 60)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: {data.DESCR.split('\\n')[0]}")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# 2. Split data (80-20 split, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# 3. Create and train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
KEY TAKEAWAYS FROM PHASE 3:
============================

1. THE GOLDEN PATTERN (memorize this!):
   - model = Algorithm()
   - model.fit(X_train, y_train)
   - predictions = model.predict(X_test)
   
   This pattern works for EVERY scikit-learn algorithm!

2. ALWAYS SPLIT YOUR DATA:
   - Use train_test_split()
   - Set random_state for reproducibility
   - Use stratify for imbalanced classification
   
3. BUILT-IN DATASETS ARE YOUR FRIENDS:
   - Perfect for learning and experimentation
   - No data cleaning required
   - load_iris(), load_diabetes(), load_breast_cancer()
   - make_classification(), make_regression() for custom data

4. WORKFLOW > ALGORITHMS:
   - Master the workflow first
   - The same workflow applies to ALL algorithms
   - Understanding one algorithm = understanding them all (API-wise)
   
5. START SIMPLE:
   - Linear Regression for regression problems
   - Logistic Regression for classification problems
   - Focus on the process, not the math

NEXT STEPS:
-----------
Once you're comfortable with this workflow, you can easily swap in
different algorithms (Random Forest, SVM, etc.) and they'll work
the exact same way!
"""