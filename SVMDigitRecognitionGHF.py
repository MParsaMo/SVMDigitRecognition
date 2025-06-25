import matplotlib.pyplot as plt # Standard way to import pyplot
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np # Used by scikit-learn internally and for array manipulation
import pandas as pd # Included for general data science context, though not strictly used for DataFrames here

# It's generally better to remove `matplotlib.use('TkAgg')` in a GitHub-friendly script
# as it can cause issues on different systems. Matplotlib usually selects an appropriate backend.

def load_digits_data():
    """
    Loads the handwritten digits dataset from scikit-learn.

    This dataset consists of 1797 8x8 pixel grayscale images of handwritten digits (0-9).
    Each image is a numerical representation of an integer (0-9).

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data, target,
                             images, and description.
    """
    print("Loading handwritten digits dataset...")
    digits = datasets.load_digits()

    print("\n--- Digits Data Overview ---")
    print(f"Number of samples: {len(digits.images)}")
    print(f"Image shape: {digits.images[0].shape} (8x8 pixels)")
    print(f"Total features per image (after flattening): {digits.images[0].size}")
    print("Target classes (digits):", np.unique(digits.target))
    print("\nFirst sample (image and target):")
    plt.figure(figsize=(2, 2))
    plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"Target: {digits.target[0]}")
    plt.axis('off')
    plt.show(block=False) # Use block=False to prevent blocking script execution immediately
    plt.pause(0.5) # Short pause to display the plot
    plt.close() # Close the figure

    return digits

def preprocess_data(digits_dataset):
    """
    Preprocesses the digits data by reshaping the 2D image arrays into 1D feature vectors.

    Machine learning models typically expect input features to be in a 2D array
    where each row is a sample and each column is a feature.

    Args:
        digits_dataset (sklearn.utils.Bunch): The loaded digits dataset.

    Returns:
        tuple: A tuple containing (features (X), target (y)).
    """
    print("\n--- Preprocessing Data ---")
    # Reshape each 8x8 image into a one-dimensional vector of 64 features.
    # `len(digits_dataset.images)` is the number of samples.
    # `-1` tells NumPy to automatically calculate the size of the second dimension
    # based on the total number of elements. So, each image becomes a row of 64 pixel values.
    features = digits_dataset.images.reshape((len(digits_dataset.images), -1))
    target = digits_dataset.target

    print(f"Original image shape: {digits_dataset.images[0].shape}")
    print(f"Reshaped features shape: {features.shape}") # Should be (n_samples, 64)
    return features, target

def split_dataset(features, target, train_size=0.7, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The preprocessed feature data.
        target (numpy.ndarray): The target labels.
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Ensures reproducibility.

    Returns:
        tuple: A tuple containing (features_train, features_test, target_train, target_test).
    """
    print(f"\nSplitting data into training ({train_size*100:.0f}%) and testing ({(1-train_size)*100:.0f}%) sets...")
    # `stratify=target` ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset. This is important for classification tasks.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, train_size=train_size, random_state=random_state, stratify=target
    )
    print(f"Training set size: {len(features_train)} samples")
    print(f"Testing set size: {len(features_test)} samples")
    return features_train, features_test, target_train, target_test

def train_svm_classifier(features_train, target_train, gamma_val=0.001):
    """
    Trains a Support Vector Machine (SVM) classifier.

    Args:
        features_train (numpy.ndarray): Training features.
        target_train (numpy.ndarray): Training target.
        gamma_val (float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
                           It defines how much influence a single training example has.
                           Small gamma: large influence, large gamma: small influence.

    Returns:
        sklearn.svm.SVC: The trained SVM classifier model.
    """
    print(f"\n--- Training SVM Classifier with gamma={gamma_val} ---")
    # SVC (Support Vector Classifier) is a classification method from the SVM family.
    # The `gamma` parameter controls the influence of individual training samples.
    # A smaller `gamma` means a larger region of influence, leading to a smoother decision boundary.
    # A larger `gamma` means a smaller region of influence, leading to a more complex decision boundary.
    model = svm.SVC(gamma=gamma_val, random_state=42) # Add random_state for reproducibility
    model.fit(features_train, target_train)
    print("SVM model training complete.")
    return model

def evaluate_model(fitted_model, features_test, target_test):
    """
    Evaluates the trained SVM model on the test data and prints performance metrics.

    Args:
        fitted_model (sklearn.svm.SVC): The trained SVM model.
        features_test (numpy.ndarray): Testing features.
        target_test (numpy.ndarray): Testing target (true labels).
    """
    print("\n--- Evaluating Model Performance ---")
    prediction = fitted_model.predict(features_test)

    # Accuracy Score: Proportion of correctly classified instances.
    accuracy = accuracy_score(target_test, prediction)
    print(f"Accuracy Score: {accuracy:.4f}")

    # Confusion Matrix: Summarizes classification model performance.
    # Rows are actual classes, columns are predicted classes.
    print("\nConfusion Matrix:")
    # Using pd.DataFrame to make confusion matrix more readable with labels
    cm = confusion_matrix(target_test, prediction)
    # Get unique target names for confusion matrix labels
    target_names = [str(i) for i in sorted(np.unique(target_test))]
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Classification Report: Provides precision, recall, f1-score for each class.
    print("\nClassification Report:")
    print(classification_report(target_test, prediction, target_names=target_names))

def predict_and_display_sample(model, digits_dataset, sample_index=-2):
    """
    Predicts the digit for a specific sample image and displays it.

    Args:
        model (sklearn.svm.SVC): The trained SVM model.
        digits_dataset (sklearn.utils.Bunch): The original digits dataset.
        sample_index (int): The index of the sample to predict (e.g., -1 for last, -2 for second last).
    """
    print(f"\n--- Predicting and Displaying Sample Image (index: {sample_index}) ---")
    # Get the image and its corresponding features
    sample_image = digits_dataset.images[sample_index]
    sample_features = digits_dataset.data[sample_index].reshape(1, -1) # Reshape for single sample prediction

    # Make the prediction
    predicted_digit = model.predict(sample_features)[0]

    print(f"Actual digit: {digits_dataset.target[sample_index]}")
    print(f"Predicted digit: {predicted_digit}")

    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(sample_image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"Predicted: {predicted_digit}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Define parameters
    TRAIN_TEST_SPLIT_RATIO = 0.7
    SVM_GAMMA_VALUE = 0.001
    RANDOM_SEED = 42 # For reproducibility of splits and model training

    # 1. Load the Digits Dataset
    digits_data = load_digits_data()

    # 2. Preprocess Data (Flatten images into feature vectors)
    X, y = preprocess_data(digits_data)

    # 3. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_dataset(
        X, y,
        train_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )

    # 4. Train the SVM Classifier
    svm_classifier = train_svm_classifier(X_train, y_train, gamma_val=SVM_GAMMA_VALUE)

    # 5. Evaluate the Model
    evaluate_model(svm_classifier, X_test, y_test)

    # 6. Predict and Display a Specific Sample Image
    predict_and_display_sample(svm_classifier, digits_data, sample_index=-2)

    print("\nScript execution complete.")
