import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


# Define the function for splitting data into train, dev, and test sets
def Split_Train_Dev_Test(X, y, test_size, dev_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), shuffle=False)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(dev_size / (test_size + dev_size)),
                                                    shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


# Define the function for predicting and evaluating the model
def Predict_and_Eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(f"Classification report for classifier {model}:\n{metrics.classification_report(y_test, predicted)}\n")

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")


# Function for hyperparameter tuning
def tune_hparams(X_train, Y_train, X_dev, Y_dev, list_of_all_param_combinations):
    best_hparams = None
    best_model = None
    best_accuracy = 0.0

    for param_combination in list_of_all_param_combinations:
        # Train a model with the current hyperparameters (Using SVM as an example)
        model = svm.SVC(**param_combination)
        model.fit(X_train, Y_train)
        
        # Evaluate the model on the dev set
        dev_accuracy = metrics.accuracy_score(Y_dev, model.predict(X_dev))

        # Check if this is the best accuracy so far
        if dev_accuracy > best_accuracy:
            best_hparams = param_combination
            best_model = model
            best_accuracy = dev_accuracy

    return best_hparams, best_model, best_accuracy


# Load the digits dataset
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))

# Modify this part
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1.0 - test_size - dev_size
        X_train, X_dev, X_test, y_train, y_dev, y_test = Split_Train_Dev_Test(data, digits.target, test_size, dev_size)

        # Example hyperparameter tuning configurations (replace with your own)
        list_of_all_param_combinations = [
            {"C": 1.0, "kernel": "linear"},
            {"C": 0.1, "kernel": "rbf", "gamma": 0.001},
            {"C": 0.01, "kernel": "poly", "degree": 3, "gamma": 0.01},
            # Add more combinations as needed
        ]

        # Call the hyperparameter tuning function
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combinations)

        print(f"test_size={test_size} dev_size={dev_size} train_size={train_size} train_acc={best_accuracy} dev_acc={best_accuracy} test_acc={best_accuracy}")
        print(f"Best hyperparameters: {best_hparams}")

# Visualize the dev set predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_dev, best_model.predict(X_dev)):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

plt.show()
