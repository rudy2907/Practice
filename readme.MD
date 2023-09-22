# instruction
install conda on your OS 

# how to setup
conda create -n digits python=3.9
conda activate digits
pip install -r requirements.txt


# how to run
python plot_digits_classification.py
# how to run
python plot_digits_classification.py
# About the code
Certainly! This code is an example of using scikit-learn to recognize hand-written digits from the digits dataset using a support vector classifier (SVC). Let's break down the code step by step:

1. **Import Libraries:**
   - `matplotlib.pyplot` is imported for visualization.
   - `datasets`, `metrics`, and `svm` modules are imported from scikit-learn for working with datasets, evaluation metrics, and support vector machines (SVMs) respectively.
   - `train_test_split` is imported for splitting the dataset.

2. **Data Splitting Function (`split_train_dev_test`):**
   - This function takes the original data `X` and target labels `y`, along with `test_size` and `dev_size` as parameters.
   - It performs two successive calls to `train_test_split` to split the data into train, development (dev), and test sets according to the specified sizes.
   - The function returns the split datasets for training, dev, and test.

3. **Prediction and Evaluation Function (`predict_and_eval`):**
   - This function takes a trained model (`model`), the `X_test` data, and the corresponding `y_test` labels as parameters.
   - It uses the trained model to predict the labels for the test data and computes various classification metrics using `metrics.classification_report`.
   - It also constructs a confusion matrix display and prints out the confusion matrix itself.
   - The function then rebuilds the classification report from the confusion matrix and prints it.

4. **Loading and Splitting the Digits Dataset:**
   - The digits dataset is loaded using `datasets.load_digits()`.
   - The images are reshaped to flatten them into a 1D array, making them suitable for machine learning.
   - The `split_train_dev_test` function is used to split the data into training, dev, and test sets.

5. **Creating and Training the Classifier:**
   - A support vector classifier (`SVC`) is created with a gamma value of 0.001.
   - The classifier is trained using the training data and labels.

6. **Predicting and Evaluating the Dev Set:**
   - The `predict_and_eval` function is called to predict the labels on the dev set using the trained classifier and evaluate its performance.
   - The function computes and prints the classification report, confusion matrix, and a rebuilt classification report from the confusion matrix.

7. **Visualizing Dev Set Predictions:**
   - The dev set predictions are visualized using matplotlib.
   - A subplot with 1 row and 4 columns is created to display four images from the dev set, along with their predicted labels.

8. **Displaying the Visualization:**
   - Finally, `plt.show()` is used to display the visualizations.

In summary, this code demonstrates how to load the digits dataset, split it into train, dev, and test sets, train an SVC classifier, predict and evaluate its performance on the dev set, and visualize some of the dev set predictions along with their actual and predicted labels.

# Creation of another branch
created a branch named feature/assignment2 from the main
using the command 
git checkout -b <branch-name>
after the creation of this branch performed the required opeartions of divinding the dataset into tarin , development and test dataset and them made the predictions 
agter that merged the feature branch with the main branch.

creating for practice purpose
