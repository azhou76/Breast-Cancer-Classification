# Breast-Cancer-Classification
Project to correctly classify patients as having benign or malignant cases of breast cancer based off of FNA image measurements.

The data uses feature information that is computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Features describe characteristics of the cell nuclei present in the image.

Dataset retrieved from the  UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) and downloaded from Kaggle at https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

After fine-tuning the hyperparameters (i.e. learning rate of 0.03, batch size of 16, convergence threshold of 0.01), I achieved a maximum test accuracy (with the train-test data split being 70-30) of 93.6%, converging after 24 epochs.

To run program, use the command "python main.py" within a proper machine learning environment. This project takes inspiration from a former course project that used a different dataset.