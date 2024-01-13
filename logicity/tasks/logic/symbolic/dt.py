import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

class DecisionTreeRunner:
    def __init__(self, data_X_train, data_Y_train, data_X_test, data_Y_test, Yname, logger, uni_boundary=0.5, w_bernoulli=False, irr_c=0):
        self.data_X_train = data_X_train
        self.data_Y_train = data_Y_train
        self.data_X_test = data_X_test
        self.data_Y_test = data_Y_test
        self.add_noise(uni_boundary, w_bernoulli, irr_c)
        
        self.Yname = Yname
        self.logger = logger
        self.log_distribution("Dataset distribution before training")

    def log_distribution(self, message):
        unique_rows, counts = np.unique(self.data_Y_train, axis=0, return_counts=True)
        # Create a PrettyTable instance
        table = PrettyTable()
        # Use Yname for the field names
        table.field_names = self.Yname + ["Count"]
        table.title = message

        # Populate the table with data
        for row, count in zip(unique_rows, counts):
            # Convert row to labels using Yname
            label_row = [self.Yname[i] if x == 1 else "UNKNOWN" for i, x in enumerate(row)]
            table.add_row(label_row + [count])

        # Sort the table by the 'Count' column
        table.sortby = "Count"
        table.reversesort = True

        # Log the table using the provided logger
        self.logger.info(f"Training Dist\n{table}")

        unique_rows, counts = np.unique(self.data_Y_test, axis=0, return_counts=True)
        # Create a PrettyTable instance
        table = PrettyTable()
        # Use Yname for the field names
        table.field_names = self.Yname + ["Count"]
        table.title = message

        # Populate the table with data
        for row, count in zip(unique_rows, counts):
            # Convert row to labels using Yname
            label_row = [self.Yname[i] if x == 1 else "UNKNOWN" for i, x in enumerate(row)]
            table.add_row(label_row + [count])

        # Sort the table by the 'Count' column
        table.sortby = "Count"
        table.reversesort = True

        # Log the table using the provided logger
        self.logger.info(f"Test Dist\n{table}")

    def run(self):
        # Convert the 0.5 in data_Y to 0 to indicate 'UNKNOWN'/'FALSE'
        data_Y_converted = np.where(self.data_Y_train == 0.5, 0, 1)

        # Initialize the Decision Tree Classifier
        self.clf = DecisionTreeClassifier()

        # Fit the classifier to your data
        self.clf.fit(self.data_X_train, data_Y_converted)
    
    def evaluate(self):
        # Make predictions
        predictions = self.clf.predict(self.data_X_test)
        data_Y_converted = np.where(self.data_Y_test == 0.5, 0, 1)
        # Calculate accuracy
        accuracy = accuracy_score(data_Y_converted, predictions)
        self.logger.info(f"Accuracy: {accuracy}")
        report_dict = classification_report(data_Y_converted, predictions, target_names=self.Yname, output_dict=True)    
        # Convert the classification report dictionary to a DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        return report_df

    def add_noise(self, uni_boundary, w_bernoulli=False, irr_c=0):
        dataX = torch.cat((self.data_X_train, self.data_X_test), dim=0)
        # Adding irr concepts
        if irr_c > 0:
            noise_irr_c = torch.rand((dataX.shape[0], irr_c))
            bernoulli_c = torch.bernoulli(noise_irr_c)
            dataX = torch.cat((dataX, bernoulli_c), dim=1)
        train_sz = self.data_X_train.shape[0]
        test_sz = self.data_X_test.shape[0]
        # Adding uniform noise
        noise_0 = torch.rand(dataX.size()) * uni_boundary
        noise_1 = 1 - torch.rand(dataX.size()) * uni_boundary
        noisy_tensor = torch.where(dataX == 0, noise_0, noise_1)
        self.data_X_train = noisy_tensor[:train_sz]
        self.data_X_test = noisy_tensor[train_sz:]
        # Adding bernoulli noise
        if w_bernoulli:
            bernoulli_tensor = torch.bernoulli(self.data_X_train)
            self.data_X_train = bernoulli_tensor
        return