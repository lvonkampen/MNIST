import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv


class simpleNN(nn.Module):               # base class for network modelling
    def __init__(self, in_feat, hidden_feat, out_feat):
        super(simpleNN, self).__init__() # ensures nn.Module is initialized for our new class
        self.layer1 = nn.Linear(in_feat, hidden_feat)    # creates a layer with 2 inputs and # outputs
        self.layer2 = nn.Linear(hidden_feat, out_feat)  # creates a layer with # inputs and 2 outputs
        self.activation_hidden = nn.Sigmoid()
        self.activation_output = nn.Softmax(dim=1)

        init.xavier_normal_(self.layer1.weight)
        init.xavier_normal_(self.layer2.weight)

    def forward(self, x):
        # print(x.shape)
        z = self.layer1(x)
        # print(z.shape)
        z = self.activation_hidden(z)
        # print(z.shape)
        z = self.layer2(z)
        # print(z.shape)
        z = self.activation_output(z)
        # print(z.shape)
        return z


class CustomDataset(Dataset):
    def __init__(self, filepath):
        self.inputs = []
        self.targets = []

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                inputs = [1 if value == 'Yes' else 0 if value == 'No' else float(value)
                          for value in row[:-1]]
                target = [1,0] if row[-1] == 'python' else [0,1]

                self.inputs.append(inputs)
                self.targets.append(target)

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

def train_patience(model, train_loader, test_loader, loss_func, initial_learning_rate, epochs, patience):
    learning_rates = [initial_learning_rate, initial_learning_rate / 10, initial_learning_rate / 100]
    optim = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    current_learning_index = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets.float())
            loss.backward()
            optim.step()
            train_loss += loss.item()

        model.eval()
        with ((torch.no_grad())):
            valid_loss = 0
            correct = 0
            total = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                predict = torch.argmax(outputs, dim=1)
                true = torch.argmax(targets, dim=1)

                correct += (predict == true).sum().item()
                total += targets.size(0)

                loss = loss_func(outputs, targets.float())
                valid_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}")

        accuracy = correct / total * 100
        # print(f"Correct: {correct}")
        # print(f"Total: {total}")
        print(f"Accuracy: {accuracy:.2f}%")

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            print(f" - Validation Loss Improved To: {valid_loss:.6f}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            current_learning_index += 1  # Move to the next learning rate stage
            if current_learning_index < len(learning_rates):
                new_lr = learning_rates[current_learning_index]
                for param_group in optim.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate to {new_lr:.6e} and resetting patience.")
                patience_counter = 0  # Reset patience
            else:
                print(f"Stopping At Epoch: {epoch + 1} - Best Validation Loss: {best_loss:.6f}")
                break

def main():

    # HYPERPARAMETERS
    train_loader, train_shuffle = 4096, True
    test_loader, test_shuffle = 4096, False
    in_feat, hidden_feat, out_feat = 18, 128, 2
    initial_learning_rate = 0.0001
    epochs = 10000
    patience = 5
    loss_func = nn.BCELoss()


    trainset = CustomDataset('C:/Users/GoatF/Downloads/hw_04/training_data_very_large.csv')
    testset = CustomDataset('C:/Users/GoatF/Downloads/hw_04/testing_data.csv')

    train_loader = DataLoader(trainset, train_loader, train_shuffle)
    test_loader = DataLoader(testset, test_loader, test_shuffle)

    model = simpleNN(in_feat, hidden_feat, out_feat)
    print("Total parameters: " + str(sum([p.numel() for p in model.parameters() if p.requires_grad])))

    print([p for p in model.parameters() if p.requires_grad])

    train_patience(model, train_loader, test_loader, loss_func, initial_learning_rate, epochs, patience)


if __name__ == "__main__":
    main()

# 11/20 CHALLENGE: - Change MSELoss() to BCELoss()
#                  - Correct the matrix
#                  - Use Proper Validation Set: "testing_data.csv"

# 11/27 CHALLENGE: - Detect when validation loss is no longer decreases -- implement patience()
#                  - Adjust out_feat to support 2 neurons
#                  - Set all hyperparameters in Main()
#      Try to implement MNIST with extra out_feats -- https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
#      https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
#      https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
#                  - Implement neural network blocks containing (subnetwork within neural network) containing layers, activation, and normalization

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch
# Add reduction (2 at most) to the patience
# Correctly identify variables hiddenFeat -> hidden_feat
# TODO: Read about softmax and implement it!!
# Complete MNIST __ Apply different files for different classes (network class - mnist/the rest class
# Possibly implement https://github.com/soerenab/AudioMNIST
# Study Universal Approximation Theorem to understand approximation
# https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch
# Learn how to read .wav files in python