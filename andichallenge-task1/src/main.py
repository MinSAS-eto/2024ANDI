import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from models.model import MyModel
from utils import train, evaluate

def main():
    # Load dataset
    train_dataset = CustomDataset('data/train_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(10):  # Number of epochs
        train(model, train_loader, criterion, optimizer, epoch)

    # Evaluate the model
    test_dataset = CustomDataset('data/test_data.csv')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()