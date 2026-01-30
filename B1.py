import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))

transform = transforms.Compose([transforms.ToTensor()])
# Biến đổi từ dữ liệu ảnh [0, 255] -> Tensor có giá trị [0, 1]
train_dataset = datasets.MNIST(root= '/data', train= True, download= True, transform= transform)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)

test_dataset = datasets.MNIST(root= '/data', train= False, download= True, transform= transform)

test_loader = DataLoader(test_dataset, batch_size= 1000, shuffle= False)

image, label = train_dataset[0]
plt.imshow(image.squeeze(), )

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x. view(-1, 784)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

class NNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 =  nn.Linear(128,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

class NNWithBatchNormRev(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 =  nn.Linear(128,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.layer3(x)
        return x

class NNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.dp1 = nn.Dropout(p= 0.3)
        self.layer2 =  nn.Linear(128,128)
        self.dp2 = nn.Dropout(p= 0.3)
        self.layer3 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.layer3(x)
        return x

def train_one_epoch(model, device, data_loader, optimizer, criterion):
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += output.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct * 100. / total

    return avg_loss, accuracy

def eval(model, device, test_loader, criterion):
    model.eval()
    total = 0
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += output.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct * 100. / total

    return avg_loss, accuracy

#==================================================================
#                     TESTING MODELS
#==================================================================
# Normal NN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device choosen: {device}')

model =SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr= 0.001)

num_epoch = 10

train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epoch):
    train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_loss, test_acc = eval(model, device, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f"Epoch: {epoch + 1} / {num_epoch}: train_loss = {train_loss} , train_acc = {train_acc}| test_loss = {test_loss} , test_acc = {test_acc}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Accuracy', marker='o')
ax2.plot(test_accs, label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
# dropout NN

model = NNWithDropout().to(device)
optimizer = optim.Adam(model.parameters(), lr= 0.001)
dropout_train_losses = []
dropout_train_accs = []
dropout_test_losses = []
dropout_test_accs = []

for epoch in range(num_epoch):
    dropout_train_loss, dropout_train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    dropout_train_losses.append(dropout_train_loss)
    dropout_train_accs.append(dropout_train_acc)
    dropout_test_loss, dropout_test_acc = eval(model, device, test_loader, criterion)
    dropout_test_losses.append(dropout_test_loss)
    dropout_test_accs.append(dropout_test_acc)
    print(f"Epoch: {epoch + 1} / {num_epoch}: dropout_train_loss = {dropout_train_loss} , dropout_train_acc = {dropout_train_acc}| dropout_test_loss = {dropout_test_loss} , dropout_test_acc = {dropout_test_acc}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(dropout_train_losses, label='Train Loss', marker='o')
ax1.plot(dropout_test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss with Dropout')
ax1.legend()
ax1.grid(True)

ax2.plot(dropout_train_accs, label='Train Accuracy', marker='o')
ax2.plot(dropout_test_accs, label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy with Dropout')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

#Batch Normalization:

model = NNWithBatchNorm().to(device)
optimizer = optim.Adam(model.parameters(), lr= 0.001)
BN_train_losses = []
BN_train_accs = []
BN_test_losses = []
BN_test_accs = []

for epoch in range(num_epoch):
    BN_train_loss, BN_train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    BN_train_losses.append(BN_train_loss)
    BN_train_accs.append(BN_train_acc)
    BN_test_loss, BN_test_acc = eval(model, device, test_loader, criterion)
    BN_test_losses.append(BN_test_loss)
    BN_test_accs.append(BN_test_acc)
    print(f"Epoch: {epoch + 1} / {num_epoch}: BN_train_loss = {BN_train_loss} , BN_train_acc = {BN_train_acc}| BN_test_loss = {BN_test_loss} , BN_test_acc = {BN_test_acc}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(BN_train_losses, label='Train Loss', marker='o')
ax1.plot(BN_test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss with Batch Normalization')
ax1.legend()
ax1.grid(True)

ax2.plot(BN_train_accs, label='Train Accuracy', marker='o')
ax2.plot(BN_test_accs, label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy with Batch Normalization')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
#Weight Decay

model = SimpleNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay= 0.0001)

WD_train_accs = []
WD_test_accs = []
WD_train_losses = []
WD_test_losses = []

for epoch in range(num_epoch):
    WD_train_loss, WD_train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    WD_test_loss, WD_test_acc = eval(model, device, test_loader, criterion)
    WD_train_accs.append(WD_train_acc)
    WD_test_accs.append(WD_test_acc)
    WD_train_losses.append(WD_train_loss)
    WD_test_losses.append(WD_test_loss)
    print(f"Epoch: {epoch + 1} / {num_epoch}: WD_train_loss = {WD_train_loss} , WD_train_acc = {WD_train_acc}| WD_test_acc = {WD_test_acc} , WD_test_loss = {WD_test_loss}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(WD_train_losses, label='Train Loss', marker='o')
ax1.plot(WD_test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss with Weight Decay')
ax1.legend()
ax1.grid(True)

ax2.plot(WD_train_accs, label='Train Accuracy', marker='o')
ax2.plot(WD_test_accs, label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy with Weight Decay')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

#Batch Normalization Rev:

model = NNWithBatchNormRev().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

BNV_train_losses = []
BNV_train_accs = []
BNV_test_losses = []
BNV_test_accs = []
for epoch in range(num_epoch):
    BNV_train_loss, BNV_train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    BNV_test_loss, BNV_test_acc = eval(model, device, test_loader, criterion)
    BNV_train_losses.append(BNV_train_loss)
    BNV_train_accs.append(BNV_train_acc)
    BNV_test_losses.append(BNV_test_loss)
    BNV_test_accs.append(BNV_test_acc)
    print(f"Epoch: {epoch + 1} / {num_epoch}, BNV_train_loss = {BNV_train_loss}, BNV_train_acc = {BNV_train_acc} | BNV_test_loss = {BNV_test_loss}, BNV_test_acc = {BNV_test_acc}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(BNV_train_losses, label='Train Loss', marker='o')
ax1.plot(BNV_test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss with Batch Normailization After ReLU')
ax1.legend()
ax1.grid(True)

ax2.plot(BNV_train_accs, label='Train Accuracy', marker='o')
ax2.plot(BNV_test_accs, label='Test Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy with Batch Normailization After ReLU')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()