import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import time

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root= '/data', train= True, download= True, transform= transform)
test_dataset = datasets.FashionMNIST(root = '/data', train= False, download= True, transform= transform)

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)
test_loader = DataLoader(test_dataset, batch_size= 1000, shuffle= False)

image, label = train_dataset[0]
plt.imshow(image.squeeze(), cmap = 'gray')
plt.axis('off')
plt.title(f'label: {label}')
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"This is a: {class_names[label]}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size= 3, padding= 1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, padding= 1)
        self.layer1 = nn.Linear(64 * 7 * 7, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)

        return x

class MiniVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size= 3, padding= 1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size= 3, padding= 1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size= 3, padding= 1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size= 3, padding= 1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size= 3, padding= 1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size= 3, padding= 1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.layer1 = nn.Linear(128 * 3 * 3, 256)
        self.drop = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.layer2(x)

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= stride, padding= 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if(stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size= 1, stride= stride, bias= False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = F.relu(out)

        return out



class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size= 3, stride= 1, padding= 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)

        self.stage1 = self._make_stage(32, 32, num_blocks = 2, stride = 1)
        self.stage2 = self._make_stage(32, 64, num_blocks = 2, stride = 2)
        self.stage3 = self._make_stage(64, 128, num_blocks = 2, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(128, 10)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride = 1))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


#=======================Train========================================
def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    total = 0
    correct = 0
    total_loss = 0.0

    for data, target in train_loader:
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

    avg_loss = total_loss / len(train_loader)
    accuracy = correct * 100. / total

    return avg_loss, accuracy


def eval(model, device, test_loader, criterion):
    model.eval()
    total = 0
    total_loss = 0.0
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



#==============================================================
#====================MAIN==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device choosen: {device}')
model = SimpleCNN().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"SimpleCNN parameters: {total_params:,}")
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epoch = 10
train_losses = []
train_accs = []
test_losses = []
test_accs = []

start_time = time.perf_counter()  

for epoch in range(num_epoch):
    train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_loss, test_acc = eval(model, device, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f"epoch: {epoch + 1}/{num_epoch}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f} | test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}")

end_time = time.perf_counter()     
total_time = end_time - start_time
print(f"Tổng thời gian chạy: {total_time} giây")
print(f" Thời gian trung bình mỗi epoch: {total_time / num_epoch} giây")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))


ax1.plot(train_losses, label="Train Loss", marker='o')
ax1.plot(test_losses, label="Test Loss", marker='s')  
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)


ax2.plot(train_accs, label="Train Accuracy", marker='o')
ax2.plot(test_accs, label="Test Accuracy", marker='s')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiniVGG().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"MiniVGG parameters: {total_params:,}")

optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.CrossEntropyLoss()

VGG_train_losses = []
VGG_train_accs = []
VGG_test_losses = []
VGG_test_accs = []

start_time = time.perf_counter()   

for epoch in range(num_epoch):
    VGG_train_loss, VGG_train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
    VGG_train_losses.append(VGG_train_loss)
    VGG_train_accs.append(VGG_train_acc)
    VGG_test_loss, VGG_test_acc = eval(model, device, test_loader, criterion)
    VGG_test_losses.append(VGG_test_loss)
    VGG_test_accs.append(VGG_test_acc)
    print(f"Epoch: {epoch + 1} / 10: VGG_train_loss = {VGG_train_loss:.4f}, VGG_train_acc = {VGG_train_acc:.4f}|| VGG_test_loss = {VGG_test_loss:.4f}, VGG_test_acc = {VGG_test_acc:.4f}")

end_time = time.perf_counter()     
total_time = end_time - start_time

print(f"Tổng thời gian chạy: {total_time} giây")
print(f" Thời gian trung bình mỗi epoch: {total_time / num_epoch} giây")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))

ax1.plot(VGG_train_losses, label = "VGG Train Losses", marker= 'o')
ax1.plot(VGG_test_losses, label = "VGG Test Losses", marker = 's')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Test Loss with VGG")
ax1.legend()
ax1.grid(True)

ax2.plot(VGG_train_accs, label = "VGG Train Accuracy", marker= 'o')
ax2.plot(VGG_test_accs, label = "VGG Test Accuracy", marker = 's')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training and Test Accuracy with VGG")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# Create ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_resnet = MiniResNet().to(device)

# Count parameters
total_params = sum(p.numel() for p in model_resnet.parameters())
print(f"MiniResNet parameters: {total_params:,}")

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)

# Train
num_epoch = 10
resnet_train_losses = []
resnet_train_accs = []
resnet_test_losses = []
resnet_test_accs = []

start_time = time.perf_counter()   
print("\nTraining MiniResNet...")
for epoch in range(num_epoch):
    train_loss, train_acc = train_one_epoch(model_resnet, device, train_loader, optimizer, criterion)
    resnet_train_losses.append(train_loss)
    resnet_train_accs.append(train_acc)
    
    test_loss, test_acc = eval(model_resnet, device, test_loader, criterion)
    resnet_test_losses.append(test_loss)
    resnet_test_accs.append(test_acc)
    
    print(f"Epoch {epoch + 1}/{num_epoch}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
          f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(resnet_train_losses, label="ResNet Train Loss", marker='o')
ax1.plot(resnet_test_losses, label="ResNet Test Loss", marker='s')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Test Loss with ResNet")
ax1.legend()
ax1.grid(True)

ax2.plot(resnet_train_accs, label="ResNet Train Accuracy", marker='o')
ax2.plot(resnet_test_accs, label="ResNet Test Accuracy", marker='s')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Test Accuracy with ResNet")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"\nFinal ResNet Test Accuracy: {resnet_test_accs[-1]:.2f}%")
end_time = time.perf_counter()     
total_time = end_time - start_time

print(f"Tổng thời gian chạy: {total_time} giây")
print(f" Thời gian trung bình mỗi epoch: {total_time / num_epoch} giây")