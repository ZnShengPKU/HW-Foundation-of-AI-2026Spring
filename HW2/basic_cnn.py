# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import copy
import csv
import os

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


# 定义超参数
val_ratio = 0.1
batch_size = 1024
learning_rate = 3e-4
num_epochs = 3000
# Not sure how many epoches are need, as models are different.
patience = 20
adamw_weight_decay = 5e-4

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

# Actually transform should be different for train and val/test
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

transform_val_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

split_generator = torch.Generator().manual_seed(42)
train_full_aug = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
train_full_eval = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_val_test
)
n_full = len(train_full_aug)
n_val = int(n_full * val_ratio)
n_train = n_full - n_val
perm = torch.randperm(n_full, generator=split_generator).tolist()
train_indices = perm[:n_train]
val_indices = perm[n_train:]

train_dataset = Subset(train_full_aug, train_indices)
val_dataset = Subset(train_full_eval, val_indices)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val_test)

# 训练/验证：每个 epoch 各写一行。测试集只在训练结束后评估一次，故 test_data.csv 仅表头 + 一行（与 val 同列：epoch, loss, accuracy）
csv_dir = os.path.dirname(os.path.abspath(__file__))
train_csv_path = os.path.join(csv_dir, 'train_data.csv')
val_csv_path = os.path.join(csv_dir, 'val_data.csv')
test_csv_path = os.path.join(csv_dir, 'test_data.csv')

for path, header in (
    (train_csv_path, ['epoch', 'loss', 'accuracy']),
    (val_csv_path, ['epoch', 'loss', 'accuracy']),
):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = Net()

# I have Nvidia GPUs only.
device = torch.device('cuda:0')

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=adamw_weight_decay)


def run_epoch_eval(loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


best_val_acc = 0.0
best_state_dict = None
patience_counter = 0
trained_epochs = 0

for epoch in range(num_epochs):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total
    val_loss, val_acc = run_epoch_eval(val_loader)

    print(
        'Epoch [{}/{}] Train Loss: {:.4f}, Train Acc: {:.2f}% | Val Loss: {:.4f}, Val Acc: {:.2f}%'.format(
            epoch + 1, num_epochs, train_loss, train_acc * 100, val_loss, val_acc * 100
        )
    )

    ep = epoch + 1
    trained_epochs = ep
    with open(train_csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([ep, train_loss, train_acc])
    with open(val_csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([ep, val_loss, val_acc])

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping at epoch {} (no Val Acc improvement for {} epochs).'.format(
                epoch + 1, patience
            ))
            break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
print('Best Val Acc: {:.2f}%'.format(best_val_acc * 100))

test_loss, test_acc = run_epoch_eval(test_loader)
print('Test Loss: {:.4f}, Test Accuracy of the model on the 10000 test images: {:.2f} %'.format(
    test_loss, test_acc * 100
))

with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['epoch', 'loss', 'accuracy'])
    w.writerow([trained_epochs, test_loss, test_acc])