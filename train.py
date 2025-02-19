import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torchvision.transforms as transforms
import time
from torch.utils.data import random_split
from tqdm import tqdm
import os

CHECKPOINT_PATH = "saved/checkpoint.pth"
DEVICE = torch.device("cuda")

def validate(model, valloader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total_samples += labels.size(0)

    val_loss = total_loss / len(valloader)
    val_acc = total_correct / total_samples
    return val_loss, val_acc

def main():

    # hyper parameters
    base_lr = 0.001
    batch_size = 128
    lr = base_lr * (batch_size / 128)
    num_workers = 1
    pin_memory = True
    totalepoch = 100

    if not os.path.exists("saved/cifar10_trainset.pth"):
        # augmentation
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # load trainset with augmentation
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        torch.save(trainset, "saved/cifar10_trainset.pth")

    # load saved trainset
    full_trainset = torch.load("saved/cifar10_trainset.pth", weights_only=False)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # load resnet18
    model = models.resnet18(pretrained=False)
    # custom layers
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)

    # def
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # load checkpoint
    start_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    # train
    epoch_times = []
    for epoch in range(start_epoch, totalepoch):
        model.train()
        start_time = time.time()
        train_loss = 0

        with tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{totalepoch}") as pbar:
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1) 
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = validate(model, valloader, criterion)

        # update best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "saved/resnet18_cifar10_v3.pth")

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        # update lr
        scheduler.step()

        epoch_time = time.time() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else epoch_time
        remaining_time = avg_epoch_time * (totalepoch - epoch - 1)
        epoch_times.append(epoch_time)

        print(f"Epoch [{epoch+1}/{totalepoch}] Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} "
            f"Time Taken: {epoch_time:.2f}s - ETA: {remaining_time:.2f}s")


if __name__ == '__main__':
    main()