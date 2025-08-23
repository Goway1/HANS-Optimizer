import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimizers.hans import HANS
from models import get_model

def get_cifar10_optimizer(model, optimizer_type='sgd'):
    if optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)
    elif optimizer_type == 'hans':
        return HANS(model.parameters(), lr=0.001, beta_fast=0.9, beta_slow=0.99, beta_adaptive=0.999, alpha=0.7, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

def get_cifar10_scheduler(optimizer, epochs, optimizer_type='sgd'):
    if optimizer_type == 'sgd':
        milestones = [epochs//2, epochs//4*3]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def train_epoch_with_gradient_clipping(model, train_loader, optimizer, scheduler, criterion, device, epoch, total_epochs, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch+1}/{total_epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | LR: {current_lr:.6f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def test_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def train_cifar10_fixed(model, train_loader, test_loader, device, optimizer_type='sgd', epochs=100):
    optimizer = get_cifar10_optimizer(model, optimizer_type)
    scheduler = get_cifar10_scheduler(optimizer, epochs, optimizer_type)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_with_gradient_clipping(model, train_loader, optimizer, scheduler, criterion, device, epoch, epochs, max_grad_norm=1.0)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'best_model_{optimizer_type}.pth')

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%')

    return train_losses, test_losses, train_accs, test_accs, best_acc

def compare_optimizers(train_loader, test_loader, device, epochs=50, architecture='cnn'):
    optimizers_to_test = ['sgd', 'adam', 'hans']
    results = {}

    for optim_name in optimizers_to_test:
        print(f"\n=== Testing {optim_name.upper()} optimizer ===")
        model = get_model(architecture).to(device)
        train_losses, test_losses, train_accs, test_accs, best_acc = train_cifar10_fixed(
            model, train_loader, test_loader, device, optimizer_type=optim_name, epochs=epochs
        )
        results[optim_name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'best_acc': best_acc
        }

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for optim_name, res in results.items():
        plt.plot(res['test_accs'], label=optim_name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    for optim_name, res in results.items():
        plt.plot(res['test_losses'], label=optim_name)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    plt.show()

    print("\n=== Best Accuracy Results ===")
    for optim_name, res in results.items():
        print(f"{optim_name.upper()}: {res['best_acc']:.2f}%")

    return results
