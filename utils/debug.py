import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimizers.hans import HANS

def simple_overfit_test(device):
    print("=== Running simple overfit test ===")
    x = torch.randn(2, 3, 32, 32).to(device)
    y = torch.tensor([0, 1]).to(device)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*32*32, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    losses, accuracies = [], []
    acc = torch.tensor(0.0).to(device)

    for i in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y).float().mean()

        losses.append(loss.item())
        accuracies.append(acc.item())

        if i % 10 == 0:
            print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.2f}")
        if acc == 1.0:
            print("Successfully overfit simple data!")
            break
    else:
        print("Failed to overfit simple data")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Simple Test Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Simple Test Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('simple_overfit_test.png')
    plt.show()

    return acc.item() == 1.0

def debug_hans_optimizer():
    print("=== Debugging HANS optimizer ===")
    print("Testing basic SGD first...")
    x_sgd = torch.tensor([5.0], requires_grad=True)

    for _ in range(100):
        x_sgd.grad = None
        loss = x_sgd ** 2
        loss.backward()
        with torch.no_grad():
            x_sgd -= 0.1 * x_sgd.grad

    print(f"SGD final x: {x_sgd.item():.6f}")

    if abs(x_sgd.item()) > 1.0:
        print("Even SGD doesn't work - the learning rate is too high for f(x)=x^2")
        print("Trying lr=0.01...")
        x_sgd2 = torch.tensor([5.0], requires_grad=True)
        for _ in range(200):
            x_sgd2.grad = None
            loss = x_sgd2 ** 2
            loss.backward()
            with torch.no_grad():
                x_sgd2 -= 0.01 * x_sgd2.grad
            if abs(x_sgd2.item()) < 1e-4:
                break
        print(f"SGD with lr=0.01 final x: {x_sgd2.item():.6f}")
        working_lr = 0.01 if abs(x_sgd2.item()) < 0.01 else 0.001
    else:
        working_lr = 0.1

    print(f"\nNow testing HANS vs Adam with lr={working_lr}")
    x_hans = torch.tensor([5.0], requires_grad=True)
    optim_hans = HANS([x_hans], lr=working_lr, alpha=0.9)

    hans_path, hans_grads = [], []
    for i in range(300):
        optim_hans.zero_grad()
        loss = x_hans ** 2
        loss.backward()
        hans_grads.append(x_hans.grad.item())
        optim_hans.step()
        hans_path.append(x_hans.item())
        if abs(x_hans.item()) < 1e-5:
            print(f"HANS converged after {i+1} steps")
            break

    x_adam = torch.tensor([5.0], requires_grad=True)
    optim_adam = torch.optim.Adam([x_adam], lr=working_lr)
    adam_path, adam_grads = [], []
    for i in range(300):
        optim_adam.zero_grad()
        loss = x_adam ** 2
        loss.backward()
        adam_grads.append(x_adam.grad.item())
        optim_adam.step()
        adam_path.append(x_adam.item())
        if abs(x_adam.item()) < 1e-5:
            print(f"Adam converged after {i+1} steps")
            break

    print(f"HANS final x: {x_hans.item():.8f}")
    print(f"Adam final x: {x_adam.item():.8f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(hans_path[:50], label='HANS', linewidth=2)
    plt.plot(adam_path[:50], label='Adam', linewidth=2)
    plt.title(f"First 50 Steps (lr={working_lr})")
    plt.xlabel("Step"); plt.ylabel("x value"); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot([abs(x) for x in hans_path], label='HANS', linewidth=2)
    plt.plot([abs(x) for x in adam_path], label='Adam', linewidth=2)
    plt.title("Convergence (|x|)")
    plt.xlabel("Step"); plt.ylabel("|x|"); plt.yscale('log'); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot([x**2 for x in hans_path], label='HANS', linewidth=2)
    plt.plot([x**2 for x in adam_path], label='Adam', linewidth=2)
    plt.title("Loss Over Time")
    plt.xlabel("Step"); plt.ylabel("f(x) = x²"); plt.yscale('log'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('hans_debug.png', dpi=150)
    plt.show()

    hans_success = abs(x_hans.item()) < 0.01
    adam_success = abs(x_adam.item()) < 0.01
    print("\nResults:")
    print(f"HANS converged: {hans_success}")
    print(f"Adam converged: {adam_success}")

    if hans_success:
        print("HANS optimizer works correctly!")
        return True
    elif adam_success:
        print("Adam works but HANS doesn't - there may be an issue with HANS")
        return False
    else:
        print("Both optimizers failed - may need different hyperparameters")
        return False
