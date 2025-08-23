from config import Config
from utils.data import get_data_loaders
from utils.debug import simple_overfit_test, debug_hans_optimizer
from utils.train import compare_optimizers
from models import get_model

def main():
    args = Config()
    print(f"Using device: {args.device}")
    print(f"Using architecture: {args.architecture}")

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size, args.num_workers, args.overfit_single_batch)

    simple_success = simple_overfit_test(args.device)
    if not simple_success:
        print("Model failed simple overfit test. There may be a fundamental issue.")
        return

    print("\nDebugging HANS optimizer...")
    hans_works = debug_hans_optimizer()
    if not hans_works:
        print("HANS had issues on the simple test, but we'll continue to CIFAR-10.")

    print("\n=== Comparing optimizers on CIFAR-10 ===")
    results = compare_optimizers(train_loader, test_loader, args.device, epochs=args.epochs, architecture=args.architecture)

    hans_acc = results['hans']['best_acc']
    adam_acc = results['adam']['best_acc']
    sgd_acc = results['sgd']['best_acc']

    print("\n=== Final CIFAR-10 Results ===")
    print(f"HANS: {hans_acc:.2f}%")
    print(f"Adam: {adam_acc:.2f}%")
    print(f"SGD: {sgd_acc:.2f}%")

    best_baseline = max(adam_acc, sgd_acc)
    if hans_acc > best_baseline:
        print(f"\n🎉 HANS outperforms both baselines by {hans_acc - best_baseline:.2f}%!")
        print("The hierarchical momentum approach is working!")
    elif hans_acc > best_baseline * 0.95:
        print("\n✓ HANS performs competitively (within 5% of best baseline)")
        print("Could improve further with hyperparameter tuning.")
    elif hans_acc > best_baseline * 0.8:
        print("\n⚠ HANS underperforms but is reasonable (within 20% of best baseline)")
        print("Consider tuning or architectural changes.")
    else:
        gap = (1 - hans_acc / best_baseline) * 100
        print(f"\n❌ HANS significantly underperforms ({gap:.1f}% worse)")
        print("There may be issues with the implementation or hyperparameters.")

    hans_final_train = results['hans']['train_accs'][-1]
    hans_final_test = results['hans']['test_accs'][-1]
    overfitting = hans_final_train - hans_final_test

    print("\nTraining insights:")
    print(f"HANS train acc: {hans_final_train:.2f}%, test acc: {hans_final_test:.2f}%")
    print(f"Overfitting gap: {overfitting:.2f}%")

    if overfitting > 20:
        print("High overfitting - consider more regularization or different hyperparameters.")
    elif overfitting < 5:
        print("Low overfitting - maybe use higher LR or less regularization.")
    else:
        print("Reasonable overfitting gap.")

if __name__ == "__main__":
    main()
