import torch
import matplotlib.pyplot as plt

def plot_metrics(loss_history, accuracy_history):
    """Vẽ đồ thị loss và accuracy theo từng vòng huấn luyện."""
    rounds = range(1, len(loss_history) + 1)
    
    # Chuyển đổi tensor sang NumPy array
    loss_history = [loss.detach().numpy() if torch.is_tensor(loss) else loss for loss in loss_history]
    accuracy_history = [acc.detach().numpy() if torch.is_tensor(acc) else acc for acc in accuracy_history]
    
    plt.figure(figsize=(12, 6))

    # Vẽ đồ thị loss
    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss_history, label="Loss", marker="o")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    # Vẽ đồ thị accuracy
    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracy_history, label="Accuracy", marker="o", color="green")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("img/result.png")