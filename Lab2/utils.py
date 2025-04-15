import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, val_losses):
    """
    Plot the training and validation loss curves.

    Args:
        train_losses (list[float]): The training loss for each epoch.
        val_losses (list[float]): The validation loss for each epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_map_curve(mAP_list):
    """
    Plot the validation mAP curve.

    Args:
        mAP_list (list[float]): The validation mAP for each epoch.
    """
    epochs = range(1, len(mAP_list) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mAP_list, 'g-', label='Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.legend()
    plt.grid(True)
    plt.show()
