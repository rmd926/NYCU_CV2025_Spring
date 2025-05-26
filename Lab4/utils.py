import matplotlib.pyplot as plt

def plot_training_curves(history):
    """
    Plot and display training and validation loss and PSNR curves.

    Args:
        history (dict): {
            'train_loss': list of float,
            'val_loss':   list of float,
            'val_psnr':   list of float
        }
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure(figsize=(10, 4))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # PSNR curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_psnr'], label='Val PSNR', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
