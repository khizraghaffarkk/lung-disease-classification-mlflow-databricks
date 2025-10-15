import matplotlib.pyplot as plt
import mlflow

def evaluate_model(model, test_ds):
    test_loss, test_acc = model.evaluate(test_ds)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    return test_loss, test_acc

def plot_history(history, epochs):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(range(epochs), history.history['accuracy'], label='Train Accuracy')
    plt.plot(range(epochs), history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(range(epochs), history.history['loss'], label='Train Loss')
    plt.plot(range(epochs), history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
