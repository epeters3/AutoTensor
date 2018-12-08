import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_tf_history(history, file_path):
    """
    Source: https://www.tensorflow.org/tutorials/keras/basic_text_classification#create_a_validation_set
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    figure, ax1 = plt.subplots()

    ax1.plot(epochs, acc, 'b--', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')

    ax2 = ax1.twinx()
    ax2.plot(epochs, loss, "r--", label="Training Loss")
    ax2.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")

    plt.title('Training and validation accuracy')
    legend = ax1.legend(loc=9, bbox_to_anchor=(0, -0.1))
    legend2 = ax2.legend(loc=9, bbox_to_anchor=(1, -0.1))

    plt.savefig(
        file_path, bbox_extra_artists=(legend, legend2), bbox_inches='tight')
