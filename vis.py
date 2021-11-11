import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def plot_curves(model):
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])
    return fig

def plot_cm(true, preds, classes, figsize: tuple = (8, 6)):
    """Plot confusion matrix"""
    cm = metrics.confusion_matrix(true, preds)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0.2
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    return fig