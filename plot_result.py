from matplotlib import pyplot as plt
from train import *
plt.title("LOSS Curve")

plt.plot(plot_loss_train, label='Train Loss')
plt.plot(plot_loss_valid, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
plt.savefig('/content/drive/MyDrive/gender_detection_loss_curve.png')

