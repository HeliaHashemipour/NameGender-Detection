from data_preprocessing import *
from model import *
from train import *

MODEL.eval()

with torch.no_grad():
    test_true_labels = []
    test_predicted_labels = []

    for batch in DATALOADER_TEST:
        DATA_BATCH, LABEL_BATCH = batch

        DATA_BATCH = DATA_BATCH.to(DEVICE)
        LABEL_BATCH = LABEL_BATCH.to(DEVICE)

        out = MODEL(DATA_BATCH)

        test_true_labels.extend(LABEL_BATCH.tolist())
        test_predicted_labels.extend(torch.round(out).tolist())

    ACCURACY = binary_acc(y_pred=test_predicted_labels,
                          y_test=test_true_labels)

    print(f'Accuracy on Test Set is {ACCURACY :.2f}%')
