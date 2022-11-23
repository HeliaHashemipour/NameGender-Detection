from data_preprocessing import *
from model import *
from train import *

MODEL.eval()
with torch.no_grad():
    TEST_TRUE_LABLES = []
    TEST_PREDICTED_LABLES = []

    for batch in DATALOADER_TEST:
        DATA_BATCH, LABEL_BATCH = batch

        DATA_BATCH = DATA_BATCH.to(DEVICE)
        LABEL_BATCH = LABEL_BATCH.to(DEVICE)

        OUTPUT = MODEL(DATA_BATCH)

        TEST_TRUE_LABLES.extend(LABEL_BATCH.tolist())
        TEST_PREDICTED_LABLES.extend(torch.round(OUTPUT).tolist())

    ACCURACY = binary_acc(y_pred=TEST_PREDICTED_LABLES,
                          y_test=TEST_TRUE_LABLES)

    print(f'Accuracy on Test Set: {ACCURACY :.2f}%')
