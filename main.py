import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(5)

## Model:
model = keras.Sequential([
    keras.layers.BatchNormalization(input_shape=[12],),
    keras.layers.Dense(256,   activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(256,  activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(256,  activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(10, activation="softmax")
])

model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            # metrics=['accuracy', 'binary_accuracy', 'categorical_accuracy']
            metrics = ["accuracy"]
)
# train data preparation:
train_df = pd.read_csv("data/train_total.csv")
train_x = np.column_stack((train_df.FSC_A.values,
                           train_df.FSC_H.values,
                           train_df.FSC_W.values,
                           train_df.SSC_A.values,
                           train_df.SSC_H.values,
                           train_df.SSC_W.values,
                           train_df.CD4.values,
                           train_df.CD5.values,
                           train_df.CD8.values,
                           train_df.LIVE_DEAD.values,
                           train_df.CD197.values,
                           train_df.CD45RA.values,
                           ))

one_hot_class_table = pd.get_dummies(train_df["class"])
    # Classes:
    # CD4_tcm - CD4_tem - CD4_temra - CD4_th0 - CD8_tc0 - CD8_tcm - CD8_tem - CD8_temra - debris - none_live
    # Number the cell type (train_y) to make the model and prediction simpler:
class_dict = {}
for i, cell_class in enumerate(one_hot_class_table.columns):
    class_dict[cell_class] = i
print(class_dict)

train_y = train_df["class"].apply(lambda x: class_dict[x])


# test data preparation:
test_df = pd.read_csv("data/test_total.csv")
test_x = np.column_stack((test_df.FSC_A.values,
                     test_df.FSC_H.values,
                     test_df.FSC_W.values,
                     test_df.SSC_A.values,
                     test_df.SSC_H.values,
                     test_df.SSC_W.values,
                     test_df.CD4.values,
                     test_df.CD5.values,
                     test_df.CD8.values,
                     test_df.LIVE_DEAD.values,
                     test_df.CD197.values,
                     test_df.CD45RA.values,
                     ))

    # Number the cell types (test_y)
test_y = test_df["class"].apply(lambda x: class_dict[x])


# Callback to stop the model early and fit
callback = EarlyStopping(monitor="val_loss",
                         mode="min",
                         verbose=1,
                         patience=2)

model.fit(train_x, train_y,
          batch_size=8000,
          epochs=10,
          verbose=2,
          validation_data =(test_x, test_y),
          callbacks = [callback]
          )

# Save the model:
model.save("FlowCytometry_model")
