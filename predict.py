import pandas as pd
import numpy as np

from tensorflow import keras

np.random.seed(5)

#Load in test dataset:
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


##### Load model #####
model = keras.models.load_model("FlowCytometry_model", compile = True)

one_hot_class_table = pd.get_dummies(test_df["class"])

# classes:
# CD4_tcm - CD4_tem - CD4_temra - CD4_th0 - CD8_tc0 - CD8_tcm - CD8_tem - CD8_temra - debris - none_live

# Add the different classes to a dict
class_dict = {}
for i, x in enumerate(one_hot_class_table.columns):
    class_dict[x] = i
test_y = test_df["class"].apply(lambda x: class_dict[x])

#validation
print("Evaluate")
model.evaluate(test_x, test_y, verbose=2)

reverse_class_dict = dict(enumerate(one_hot_class_table.columns))

##### Predict #####

predict = pd.Series(np.round(model.predict(test_x)).argmax(1))

predict_class = pd.Series(predict.apply(lambda x: reverse_class_dict[x]))
test_df["predict_class"] = predict_class
false_predict = test_df[test_df["class"] != test_df["predict_class"]]

print(f"Accuracy = {100-((len(false_predict)/len(test_df))*100)}%")
test_df.to_csv("data/test_predicted.csv", index=False)

#Accuracy on test set: 99.22%





