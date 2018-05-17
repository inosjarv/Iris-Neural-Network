import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np 
import tflearn
import matplotlib.pyplot as plt
from tflearn.data_utils import load_csv

to_ignore = [0, 1]
data, labels = load_csv('iris.csv', target_column=4, columns_to_ignore=to_ignore, categorical_labels=True, n_classes=3)
                      
net = tflearn.input_data(shape=[None, 2])
net = tflearn.layers.core.fully_connected(net, 32)
net = tflearn.layers.core.fully_connected(net, 32)
net = tflearn.layers.core.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net) 
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=20, batch_size=8, show_metric=True, validation_set=0.1) 
f0 = [1.3, 0.2]
f1 = [1.4, 0.2]
f2 = [1.1, 0.1]
f3 = [1.0, 0.3]
f4 = [3.5,1]
f5 = [3.9,1.1]
f6 = [3.6,1.3]
f7 = [5.8,2.1]
f8 = [5.7,1.9]
f9 = [6.1,2.2]
x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
max_pred, index, predicted_values = [], [], []
flowers = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]
pred = model.predict(flowers)
i=0
while i<len(flowers):
  if pred[i][0]>=pred[i][1] and pred[i][0]>=pred[i][2]:
    x1.append(flowers[i][0])
    y1.append(flowers[i][1])
  elif pred[i][0]<=pred[i][2]:
    x2.append(flowers[i][0])
    y2.append(flowers[i][1])
  else:
    x3.append(flowers[i][0])
    y3.append(flowers[i][1])
  i = i + 1

plt.scatter(x1, y1, c="red", s=10, label="Setosa")
plt.scatter(x2, y2, c="green", s=10, label="Versicolor")
plt.scatter(x3, y3, c="blue", s=10, label="Virginica")

plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend()
plt.show()


