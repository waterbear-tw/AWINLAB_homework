#導入必要套件
from builtins import range
import tensorflow as tf
## 若非使用resnet50，則會直接使用sequential配合Conv2D, MaxPool2d以及下列layers方法等等，來搭建CNN
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.models import Model, load_model

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os
import random

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# 設定資料路徑，此處因應環境不同有所差別；現為在macbook上運行，故逕以資料夾路徑
train_dir = "train"
valid_dir = "valid"
test_dir = "test"
classList = ["Airedale", "Beagle", "Bloodhound", "Bluetick", "Chihuahua", "Collie", "Dingo", "French Bulldog", "German Sheperd", "Malinois", "Newfoundland", "Pekinese", "Pomeranian", "Pug", "Vizsla"]

# 影像前處理preprocessing
image_gen = ImageDataGenerator()
train_data_gen = ImageDataGenerator(horizontal_flip = True,
                                    rotation_range=20,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2)

train_data = train_data_gen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    color_mode = "rgb",
    batch_size = 32,
    class_mode ="categorical",
    classes = classList,  ## 使用自定義classList來限定欲訓練狗狗品種
    shuffle = True
)

valid_data = image_gen.flow_from_directory(
    valid_dir,
    target_size = (224, 224),
    color_mode = "rgb",
    batch_size = 32,
    class_mode ="categorical",
    classes = classList,
    shuffle = False
)
test_data = image_gen.flow_from_directory(
    test_dir,
    target_size = (224, 224),
    color_mode = "rgb",
    batch_size = 32,
    class_mode ="categorical",
    classes = classList,
    shuffle = False
)


# 此處用以確定標籤名稱以及對應的索引值
labels = train_data.class_indices
class_mapping = dict((v,k) for k,v in labels.items())
class_mapping
print(class_mapping, "!!!!!!!!")


# 使用ResNet50 model
resnetModel = ResNet50(weights="imagenet", include_top=False, 
                       input_tensor=Input(shape=(224, 224, 3))) ## 將input設定與前處理時相同

outputs = resnetModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.2)(outputs)
outputs = Dense(15, activation="softmax")(outputs)

model = Model(inputs=resnetModel.input, outputs=outputs)

# 設定model
opt = Adam(learning_rate=0.00001)
model.compile(optimizer = opt,
              loss = "categorical_crossentropy",
              metrics = ["accuracy"]
)


model.build(((None, 224, 224, 3)))
model.summary()


# model.fit: 訓練 model
train_cb = ModelCheckpoint("model.keras", save_best_only = True) ## 修改過，從./model 改了加上.keras
history = model.fit(train_data, validation_data = valid_data, callbacks = [train_cb], epochs = 20, batch_size = 32)

# 使用已知答案的測試集預測資料
batch_size = 32
y_pred = model.predict(test_data, batch_size=batch_size)


# 將訓練完畢的model存擋，可以單獨調用測試其他圖片
tf.keras.models.save_model(model, "model.h5")


# 取得訓練時的accuracy資料並印出
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

epoch_count = range(1, len(val_accuracy) + 1)

plt.plot(epoch_count, accuracy, "b--", label = "Accuracy")
plt.plot(epoch_count, val_accuracy, "r--",  label = "Valid Accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show();

# 載入模組，對測試集進行預測（因一條龍式執行時間過久，難以debug，故以載入方式進行）
test_model = load_model('model.h5')
testData = 'TestingSet'
test_files = os.listdir('TestingSet')
random.shuffle(test_files)

# 建立list放置預測答案
test_results = []

# 受到路徑格式限制，採取迭代方式來對圖片前處理
for file_name in test_files:
    img_path = os.path.join(testData, file_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = img_array.reshape((1,) + img_array.shape)

    # 進行預測，將結果加入到test_results之中
    predictions = test_model.predict(img_array)
    test_results.append((file_name, classList[predictions.argmax()]))
    print(file_name ,':',classList[predictions.argmax()])

df1 = pd.DataFrame(test_results,
                   columns=["檔名", "結果"])
df1.to_excel("output.xlsx", index = False)  