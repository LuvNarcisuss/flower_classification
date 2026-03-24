import scipy.io
import numpy as np
import os
from PIL import Image

labels = scipy.io.loadmat('./flower_photos/imagelabels.mat')#该地址为imagelabels.mat的相对地址
labels = np.array(labels['labels'][0]) - 1
print("labels:", labels)

setid = scipy.io.loadmat('./flower_photos/setid.mat')#该地址为setid.mat的相对地址

validation = np.array(setid['valid'][0]) - 1 # 验证集 0.1
np.random.shuffle(validation)

test = np.array(setid['trnid'][0]) - 1 # 训练集 0.1
np.random.shuffle(test)

train = np.array(setid['tstid'][0]) - 1 #测试集 0.8
np.random.shuffle(train)

flower_dir = list()

for img in os.listdir("./flower_photos/jpg"):#该地址为源数据图片的相对地址
    flower_dir.append(os.path.join("./flower_photos/jpg", img))

flower_dir.sort()

des_folder_train = "./flower_photos_split/train"
#该地址可为新建的训练数据集文件夹的相对地址
for tid in train:
    #打开图片并获取标签
    img = Image.open(flower_dir[tid])
    print(img)
    # print(flower_dir[tid])
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_train, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)

des_folder_validation = "./flower_photos_split/val"
#该地址为新建的验证数据集文件夹的相对地址
for tid in validation:
    img = Image.open(flower_dir[tid])
    # print(flower_dir[tid])
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_validation, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)

des_folder_test = "./flower_photos_split/test"
#该地址为新建的测试数据集文件夹的绝对地址
for tid in test:
    img = Image.open(flower_dir[tid])
    # print(flower_dir[tid])
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_test, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)
