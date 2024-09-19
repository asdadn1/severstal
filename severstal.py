#Импорт библиотек
import numpy as np 
import pandas as pd 
from ultralytics import YOLO
import os
import glob
import cv2
from PIL import Image, ImageDraw 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Задаем пути к файлам
for dirname, _, filenames in os.walk('C:/Users/admin/Desktop/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break
DATASET_PATH = "C:/Users/admin/Desktop/data"
TRAIN_IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "data/images/train")
TEST_IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "data/images/val")

init_dir=os.getcwd()

train_img_names = sorted(glob.glob(TRAIN_IMAGE_DATASET_PATH +'/' + "*.jpg"))
test_img_names = sorted(glob.glob(TEST_IMAGE_DATASET_PATH +'/' + "*.jpg"))


#Дебаг принты чтобы понять нашли ли мы все файлы
print(len(train_img_names))
print(len(test_img_names))


#Чтение файла train.csv и его систематизация
train_df_b = pd.read_csv(DATASET_PATH+"/train.csv")
train_df_b['ClassId'] = train_df_b['ClassId'].astype(int)
train_df_b.groupby(['ImageId'])['ImageId'].filter(lambda x: len(x) > 1).count()
train_df_b.groupby(['ImageId'])['ImageId'].filter(lambda x: len(x) > 2)
train_df_b.loc[train_df_b['ImageId'].isin(['db4867ee8.jpg'])]
train_df_b.groupby(['ImageId','ClassId'])['ImageId'].count().max()
train_df = train_df_b.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
train_df['count'] = train_df.count(axis=1)



#Функции для обработки, создания и преобразования маски к формату YOLO
def read_image_with_masks(row_id, df):
    fname_o = df.iloc[row_id].name
    fname = TRAIN_IMAGE_DATASET_PATH +'/' + fname_o
    onlyname = fname_o.split('.')[0]
    
    labels = df.iloc[row_id][:4]
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape=image.shape

    list_names = []
    list_with_all_masks = []

    for classname, label in enumerate(labels.values):
        if label is not np.nan:
            string_array = label.rstrip().split()
            int_array = [int(string) for string in string_array]
            binary_mask = create_mask(int_array, shape)
            list_with_all_masks.append(binary_mask)
            list_names.append(int(classname))

    return fname, onlyname, image, list_with_all_masks, list_names

def create_mask(int_array, shape):
    result = np.zeros((shape[0], shape[1]), dtype=np.float32)
    positions = map(int, int_array[0::2])
    length = map(int, int_array[1::2])
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for pos, le in zip(positions, length):
        mask[pos:(pos + le)] = 1
    result[:, :] = mask.reshape(shape[0], shape[1], order='F')
    
    binary_mask = result.astype(np.uint8)
    return binary_mask


def convert_masks_to_xy(masks, class_labels):
    list_with_all_masks = []
    list_names = []
    for i,mask in enumerate(masks):
        xy_arr = convert_mask_to_polygons(mask)
        for xy_sequence in xy_arr:
            list_with_all_masks.append(xy_sequence)
            list_names.append(int(class_labels[i]))
            
    return list_with_all_masks, list_names

def convert_mask_to_polygons(binary_mask) -> list[list[int | float]]:

    annotations = []

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    wh = np.flip(np.array(binary_mask.shape)) # for normalization purposes
    
    for contour in contours:

        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        
        #normalization
        contour_approx=contour_approx / wh
        
        polygon = contour_approx.flatten().tolist()
        annotations.append(polygon)
    return annotations




#Функция визуализации маски 
def visualize(image, keypoints):
    h,w = image.shape[:2]
    cpy = image.copy()
    
    for keypoint in keypoints:
        poly = np.asarray(keypoint,dtype=np.float16).reshape(-1,2) # Read poly, reshape
        poly *= [w,h] # Unscale
        
        cv2.polylines(cpy, [poly.astype('int')], True, (255, 0, 0), 2) # Draw Poly Lines
    
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(cpy)
    plt.show()
    
train_df_n, val_df_n = train_test_split(train_df, test_size=0.1, stratify=train_df["count"], random_state=54)




# Задаем формат для изображения
import albumentations as A 

img_size = 640

train_transform = A.Compose([
    A.Resize(width=img_size, height=img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=45, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

val_transform = A.Compose([
    A.Resize(width=img_size, height=img_size),
])


row_id=12

fname, onlyname, image, masks, class_labels = read_image_with_masks(row_id, train_df_n)

transformed = train_transform(image=image, masks=masks)
transformed_image = transformed['image']
transformed_masks = transformed['masks']

transformed_keypoints, transformed_class_labels = convert_masks_to_xy(transformed_masks, class_labels)


visualize(transformed_image, transformed_keypoints)

import os
import shutil



# Создаем директории для обучения/тестирования
if os.path.isdir('images'):
    shutil.rmtree('images')
if os.path.isdir('labels'):
    shutil.rmtree('labels')
os.mkdir ("images")
os.mkdir ("labels")
os.mkdir ("images/train")
os.mkdir ("images/val")
os.mkdir ("labels/train")
os.mkdir ("labels/val")

def transform_element(transformM, image, masks, class_labels, onlyname, outdir1, outdir2, img_size, ii=None):
    transformed = transformM(image=image, masks=masks)
    
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']
    transformed_class_labels = class_labels
    transformed_name = onlyname
    if ii:
        transformed_name += '_'+str(ii)
    nn=os.path.basename(os.path.normpath(transformed_name))
    cv2.imwrite(outdir1 + nn +'.jpg', cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)) 
    out_file = open(outdir2 + nn +'.txt', 'w')
    transformed_keypoints, transformed_class_labels = convert_masks_to_xy(transformed_masks, transformed_class_labels)
    for iii in range(len(transformed_keypoints)):
        kOut=transformed_keypoints[iii]
        clOut=transformed_class_labels[iii] 
        if(len(kOut)<6):
            continue

        text=str(clOut) + " " + " ".join([str(b) for b in kOut]) + '\n'
        out_file.write(text)
    out_file.close()




# Искусственно увеличиваем датасет
outdir1 = "./images/train/"
outdir2 = "./labels/train/"
AUG_COUNT=5
for i in range(len(train_df_n)):
    print(train_df_n["ImageId"])
    fname, onlyname, image, masks, class_labels = read_image_with_masks(i,train_df_n)
    transform_element(val_transform, image, masks, class_labels, onlyname, outdir1, outdir2, img_size)
    
    for ii in range(AUG_COUNT):
        transform_element(train_transform, image, masks, class_labels, onlyname, outdir1, outdir2, img_size, ii)
        
outdir1 = "./images/val/"
outdir2 = "./labels/val/"
for i in range(len(val_df_n)):
    fname, onlyname, image, masks, class_labels = read_image_with_masks(i,val_df_n)
    transform_element(val_transform, image, masks, class_labels, onlyname, outdir1, outdir2, img_size)
    
    
    
# Задаем файл конфигураций для обучения
f = open("./train.yaml", "w")
f.write('path: '+init_dir+"\n")
f.write('train: images/train'+"\n")
f.write('val: images/val'+"\n")
f.write('names:'+"\n")
f.write('  0: 0'+"\n")
f.write('  1: 1'+"\n")
f.write('  2: 2'+"\n")
f.write('  3: 3'+"\n")
f.write('  4: 4'+"\n")
f.close()



# Задаем файл конфигураций модели
a = """

nc: 5  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov9c-seg.yaml' will call yolov9-seg.yaml with scale 'c'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

# YOLOv9.0с backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv9.0с head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Segment, [nc, 32, 256]]  # Segment(P3, P4, P5)"""

f = open("yolov9-seg.yaml", "w", encoding="utf-8")
f.write(a)
f.close()


# Само обучение модели
if os.path.isdir('./runs/segment/train'):
    shutil.rmtree('./runs/segment/train')
    
from ultralytics import YOLO
model = YOLO('yolov9c-seg.yaml')
results = model.train(data='train.yaml', epochs=30, batch=16, pretrained=False)


# Вывод графиков результатов
image = np.array(Image.open('./runs/segment/train/results.png'))
plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.show(image)


#Тестирование обученной модели
img_names = sorted(glob.glob(TEST_IMAGE_DATASET_PATH + "/*.jpg"))

model = YOLO('train6/weights/last.pt') 
for i in img_names:
    test_img = i
    img = Image.open(test_img)
    draw = ImageDraw.Draw(img)

    pred_results = model(test_img)

    if(pred_results[0].masks):
        masks = pred_results[0].masks.cpu()
        for mask in masks:
            #maskt = mask.data[0].numpy()
            for polygon in mask.xy:
                #mask_img = Image.fromarray(maskt,"I")
                draw.polygon(polygon,outline=(0,255,0), width=5)
                plt.imshow(img)
                plt.show()