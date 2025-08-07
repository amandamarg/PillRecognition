import pandas as pd
import cv2
import json
from jsonschema import validate
import ultralytics
from torch.utils.data import Dataset
from PIL import Image
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from glob import glob
import os

def readLabelsObjDetect(path):
    with open(path, 'r') as f:
        labels = list(map(lambda x: x.rstrip().split(' '),f.readlines()))
    f.close()
    labels = pd.DataFrame(labels, columns=['class', 'x', 'y', 'w', 'h'])
    labels = labels.apply(lambda x: x.astype(pd.Int64Dtype()) if x.name == 'class' else x.astype(pd.Float64Dtype()))
    return labels

def drawBoundingBoxes(img, labels):
    img_with_boxes = img.copy()
    height, width, __ = img.shape
    for __,row in labels.iterrows():
        __,x,y,w,h = row.values
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = x1 + int(w * width)
        y2 = y1 + int(h * height)
        #draws a green rectangle of thickness 5 on (copy of) img
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 5)
    '''
    x1 = labels['x'].sub(labels['w']).mul(width).astype(pd.Int64Dtype())
    y1 = labels['y'].sub(labels['h']).mul(height).astype(pd.Int64Dtype())
    x2 = x1.add(labels['w'].mul(width).astype(pd.Int64Dtype()))
    y2 = y1.add(labels['h'].mul(height).astype(pd.Int64Dtype()))
    for i in range(0,labels.shape[1]):
        cv2.rectangle(img_with_boxes, (x1[i], y1[i]), (x2[i], y2[i]), (0, 255, 0), 5)
    '''
    return img_with_boxes


'''
adds dataset info to datasets.json
'''
def addToDatasets(label_type, dataset_name, root_path, yaml_path):
    with open('./datasets.json', 'r') as file:
        data = json.loads(file.read())
    file.close()

    data[dataset_name] = {'root_path': root_path, 'yaml_path': yaml_path, 'label_type': label_type}

    with open('./datasets.json', 'w') as file:
        json.dump(data, file)
    file.close()    


'''
logs content into file at log_file_path
if overwrite is Ture, overwrites file if it exists, otherwise just appends 
'''
def log(log_file_path, content, overwrite=False):
    mode = 'w' if overwrite else 'a'
    with open(log_file_path, mode) as log_file:
        log_file.write(content + '\n')
    log_file.close()


'''
checks if labels in file at file_path match labels in class_labels and returns result
    ** assumes each line in file corresponds to label for a seperate instance of a class
    ** labels are formatted with the class label first and the segmentation label second, seperated by a single space

file_path: path to file containing labels
class_labels: correct class label(s) to check against
    * if class_labels is a list, must contain exactly one label for each line in file 
    * if class_labels is not a list, then file must only contain one label
fix: if True and labels in file don't match labels in class_labels, overwrites file with correct class_labels
log_change: if True then function will log any changes made to log_file_path. default True.
log_file_path: path to file where changes will be logged if log_change is True. Ignored if log_change is False. default './log_changes.txt'. 
'''
def labels_match(file_path, class_labels, fix=True, log_change=True, log_file_path='./log_changes.txt'):
    with open(file_path, 'r') as file:
        if isinstance(class_labels, list):
            orig_content = file.readlines()
            data = list(zip(class_labels, orig_content))
            updated_content = list(map(lambda x: str(x[0]) + ' ' + (' ').join(x[1].split(' ')[1:]), data)) 
            orig_content = ('').join(orig_content)
            updated_content = ('').join(updated_content)
        else:
            orig_content = file.read()
            data = orig_content.split(' ')
            updated_content = str(class_labels) + ' ' + (' ').join(data[1:])

    file.close()
    if orig_content == updated_content:
        return True
    elif fix:
        with open(file_path, 'w') as file:
            file.write(updated_content)
        file.close()
        if log_change:
            log_data = (',').join([file_path, orig_content, updated_content])
            log(log_file_path, log_data)
    return False

'''
validates a json object against a schema

instance is either an instance of a json object or a path to a json file
    * if instance is a file_path, then will return contents of file as json object
schema_file_path is a path to a json schema
'''
def validateJSON(instance, schema_file_path):
    with open(schema_file_path, 'r') as file:
        schema = json.load(file)
    file.close()
    if isinstance(instance, str):
        with open(instance, 'r') as file:
            json_obj = json.load(file)
        file.close()
        validate(instance=json_obj, schema=schema)
        return json_obj
    else:
        validate(instance=instance, schema=schema)


def zeroPadFront(x, desiredLength):
    x = str(x)
    while len(x) < desiredLength:
        x = '0' + x
    return x

def get_terminology(list_name):
    return pd.read_xml("./terminology_lists/" + list_name, iterparse={"choice": ["label", "value"]}, names=["name", "code"])

def getNormalizeTransform(x):
    images = np.stack(x.map(Image.open))
    mean = np.mean(images, axis=(0,1,2))
    std = np.std(images, axis=(0,1,2))
    return transforms.Normalize(mean, std)
    

 
class PillDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, target_transform=None):
        self.img_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = (self.img_paths.iloc[idx])
        image = Image.open(image_path)
        label = self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def load_epillid(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', path_to_folds = 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base', train_with_side_labels = True, encode_labels=True):
    data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata'
    csv_files = glob(os.path.join(data_root_dir, path_to_folds, '*.csv'))

    all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
    csv_files = sorted([x for x in csv_files if not x.endswith("all.csv")])
    test_imgs_csv = csv_files.pop(-1)
    val_imgs_csv = csv_files.pop(-1)

    all_images_df = pd.read_csv(all_imgs_csv)
    val_df = pd.read_csv(val_imgs_csv)
    test_df = pd.read_csv(test_imgs_csv)

    img_dir = 'classification_data'
    for df in [all_images_df, val_df, test_df]:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(data_root_dir, img_dir, x))

    if train_with_side_labels:
        all_images_df['label'] = all_images_df.apply(lambda x: x['label'] + '_' + ('0' if x.is_front else '1'), axis=1)
        val_df['label'] = val_df.apply(lambda x: x['label'] + '_' + ('0' if x.is_front else '1'), axis=1)
        test_df['label'] = test_df.apply(lambda x: x['label'] + '_' + ('0' if x.is_front else '1'), axis=1)

    if encode_labels:
        label_encoder = LabelEncoder().fit(all_images_df.label)
        all_images_df['encoded_label'] = label_encoder.transform(all_images_df.label)
        val_df['encoded_label'] = label_encoder.transform(val_df.label)
        test_df['encoded_label'] = label_encoder.transform(test_df.label)

    val_test_image_paths = list(val_df['image_path'].values) + list(test_df['image_path'].values)
    train_df = all_images_df[~all_images_df['image_path'].isin(val_test_image_paths)]

    return {'train': train_df, 'val': val_df, 'test': test_df}


from itertools import combinations

def generate_positive_pairs(df, labelcol="pilltype_id", enforce_ref_cons=True, enforce_same_side=True, side=None):
    '''
    enforce_ref_cons=True will return only pairs with exactly one reference image and one consumer image
    enforce_same_side=True will return only pairs where both images in the pair are the same side
    side can be one of ['front', 'back', None], and if not None, will return just the pairs corresponding to the side that is passed.
    Note: if side is not None, will override enforce_same_side and return only same-sided pairs for that side.
    '''

    df_copy = df.copy().reset_index()
    indicies = df_copy.index.values
    pairs = np.array(list(combinations(indicies, 2)))
    pairs = pairs[np.equal(df_copy.iloc[pairs[:,0]][labelcol].to_numpy(), df_copy.iloc[pairs[:,1]][labelcol].to_numpy())]

    assert side in ['front', 'back', None]

    if side == 'front':
        pairs = pairs[np.logical_and(df_copy.iloc[pairs[:,0]].is_front.to_numpy(), df_copy.iloc[pairs[:,1]].is_front.to_numpy())]
    elif side == 'back':
        pairs = pairs[np.logical_and(~df_copy.iloc[pairs[:,0]].is_front.to_numpy(), ~df_copy.iloc[pairs[:,1]].is_front.to_numpy())]
    elif enforce_same_side:
        front_side_pairs = np.logical_and(df_copy.iloc[pairs[:,0]].is_front.to_numpy(), df_copy.iloc[pairs[:,1]].is_front.to_numpy())
        back_side_pairs = np.logical_and(~df_copy.iloc[pairs[:,0]].is_front.to_numpy(), ~df_copy.iloc[pairs[:,1]].is_front.to_numpy())        
        pairs = pairs[np.logical_or(front_side_pairs, back_side_pairs)]

    if enforce_ref_cons:
        pairs = pairs[np.logical_xor(df_copy.iloc[pairs[:,0]].is_ref.to_numpy(), df_copy.iloc[pairs[:,1]].is_ref.to_numpy())]
    
    return pairs
    