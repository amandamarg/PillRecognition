import pandas as pd
import cv2
import json
from jsonschema import validate
import ultralytics

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
