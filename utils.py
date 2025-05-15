import pandas as pd
import cv2
import json

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


def addDatasetConfigs(dataset_type, dataset_name, root_path, yaml_path):
    with open('./datasets_configs.json', 'r') as file:
        data = json.loads(file.read())
    file.close()

    data[dataset_type][dataset_name] = {'root_path': root_path, 'yaml_path': yaml_path}

    with open('./datasets_configs.json', 'w') as file:
        json.dump(data, file)
    file.close()    



def log(log_file_path, content, overwrite=False):
    mode = 'w' if overwrite else 'a'
    with open(log_file_path, mode) as log_file:
        log_file.write(content + '\n')
    log_file.close()