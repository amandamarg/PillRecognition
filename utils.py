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




''' 
Checks that the numeric label in each label file in file_paths corresponds to mapping represented by semantic_labels
** Note: requires that the semantic label must be able to be parsed from the label file name/path using semantic_label_parser **


semantic_labels: a list of semantic labels ordered such that the index of each label corresponds to its numeric label equivelant
file_paths: a list of file_paths to labels 
    *each file in file_paths contains  only one label corresponding to a single instance of a single
    *labels are formatted with the numeric label first and the segmentation label second, seperated by a single space
semantic_label_parser: a function that parses semantic label from path in file_path
log_change: if True then function will log any changes made to log_file_path. default True. 
log_file_path: path to file where changes will be logged if log_change is True. Ignored if log_change is False. default './log_changes.txt'. 
overwrite: if True and then overwrites file at log_file_path with header and any changes, otherwise will just append any new changes without overwriting existing file. Ignored if log_change is False. default True.
'''

def checkLabelMapping(semantic_labels, file_paths, semantic_label_parser, log_change, log_file_path, overwrite):
        
    if log_change and overwrite:
        #clears change log and writes header to change log
        log(log_file_path, "file,original content,updated content", overwrite=True)

    for path in file_paths:
        semantic_label = semantic_label_parser(path)
        idx = semantic_labels.index(semantic_label)

        #reads in file data
        with open(path, 'r') as file:
            content = file.read()
        file.close()
        data = content.split(' ')
        numeric_label = data[0]

        #checks if numeric label in lable file matchs the semantic_label index and if not, overwrites 
        if numeric_label != str(idx):
            data[0] = str(idx)
            updated_content = (' ').join(data)

            #overwrites file with updated class
            with open(path, 'w') as file:
                file.write(updated_content)
            file.close()

            #logs change
            if log_change:
                log_data = (',').join([path, content, updated_content])
                log(log_file_path, log_data)
            
