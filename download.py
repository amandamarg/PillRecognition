import requests
import json
from pathlib import Path
import zipfile
from io import BytesIO
import kagglehub
from roboflow import Roboflow
from sklearn.model_selection import train_test_split
import os
import yaml
from utils import addDatasetConfigs, log, labels_match
import glob
import re


'''
file: file to be unzipped as either BytesIO obj or path to file
unzipTo: path to directory to unzip files to
'''
def unzipFile(file, unzipTo):
    with zipfile.ZipFile(file, 'r') as zip:
        zip.extractall(unzipTo)
    zip.close()

def downloadZip(link, path='./', keepZipped = False):
    user_agent = {"User-agent": 'Mozilla/5.0'}
    response = requests.get(link, headers=user_agent)
    Path.mkdir(Path(path), parents=True, exist_ok=True)
    if keepZipped:
        filename = link.split('/')[-1]
        if not filename.endswith('.zip'):
            filename = response.headers.get('Content-Disposition').split('filename=')[-1]
        with open(path + filename, 'wb') as file:
            file.write(response.content)
        file.close()
    else:
        unzipFile(BytesIO(response.content), path)

'''
Download data from fda api and save to path
'''
def fdaApiDownload(endpoint1, endpoint2, path='./', keepZipped = False):
    downloadJSON = json.loads(requests.get("https://api.fda.gov/download.json").content)
    partitions = downloadJSON["results"][endpoint1][endpoint2]["partitions"]
    for part in partitions:
        link = part["file"]
        downloadZip(link, path, keepZipped)


'''
Downloads and configures taiwan_pill_for_label from Roboflow
'''
def downloadTaiwanPillDataset():
    dataset_path = os.getcwd() + "/datasets/taiwan_pill_for_label"

    #download from roboflow
    rf = Roboflow(api_key="zCMD7YHOUliH3cklpoLi")
    project = rf.workspace("doccampill").project("taiwan_pill_for_label")
    version = project.version(35)
    dataset = version.download("yolov11", location=dataset_path)


    #rename without added roboflow extension
    train_path_imgs = dataset_path + '/train/images/'
    train_path_labels = dataset_path + '/train/labels/'
    valid_path_imgs = dataset_path + '/valid/images/'
    valid_path_labels = dataset_path + '/valid/labels/'
    test_path_imgs = dataset_path + '/test/images/'
    test_path_labels = dataset_path + '/test/labels/'

    for file in os.listdir(train_path_imgs):
        old_name = os.path.basename(file)
        new_name = old_name.split('_')[0]
        os.rename(train_path_imgs + old_name, train_path_imgs + new_name + '.jpg')
        os.rename(train_path_labels + old_name.replace('.jpg', '.txt'), train_path_labels + new_name + '.txt')

    #make directories for test and valid
    os.makedirs(valid_path_imgs, exist_ok=True)
    os.makedirs(valid_path_labels, exist_ok=True)
    os.makedirs(test_path_imgs, exist_ok=True)
    os.makedirs(test_path_labels, exist_ok=True)

    #partition data into train/valid/test
    x = []
    y = []

    for file in os.listdir(train_path_labels):
        label = []
        with open(train_path_labels + file, 'r') as f:
            for line in f:
                label.append(line.split(' ')[0])
        f.close()
        if (len(set(label)) > 1):
            print('Error: multiple classes in file')
            print(file)
            break
        else:
            x.append(file)
            y.append(label[0])

    #split data (train data is stratified as to keep proportions similar to overall proportions)
    x_train, x_temp, y_train, y_temp = train_test_split(x,y,test_size=.2, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=.5, random_state=42)

    #move images and labels from valid partition into valid folder
    for file in x_val:
        os.rename(train_path_labels + file, valid_path_labels + file)
        os.rename(train_path_imgs + file.replace('.txt', '.jpg'), valid_path_imgs + file.replace('.txt', '.jpg'))

    #move images and labels from test into test folder
    for file in x_test:
        os.rename(train_path_labels + file, test_path_labels + file)
        os.rename(train_path_imgs + file.replace('.txt', '.jpg'), test_path_imgs + file.replace('.txt', '.jpg'))

    #edit yaml
    with open(dataset_path + '/data.yaml', 'r') as file:
        data = {'path': dataset_path}
        data.update(yaml.safe_load(file))
    file.close()

    data['names'] = {idx: val for idx, val in enumerate(data['names'])}
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    data['test'] = 'test/images'

    with open(dataset_path + '/data.yaml', 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)
    file.close()

    addDatasetConfigs('segmentation', 'taiwan_pills', dataset_path, dataset_path + '/data.yaml')


'''
Downloads and configures ogyeiv2 from kaggle
    ** also corrects incorrect class label **
'''
def download_ogyeiv2_dataset():
    # Download latest version of ogyeiv2 dataset from kaggle
    dataset_path = kagglehub.dataset_download("richardradli/ogyeiv2")
    print("Path to dataset files:", dataset_path)

    dataset_path = dataset_path + '/ogyeiv2/ogyeiv2'

    log_file_path='./log_changes.txt'

    #gets list of all label files
    file_paths = glob.glob(dataset_path + '/*/labels/*.txt', recursive=True)

    #function that gets the pill name given a path to pill file
    getPillName = lambda x: re.match(r'(.+)/labels/(.+)_(s|u)_\d+\.txt', x).group(2)

    #converts list of file paths to list of all pill names where index of each pill name corresponds to its' numeric label
    pill_names = sorted(set(map(getPillName, file_paths)))

    #clears log file and writes header
    log(log_file_path, "file,original content,updated content", overwrite=True)

    for path in file_paths:
        #gets pill name from path and gets index of that pill name
        name = getPillName(path)
        idx = str(pill_names.index(name))
        
        #checks if labels match and if not, fixes them
        labels_match(path, idx, fix=True, log_change=True, log_file_path=log_file_path)

    # write a yaml file for dataset
    data = {'path': dataset_path, 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images', 'names': dict(enumerate(pill_names))}

    with open(dataset_path + '/data.yaml', 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)
    file.close()

    #add to dataset_configs.json
    addDatasetConfigs('segmentation', 'ogyeiv2', dataset_path, dataset_path + '/data.yaml')



if __name__ == "__main__":
    #download drug ndc codes from https://api.fda.gov/
    fdaApiDownload("drug", "ndc", path="./drug/ndc/") 
   
    '''
    #download ndc product and packge codes using link (txt or xls format)
    ndctext = "https://www.accessdata.fda.gov/cder/ndctext.zip"
    ndcxls = "https://www.accessdata.fda.gov/cder/ndcxls.zip"
    downloadZip(ndcxls, path="./product_package/")
    '''
    
    '''
    #download fda terminology lists
    terminology = "https://www.fda.gov/media/86426/download"
    downloadZip(terminology, path="./")
    '''

    '''
    link = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/medical-pills.zip'
    downloadZip(link, './ultralytics_medical_pills/')
    '''

    #download vaipe dataset from kaggle
    # vaipe = kagglehub.dataset_download("kusnguyen/vaipe-dataset")

    #download minimal vaipe dataset from kaggle
    # vaipe_mini = kagglehub.dataset_download('anhduy091100/vaipe-minimal-dataset')
    
    






        
