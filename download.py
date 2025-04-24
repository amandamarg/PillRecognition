import requests as re
import json
from pathlib import Path
import zipfile
from io import BytesIO

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
    response = re.get(link, headers=user_agent)
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
    downloadJSON = json.loads(re.get("https://api.fda.gov/download.json").content)
    partitions = downloadJSON["results"][endpoint1][endpoint2]["partitions"]
    for part in partitions:
        link = part["file"]
        downloadZip(link, path, keepZipped)

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
    
    






        
