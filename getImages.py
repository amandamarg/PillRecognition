import pandas as pd
import requests as re
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path


def checkNDCMatch(ndc1,ndc2):
    ndc1Parts = list(map(int, ndc1.split('-')))
    ndc2Parts = list(map(int, ndc2.split('-')))
    return ndc1Parts == ndc2Parts

def getImgLinks(ndc):
    soup = BeautifulSoup(re.get('https://www.drugs.com/imprints.php?ndc=' + ndc).content)
    items = soup.find_all('dt', string='NDC')
    img_links = []
    for i in items:
            ndc_check = i.next_sibling.getText()
            if checkNDCMatch(ndc,ndc_check):
                path = i.parent.find('dt', string='Imprint').nextSibling.find('a').get('href')
                imgs = BeautifulSoup(re.get('https://www.drugs.com' + path).content).find(class_ = "ddc-pid-info-main").find_all('img')
                for im in imgs:
                    img_links.append(im.get('src'))
    return img_links

def getImgs(ndc):
    img_links=getImgLinks(ndc)
    path = os.getcwd() + '/drug/img/' + ndc + '/'
    Path.mkdir(Path(path), parents=True, exist_ok=True)
    count = 0
    for img in img_links:
        if not 'no-image-placeholder' in img:
            with open(path + ndc + '-' + str(count) + '.jpg', 'wb') as file:
                file.write(re.get(img).content)
            file.close()
            count += 1

    #if directory is empty, remove it
    if len(os.listdir(path)) == 0:
        os.removedirs(path)

if __name__ == "__main__":
    with open('./drug/ndc/drug-ndc-0001-of-0001.json', 'r') as file:
        data = json.loads(file.read())
    file.close()
    df = pd.DataFrame(data['results'])

    #filter out anything that isn't a tablet or capsule
    pills = df[df['dosage_form'].str.contains('TABLET') | df['dosage_form'].str.contains('CAPSULE')]
    
    #filter out anything that isn't for humans
    pills = pills[pills['product_type'].str.contains('HUMAN')]

    #loop over all codes in pills and get images
    for code in pills['product_ndc']:
        try:
            getImgs(code)
        except:
            print('Error: ' + code)
            #Errors occur with two ndc codes: 41167-1031 and 50090-6198
            #41167-1031 is for rolaids chews (not a pill)
            #50090-6198 is for Esomeprazole Magnesium, but is not on Drugs.com at this time

        