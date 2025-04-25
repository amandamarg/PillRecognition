# PillRecognition
download.py: functions/script for downloading ndc codes and related information

getImages.py: scrapes and saves images from Drugs.com using ndc codes downloaded from https://api.fda.gov and saved in ./drug/ndc/drug-ndc-0001-of-0001.json. Images are saved in /.drug/img/{ndc}/

RxNormAPI.py: makes calls to https://rxnav.nlm.nih.gov/REST/ to get codes and property information for (human) pills. Includes function that can be used to parse the response to make it more readable and get rid on unnecessary columns. 