import pandas as pd
import requests as re
from bs4 import BeautifulSoup
import os
import json
import numpy as np

def makeCall(path, base='https://rxnav.nlm.nih.gov/REST/', query=''):
    return json.loads(re.get(base + path + query).content)

def getPillProducts():
    #get drug form groups
    dfg = pd.DataFrame(makeCall('allconcepts.json', query = '?tty=DFG')['minConceptGroup']['minConcept'])
    
    #get rxcui associated with drug form group 'Pill'
    pill_rxcui = dfg[dfg['name'] == 'Pill']['rxcui'].item()

    #get drug forms related to drug form group, Pill
    pill_forms = pd.concat(list(map(lambda x: pd.DataFrame(x['conceptProperties']), makeCall('rxcui/' + pill_rxcui + '/related.json', query='?rela=doseformgroup_of')['relatedGroup']['conceptGroup'])))

    #filter out everything that isn't a human drug
    human_pills = pill_forms['rxcui'].map(lambda x: makeCall('rxcui/' + x + '/filter.json',query='?propName=RXNAV_HUMAN_DRUG'))
    human_pills = pd.json_normalize(human_pills).dropna().drop_duplicates()

    #get rxcui associated with pill products
    product_terms = 'SCD+SBD+GPCK+BPCK'
    pill_products = human_pills['rxcui'].map(lambda x: makeCall('rxcui/' + x + '/related.json', query= '?tty=' + product_terms))
    pill_products = pd.json_normalize(pd.json_normalize(pd.json_normalize(pill_products)['relatedGroup.conceptGroup'].explode())['conceptProperties'].dropna().explode()).drop_duplicates()
    return pill_products

def getNDCProps(codes):
    return codes.map(lambda x: makeCall('ndcproperties.json', query='?id=' + x))


if __name__ == "__main__":
    products = getPillProducts()
    properties = getNDCProps(products['rxcui'])
