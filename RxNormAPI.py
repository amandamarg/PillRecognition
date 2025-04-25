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


'''
takes in properties from api call and returns parsed version
if propNames is empty, will get all properties, otherwise just gets properties specified
'''
def parseProps(props, propNames=[]):
    propConcepts = props['propertyConceptList.propertyConcept']
    makePropTuple = lambda x: (x['propName'], x['propValue'])
    propConcepts = propConcepts.map(lambda x: dict(map(makePropTuple, x)), na_action='ignore')
    parsed = pd.json_normalize(propConcepts.values)
    parsed.index = props['ndc10']
    parsed = parsed.drop_duplicates()
    if len(propNames) != 0:
        parsed = parsed.get(propNames).drop_duplicates()
    return parsed

if __name__ == "__main__":
    #check if RxNorm_properties.json and if not, then create it
    if not os.path.exists('./RxNorm_properties.json'):
        products = getPillProducts()
        properties = getNDCProps(products['rxcui'])
        properties = pd.json_normalize(pd.json_normalize(properties)['ndcPropertyList.ndcProperty'].explode())
        properties.to_json('RxNorm_properties.json')
    else:
        properties = pd.read_json('RxNorm_properties.json')

    #parse properties to get just values associated with color, shape, size, and imprints and then save in file called parsedProperties.json
    parsedProperties = parseProps(properties, ['COLOR', 'SHAPE', 'SIZE', 'IMPRINT_CODE'])
    parsedProperties.to_json('./parsedProperties.json')

    
    

