import sys
import os
import json
import ast
from time import sleep
import pprint
#import nltk
import ast
import unicodedata
import time
import logging
import requests


logger = logging.getLogger(__name__)

class GoogleConfig:
    def __init__(self, api_key=None, cse_id=None, site='', num=10, max_docs=10):
        if 'API_KEY' in os.environ:
            api_key = os.environ['API_KEY']
        if 'SEARCH_ID' in os.environ:
            cse_id = os.environ['SEARCH_ID']
        assert(api_key is not None and cse_id is not None)
        
        self.api_key = api_key
        self.cse_id = cse_id
        self.site = site
        self.num = num
        self.max_docs = max_docs
        self.credible_domains = ["wsj.com", "washingtonpost.com", "nytimes.com", "bbc.com", "economist.com", "newyorker.com", "ap.org", "reuters.com", "bloomberg.com", "foreignaffairs.com", "theatlantic.com", "ceasefiremagazine.co.uk", "canadiandimension.com", "aljazeera.com", "taipeitimes.com", "france24.com", "indiatimes.com", "straitstimes.com", "egypttoday.com", "trtworld.com", "thelancet.com", "sciencemag.org", "journals.plos.org/plosmedicine", "journals.plos.org/plosbiology", "academic.oup.com/nar", "nature.com", "embopress.org", "jamanetwork.com", "cell.com", "ahajournals.org", "ashpublications.org", "bmj.com", "biorxiv.org", "medrxiv.org", "ncbi.nlm.nih.gov", "cdc.gov", "who.int"]
        self.credible_domains.extend(["theguardian.com", "un.org", "sciencedirect.com", "unesco.org", "nationalgeographic.org", "nationalgeographic.com", "abc.net.au", "forbes.com", "pewresearch.org", "pewtrusts.org", "emerald.com", "unicef.org", "bbc.co.uk", "ft.com"])
        self.credible_domains.extend(["climatecentral.org", "ipcc.ch", "carbonbrief.org", "climateinterpreter.org"])#climate change domains

    def __str__(self):
        return 'GoogleConfig(api_key={}, cse_id={}, site={}, num={}, max_docs={}'.format(self.api_key,
                                                                                         self.cse_id,
                                                                                         self.site,
                                                                                         self.num,
                                                                                         self.max_docs)

    def isCredibleDomain(self, domain):
        if domain.endswith(tuple(self.credible_domains)):
            return True
        if ".ac." in domain or domain.endswith(".edu") or domain.endswith(".gov") or ".gov." in domain:
            return True
        print(domain)
        return False

def google_search(search_term, api_key, cse_id, num):

    r = requests.get(f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={search_term}' )
    res = r.json()
    #print(res)

    if "error" in res:
        print(res["error"])
        sys.exit()

    if 'items' in res:
        return res['items']
    else:
        return []

def getDocumentsForClaimFromGoogle(claim, google_config):
    results = google_search(google_config.site+''+claim, google_config.api_key,
                            google_config.cse_id, num=google_config.num)
    logger.info(google_config)
    res = []
    res_snippet = []
    c = 0
    for result in results:
        if result["link"].endswith(".pdf") or "snippet" not in result:
            continue
        try:
            logger.info(result['displayLink'])
        
            #if 'https://en.wikipedia.org/wiki/' in result['formattedUrl'] and c<google_config.max_docs:
            if 'en.wikipedia.org' in result['displayLink'] and c<google_config.max_docs:
                b = result['formattedUrl'].split('/')[-1].replace(' ', '')#replace('https://en.wikipedia.org/wiki/','')
                c = c+1
                b = b.replace('(','-LRB-')
                b = b.replace(')','-RRB-')
                b = b.replace(':','-COLON-')
                b = b.replace('%26','&')
                b = b.replace('%27',"'")
                b = b.replace('%22','"')
                b = b.replace('%3F','?')
                res.append(b)
                #snippet_words = result["snippet"].strip().replace('\n', '').split(" ")
                #res_snippet.append(" ".join([word for word in snippet_words if not word.startswith("\\u")]))
                res_snippet.append(result["snippet"].strip().replace('\n', '').replace("\u00a0...",""))
            elif c < google_config.max_docs and google_config.isCredibleDomain(result['displayLink']):
                #print(result['displayLink'])
                c = c+1
                b = result['displayLink'] #result['formattedUrl'].split('/')[-1].replace(' ', '')
                res.append(b)
                #snippet_words = result["snippet"].strip().replace('\n', '').split(" ")
                #res_snippet.append(" ".join([word for word in snippet_words if not word.startswith("\\u")]))
                res_snippet.append(result["snippet"].strip().replace('\n', '').replace("\u00a0...",""))
        except:
            print(result)
            import pdb; pdb.set_trace()
            #sys.exit()

    return res, res_snippet


def getDocsForClaim(claim, google_config):
    # try:
    docs_google, snippets  = getDocumentsForClaimFromGoogle(claim, google_config)
    # except Exception:
    #     docs_google = []
   
    docs = []
    for elem, snippet in zip(docs_google, snippets):
        if 'disambiguation' not in elem or 'List_of_' not in elem:
            #docs.append(elem)
            docs.append(snippet)
                        
    docs = [[d] for d in docs ]
    #return dict(predicted_pages=docs, predicted_google=docs_google)
    return dict(sentences=snippets)


def getDocsBatch(file,google_config):
    for line in open(file):
        line = json.loads(line.strip())
        line.update(getDocsForClaim(line['claim'],google_config))
        yield line
