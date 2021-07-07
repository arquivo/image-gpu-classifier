import csv
import requests
import re
import json
import sys

class MetaClassifierBlocked():
"""Check if this record matches Arquivo's block list"""

    def __init__(self, block_list):
        download = requests.get(block_list)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        self.regexes = []
        # remove unnecessary stuff from the URL regexes in the CSV to match URLs better 
        for row in my_list:
            r = row[0].strip()
            if r.endswith("/"):
                r = r[:-1]
            r = r.replace("?","\?").lower()
            self.regexes.append(re.compile("^"+r+".*"))

    def classify(self, jsonline):
        jsonline['blocked'] = 0
        # fields from which to try and match the URL regexes
        fields = ['pageHost', 'imgUrl', 'pageUrl']
        for field in fields:
            if field in jsonline:
                url = re.sub('^http[s]?://(www.)?', '', jsonline[field].lower())
                for regex in self.regexes:
                    if regex.search(url):
                        jsonline['blocked'] = 1
                        return jsonline
        return jsonline
