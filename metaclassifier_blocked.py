import csv
import requests
import re


class MetaClassifierBlocked():

    def __init__(self, block_list):
        download = requests.get(block_list)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        self.regexes = []
        for row in my_list:
            r = row[0]
            if r.endswith("/"):
                r = r[:-1]
            self.regexes.append(re.compile(r))

    def classify(self, jsonline):
        fields = ['pageHost', 'imgSrc']
        for regex in self.regexes:
            for field in fields:
                if field in jsonline and regex.search(jsonline[field].lower()):
                    jsonline['blocked'] = 1
                    return jsonline

        return jsonline
