import requests
requests.packages.urllib3.disable_warnings()

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
def googleSearch(query):
    with requests.session() as c:
        url = 'https://www.google.co.in'
        query = {'q': query}
        urllink = requests.get(url, params=query)
        print (urllink.url)

googleSearch('Linkin Park')