import requests                                                                                                                                                                                                    

def googlesearch(searchfor):                                                                                                                                                                                       
    link = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&%s' % searchfor                                                                                                                              
    ua = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36'}                                                                
    payload = {'q': searchfor}                                                                                                                                                                                     
    response = requests.get(link, headers=ua, params=payload)                                                                                                                                                      
    print response.text                                                                                                                                                                                            

googlesearch('test')