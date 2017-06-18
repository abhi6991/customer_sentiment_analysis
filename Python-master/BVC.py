# import requests 
# requests.packages.urllib3.disable_warnings()

# def getAnalysis(API_Key,WavPath):
#     res = requests.post("https://token.beyondverbal.com/token",data={"grant_type":"client_credentials","apiKey":API_Key})
#     token = res.json()['access_token']
#     headers={"Authorization":"Bearer "+token}
#     pp = requests.post("https://apiv3.beyondverbal.com/v3/recording/start",json={"dataFormat": { "type":"WAV" }},verify=False,headers=headers)
#     recordingId = pp.json()['recordingId']
#     with open(WavPath,'rb') as wavdata:
#         r = requests.post("https://apiv3.beyondverbal.com/v3/recording/"+recordingId,data=wavdata, verify=False, headers=headers)
#         return r.json()



#commented code is what was present in their implementation --but it did not work

from urllib import request, parse
import json

def getAnalysis(API_Key,WavPath):
	data = parse.urlencode({"grant_type":"client_credentials","apiKey":API_Key}).encode()
	req =  request.Request("https://token.beyondverbal.com/token", data=data) # this will make the method "POST"
	x = request.urlopen(req)
	# print(x)
	raw_data = x.read()
	# print("raw_data:",raw_data)
	encoding = x.info().get_content_charset('utf8')  # JSON default
	# print("encoding:",encoding)
	data = json.loads(raw_data.decode(encoding))
	# print("data:",data['access_token'])
	token=data['access_token']
	headers={"Authorization":"Bearer "+token}
	print("headers",headers)
	data=parse.urlencode({"json":{"dataFormat": { "type":"WAV" }},"verify":False,"headers":headers}).encode()
	print("data:",data)
	req= request.Request("https://apiv3.beyondverbal.com/v3/recording/start",data=data)
	res=request.urlopen(req)
	print("resp:",res)

json = getAnalysis("370f78ac-ac02-4d3a-ab41-a8d221fb155b","abcd.wav")
print(json)
