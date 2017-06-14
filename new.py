

from urllib.request import urlopen
import re

file=open('new.txt','r').read().split('\n')
fileArr=[open('rating0.txt','a'),open('rating1.txt','a'),open('rating2.txt','a'),open('rating3.txt','a'),open('rating4.txt','a'),open('rating5.txt','a')]

count=1
cnt=0
while count<250: 
	print(count)
	html = urlopen("https://fidelity-ssl.ugc.bazaarvoice.com/5508tridion-en_us/007/reviews.djs?format=embeddedhtml&page="+str(count)+"&scrollToTop=true").read()
	print(count)
	

	for each in re.findall(r'<span itemprop=\\\\"ratingValue\\\\" class=\\\\"BVRRNumber BVRRRatingNumber\\\\">(.*?)<\\\\/span>',str(html))[1:]:
		#print(each)
		fileArr[int(each)].write(file[cnt]+'\n')
		cnt=cnt+1
	count=count+1
		
