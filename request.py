import requests
url='http://localhost:1800/predict_api'
r=requests.post(url,json={'House age':0,'Distance to the nearest MRT station':0,
                          'Number of convenience stores':0,'Latitude':0,'Longitude':0})
print(r.json())
