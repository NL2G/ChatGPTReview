import pandas as pd
import requests, json

# Write papers from Scholar to tsv. Manual pagination, as it only requires 2 calls (less than 200 papers at query time)
url = "https://api.semanticscholar.org/graph/v1/paper/search?query=chatgpt&limit=100&fields=authors,title,abstract,venue,publicationDate"
r = requests.get(url)
content = json.loads(r.content)["data"]

url = "https://api.semanticscholar.org/graph/v1/paper/search?query=chatgpt&limit=100&offset=100&fields=authors,title,abstract,venue,publicationDate"
r = requests.get(url)
content += json.loads(r.content)["data"]

df = pd.DataFrame(content)
df.to_csv("scholar_results_aes", sep="\t")
