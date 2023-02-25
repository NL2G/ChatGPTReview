import requests
import pandas as pd
import xmltodict

# Parse Arxiv Papers into a TSV file. Currently without pagination as there were less arxiv papers at query time
r = xmltodict.parse(requests.get(
    "http://export.arxiv.org/api/query?search_query=all:chatgpt&max_results=50"
).content)["feed"]["entry"]

df = pd.DataFrame(r)
df.to_csv("chatgpt.tsv", sep="\t")
