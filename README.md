# ChatGPTReview

Code for our analysis of paper abstracts can be found in paper_analysis.  
`paper_analysis/paper_crawler_arxiv.py` contains the code we used to query paper abstracts from Arxiv  
`paper_analysis/paper_crawler_scholar.py` contains the code we used to query paper abstracts from Schoar  
`paper_analysis/visualize.py` contains the code for majority votes and data visualization/analysis after our manual
annotation


Following data files are included:  
`paper_analysis/annotations/combined3.tsv` - Contains manually acquired annotations for paper titles and abstracts. The columns are:

* Annotator: We had different annotators. The field mixed annotators specifies that one of the annotators annotated a field. This is based on our annotation structure and the way the combined file was created.
* Venue: The venue of the resp. paper
* Authors: The authors of the resp. paper
* Title: The title of the resp. paper
* Abstract: The abstract of the resp. paper
* Performance Sentiment: The Performance Sentiment (0, 1, 2, 3, 4, 5) annoated for the resp. paper. 0 is later replaced with NAN
* Topic 1: The topic of the resp. paper (ETHICS, EDUCATION, REST, EVALUATION, APPLICATION, MEDICAL)
* Topic 2: Secondary topic, in few cases chosen more freely. Usually uses the same distinction
* Social Sentiment: The social sentiment (Threat, Opportunity, Mixed, NAN) of the resp paper
* Real Abstract: An indicator of whether we labeled the shown text to be a real paper abstract

`paper_analysis/data/combined3.tsv` - Contains the original arxiv query, which we extended in our annotation by 3 days.
Columns are: id, updated, published, title, summary, author, arxiv:comment, link, arxiv:primary_category, category

`paper_analysis/data/old`, `paper_analysis/data/scholar_results.tsv`, `paper_analysis/data/scholar_results_cse` - Contain the original query from Semantic Scholar. `old` contains the papers that we annotated. It also contains Arxiv Papers, which we ignored, as we already annotated them in the query from Arxiv. The other two files were generated lateron to add timestamps to the original data. New papers published at this point are part of the list, but were not annotated anymore, hence are not part of the analysis.
Columns are: 	paperId, title, abstract, venue, authors for `old` and 	paperId, title, abstract, venue, publicationDate, authors for the other two files. Two other querys had to be run for timestamp matching as view papers in the Semantic Scholar API search results are non-deterministically selected.

