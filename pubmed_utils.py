import re
import pandas as pd
from Bio import Entrez

def search_pubmed(query, email, retmax=1000):
    """Search PubMed using the given query."""
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle, validate=False)
    handle.close()
    pmid_list = record["IdList"]
    count = int(record["Count"])
    return pmid_list, count

def fetch_details(id_list, email):
    """Fetches details for a list of PubMed IDs."""
    if not id_list:
        return None
    Entrez.email = email
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
    papers = Entrez.read(handle, validate=False)
    handle.close()
    return papers

def papers_to_df(papers):
    """Converts the PubMed XML structure into a Pandas DataFrame."""
    if not papers or 'PubmedArticle' not in papers:
        return pd.DataFrame()

    rows = []
    for paper in papers['PubmedArticle']:
        article = paper['MedlineCitation']['Article']
        title = article.get('ArticleTitle', 'Not available')
        pubmed_id = paper['MedlineCitation']['PMID']
        journal = article['Journal'].get('Title', 'Not available')
        pubdate = article['Journal']['JournalIssue']['PubDate']
        year = pubdate.get('Year', 'Not available')
        authors_list = article.get('AuthorList', [])
        authors = ', '.join([f"{a.get('LastName', '')}, {a.get('ForeName', '')}" 
                             for a in authors_list])
        abstract_text = 'Not available'
        if 'Abstract' in article and 'AbstractText' in article['Abstract']:
            abstract_text = " ".join(article['Abstract']['AbstractText'])
        doi = 'Not available'
        for id_item in paper['PubmedData']['ArticleIdList']:
            if id_item.attributes.get('IdType') == 'doi':
                doi = str(id_item)
                break
        rows.append([
            title, 
            authors, 
            journal, 
            year, 
            pubmed_id, 
            doi, 
            abstract_text
        ])
    
    return pd.DataFrame(rows, columns=[
        'Title', 
        'Authors', 
        'Journal', 
        'Year', 
        'PMID', 
        'DOI', 
        'Abstract'
    ])