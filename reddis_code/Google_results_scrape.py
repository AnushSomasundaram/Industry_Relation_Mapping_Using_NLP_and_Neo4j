from urllib import response
import requests 
import urllib
from requests_html import HTML
from requests_html import HTMLSession


def scrape_google(query):

    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.co.uk/search?q=" + query)
    links = list(response.html.absolute_links)

    google_domains = ('https://www.google.', 
                      'https://google.', 
                      'https://webcache.googleusercontent.', 
                      'http://webcache.googleusercontent.', 
                      'https://policies.google.',
                      'https://support.google.',
                      'https://maps.google.')

    for url in links[:]:
        if url.startswith(google_domains):
            links.remove(url)
    
    return links
    

def get_source(url):
    """
    return the source code for provided URL.

    Args:
        url(string): URL of the page to scrape
    
    Returns:
        response (object): HTTP respose object from request_html
    
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response
    except requests.exceptions.RequestException as e :
        print(e)


#print(scrape_google("data science blogs"))











