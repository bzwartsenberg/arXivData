#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:35:00 2018

@author: berend
"""

from bs4 import BeautifulSoup
import json
import requests

import xml.etree.ElementTree as ET
import time



class entry():
    """An arxiv metadata entry, ordered as a dictionary:
        keys:
            'id'
            'created'
            'authors'
            'title'
            'categories'
            'comments'
            'doi'
            'abstract'
            ''
        """
    def __init__(self,xml_element):
        
        self.data = {}

        for child in xml_element[1][0]:
            tag = child.tag[child.tag.rfind('}')+1:]
            
            if tag == 'authors':
                authornames = []
                for author in child:
                    try:
                        authornames.append([author[0].text,author[1].text])
                    except IndexError:
                        print('Only last name')
                        authornames.append([author[0].text])
                
                self.data[tag] = authornames
            
            else:
                self.data[tag] = child.text
    


class arxiv_harvester():
    """Class for downloading and saving all metadata from the arxiv"""
    def __init__(self,setname, savepath, min_req_time = 1):
        """"""
        pass
        self.doc_counter = 0
        self.last_time = time.time()
        self.save_counter = 0
        self.setname = setname
        self.savepath = savepath
        self.min_req_time = min_req_time
        
        #create link
        
        
    def next_request(self,resumption_token = None):
        """Make the next request"""
    
        #create link, request, and parse:        
        entries, new_resumption_token = self.parse_xml(self.get_xml(self.make_link(resumption_token)))
        
        #increase doc counter:
        self.doc_counter += len(entries)
        
        ##save entries:
        self.save_entries(entries)
        
        return new_resumption_token
    
    def parse_xml(self,content):
        """Interpret metadata for a request and return a list of entry objects and 
        a resumption token"""
        
        #parse xml:
        root = ET.fromstring(content)
        
        #get entries
        entries = self.get_articles(root)
        
        #check resumption token
        resumption_token = self.check_resumption_token(root)
    
        return entries,resumption_token
        
        
    def get_articles(self,root):
        """Get the articles out of xml root"""
        
        articles = [self.get_article(article_elem) for article_elem in root[2][:-1]]
        
        return articles
        
        
    def get_article(self,article_elem):
        """Get the article entry from xml element, return dictionary data"""
        
        return entry(article_elem).data
        
        
        
        
        
    def check_resumption_token(self,root):
        """Check if there is a resumption_token and return it's value"""
        if root[2][-1].tag == '{http://www.openarchives.org/OAI/2.0/}resumptionToken':
            resumption_token = root[2][-1].text
        else:
            print('No resumption output found, check algorithm, you clipped articles!')
        
        return resumption_token
        
        
    def get_xml(self,link):
        """return xml content from link"""
        #do some waiting before in case min request time hasn't passed
        while (time.time() - self.last_time) < self.min_req_time:

            pass
        print('Time passed since last time: %s' % time.time() - self.last_time)
        self.last_time = time.time()
        
        while True:
            print('Making request')
            r = requests.get(link)
            if r.status_code == 200:
                print('Exit code 200, continuing')
                break
            if r.status_code == 503:
                wait_time = float(BeautifulSoup(r.content, 'lxml').findAll('h1')[0].text.split(' ')[2])
                print('Maximum request reached, trying after %s seconds' % wait_time)
                time.sleep(wait_time)
                
            else:
                print('Exit code %s, retrying in 2 seconds' % r.status_code)
                time.sleep(2)
        
        return r.content

        
    
    def make_link(self, resumption_token = None):
        """Generate the next link"""
        if resumption_token is None:
            link = 'http://export.arxiv.org/oai2?verb=ListRecords&set=%s&metadataPrefix=arXiv' % self.setname
        else:
            link = 'http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=' + resumption_token
        return link
        
        
    def save_entries(self, entries):
        """Save list of entries as json"""
        
        with open(self.savepath + ('%04d' % self.save_counter) + '.json', 'w') as f:
            json.dump(entries,f)
            
        self.save_counter += 1
        
        
        
        
        
    def harvest_data(self,resumption_token = None, max_articles = None):
        """harvest and save:
            Args:
                resumption_token: token to restart from if algorithm fails
                max articles: maximum articles to grab"""
        self.doc_counter = 0
        nolimit = False
        if max_articles is None:
            max_articles = 0
            nolimit = True
            
        
        #loop for doc counters (or indefinitely). Doc_counter is updated by next_request
        while (self.doc_counter < max_articles) or nolimit:
            
            #call next_request
            resumption_token = self.next_request(resumption_token)
            
            #save resumption_token every successful iteration:
            with open('resumption_token.txt', 'w') as f:
                f.write(resumption_token)
            
            #if none, we are done
            if resumption_token is None:
                break
            
            
        
            
if __name__ == "__main__":
    setname = 'physics:cond-mat'
    savepath = 'data/physics_cond-mat_'
    min_req_time = 1.
    
    harvester = arxiv_harvester(setname, savepath, min_req_time)
    
    harvester.harvest_data(max_articles = 10000)
    
    
    