import redis
from redis import Redis
from rq import Queue
import Google_results_scrape
from rq_scheduler import Scheduler
import time
import essential_functions

redis_server = redis.Redis(host="localhost",port=6379)
scheduler= Scheduler(connection = redis_server)


text_extraction_queue=Queue(name="text_extraction_queue",connection = redis_server)


topic_of_search = input("Enter the topic of search :- ")
links = Google_results_scrape.scrape_google(topic_of_search)


def add_to_queue(url):
   
   job=text_extraction_queue.enqueue(essential_functions.data_upload_with_redis_job_scheduling,url)
   
   
    
def worker_process_for_text(links):
   
   for url in links:
      add_to_queue(url)

worker_process_for_text(links)

