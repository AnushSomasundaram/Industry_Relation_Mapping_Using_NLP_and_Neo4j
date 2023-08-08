import queue
from flask import Flask, request
import redis
from rq import Queue
import Google_results_scrape
import essential_functions


app = Flask(__name__)

r = redis.Redis()
q = Queue(connection=r)

topic_of_search = input("Enter the topic of search :- ")
links = Google_results_scrape.scrape_google(topic_of_search)

@app.route("/task")
def add_to_queue():
   
   for url in links:   
      
      job=q.enqueue(essential_functions.data_upload_with_redis_job_scheduling,url)
      q_len = len(q)
      return f"Task {job.id} added to queue at {job.enqueued_ate}. {q_len} Tasks in queue."

    

if __name__ == "__main__":
   app.run()
