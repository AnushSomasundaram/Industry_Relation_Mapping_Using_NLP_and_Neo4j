
import redis
from redis import Redis
from rq import worker, Queue
from rq_scheduler import Scheduler

import essential_functions
#import reddis_queue

redis_server_connection = redis.Redis() 

scheduler= Scheduler(connection = redis_server_connection)

Entity_extraction_queue = Queue(name="Entity_extraction_queue",connection=redis_server_connection)

def get_keys_at_present():   
   
   keys = list(redis_server_connection.keys("*"))
   required_keys = []
   for key in keys:
      key = str(key)[2:]
      
      if key.startswith("http"):
         required_keys.append(key)   
   
   return required_keys
   


def enqueue_entitie_jobs():

   keys_required=get_keys_at_present()
   keys_already_accessed = []

   for key in keys_required:   
      if key not in keys_already_accessed:   
         Entity_extraction_queue.enqueue(essential_functions.Entity_Upload_with_redis_job_scheduling,key)
         keys_already_accessed.append(key)  
      else:
         pass

def enqueue_entitie_jobs_for_loop(keys):

      for key in keys:
           Entity_extraction_queue.enqueue(essential_functions.Entity_Upload_with_redis_job_scheduling,key)

def worker_process_for_entities_from_text():
   keys=get_keys_at_present()
   #for key in keys :   
   enqueue_entitie_jobs_for_loop(keys)
     


worker_process_for_entities_from_text()
#print(get_keys_at_present())





    
      

      



      

