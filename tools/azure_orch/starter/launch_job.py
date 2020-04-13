import sys 
sys.path.append("..") 

from provison import generate_job_config
from docker import allocate_job

if __name__ == "__main__":
    generate_job_config()
    allocate_job()