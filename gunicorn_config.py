import multiprocessing

bind = ":8080"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornH11Worker'
worker_connections = 1000
timeout = 300
max_requests = 1000
backlog = 2048
threads = 1
log_level = "debug"
