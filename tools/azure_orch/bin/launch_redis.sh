sudo docker run -p 6379:6379 -v /codepoint/tools/azure_orch/redis_conf/redis.conf:/redis.conf --name maro_redis -d redis redis-server /redis.conf