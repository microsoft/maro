mkdir -p ./data/grafana
CURRENT_UID=$(id -u):$(id -g) docker-compose up -d 
