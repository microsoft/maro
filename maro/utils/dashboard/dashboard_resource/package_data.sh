sudo rm -rf volumns/influxdb
sudo rm -rf volumns/grafana
sudo rm -rf panels/line_chart/node_modules
sudo rm -rf panels/heatmap_chart/node_modules
rm -rf ../resource.tar.gz
tar -czf ../resource.tar.gz *