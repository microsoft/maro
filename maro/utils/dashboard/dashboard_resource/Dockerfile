FROM node:10 as ext_build

WORKDIR /panels

COPY ./panels/line_chart ./line_chart
COPY ./panels/heatmap_chart ./heatmap_chart
COPY ./panels/stack_chart ./stack_chart

RUN cd /panels/line_chart; yarn install --pure-lockfile; yarn run dev; yarn run build; rm -rf node_modules
RUN cd /panels/heatmap_chart; yarn install --pure-lockfile; yarn run dev; yarn run build; rm -rf node_modules
RUN cd /panels/stack_chart; yarn install --pure-lockfile; yarn run dev; yarn run build;  rm -rf node_modules

FROM grafana/grafana:6.5.2

USER grafana
WORKDIR $GF_PATHS_HOME

COPY --from=ext_build /panels/line_chart/dist /var/lib/grafana/plugins/grafana-linechart-panel/dist 
COPY --from=ext_build /panels/heatmap_chart/dist /var/lib/grafana/plugins/grafana-heatmapchart-panel/dist
COPY --from=ext_build /panels/stack_chart/dist /var/lib/grafana/plugins/grafana-stackchart-panel/dist 

COPY ./provisioning /etc/grafana/provisioning
COPY ./templates /var/lib/grafana/dashboards


ENV GF_INSTALL_PLUGINS grafana-piechart-panel,grafana-influxdb-flux-datasource,natel-plotly-panel,briangann-datatable-panel
ENV GF_PATHS_DATA /var/lib/grafana/db

ENTRYPOINT [ "/run.sh" ]