// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { MetricsPanelCtrl } from 'grafana/app/plugins/sdk';
import _ from 'lodash';
import kbn from 'grafana/app/core/utils/kbn';
// @ts-ignore
import TimeSeries from 'grafana/app/core/time_series';
import rendering from './rendering';
import './legend';
import * as d3 from 'd3';

class DotChartCtrl extends MetricsPanelCtrl {
  static templateUrl = 'module.html';
  $rootScope: any;
  hiddenSeries: any;
  unitFormats: any;
  series: any;
  data: any;

  /** @ngInject */
  constructor($scope: any, $injector: any, $rootScope: any) {
    super($scope, $injector);
    this.$rootScope = $rootScope;
    this.hiddenSeries = {};

    const panelDefaults = {
      x_axis: 'ep',
      xColumns: [],
      ignoreColumn: '',
      ignoreColumns: [],
      aggregation: 'sum',
      aggregationOptions: ['sum', 'no', 'max', 'min', 'median', 'mean', 'variance', 'deviation'],
      legend: {
        show: true, // disable/enable legend
      },
      links: [],
      datasource: null,
      maxDataPoints: 3,
      interval: null,
      targets: [{}],
      cacheTimeout: null,
      nullPointMode: 'connected',
      legendType: 'Under graph',
      breakPoint: '10%',
      aliasColors: {},
      fontSize: '80%',
    };

    _.defaults(this.panel, panelDefaults);
    _.defaults(this.panel.legend, panelDefaults.legend);

    this.events.on('render', this.onRender.bind(this));
    this.events.on('data-received', this.onDataReceived.bind(this));
    this.events.on('data-error', this.onDataError.bind(this));
    this.events.on('data-snapshot-load', this.onDataReceived.bind(this));
    this.events.on('init-edit-mode', this.onInitEditMode.bind(this));

    this.setLegendWidthForLegacyBrowser();
  }

  onInitEditMode() {
    this.addEditorTab('Options', 'public/plugins/grafana-dotchart-panel/editor.html', 2);
    this.unitFormats = kbn.getUnitFormats();
  }

  setUnitFormat(subItem: any) {
    this.panel.format = subItem.value;
    this.render();
  }

  onDataError() {
    this.series = [];
    this.render();
  }

  changeSeriesColor(series: any, color: any) {
    series.color = color;
    this.panel.aliasColors[series.alias] = series.color;
    this.render();
  }

  onRender() {
    this.data = this.parseSeries(this.series);
  }

  parseSeries(series: any) {
    const seriesData = [];
    if (this.panel.xColumns && this.panel.xColumns.length > 0 && this.panel.xColumns && this.panel.xColumns.indexOf(this.panel.x_axis) >= 0) {
      const xProcessed = series.map(this.seriesHandler.bind(this));
      if (xProcessed) {
        for (let j = 0; j < xProcessed.length; j++) {
          const ddd = xProcessed[j];
          for (let i = 0; i < ddd.length; i++) {
            seriesData.push({
              label: ddd[i].label,
              data: ddd[i].data,
              color: this.panel.aliasColors[ddd[i].alias] || this.$rootScope.colors[i],
              legendData: ddd[i].data,
            });
          }
        }
      }
    }
    return seriesData;
  }

  onDataReceived(dataList: any) {
    const ignoreOptions: string[] = [];
    if (dataList) {
      for (let j = 0; j < dataList.length; j++) {
        for (let i = 0; i < dataList[j].columns.length; i++) {
          ignoreOptions.push(dataList[j].columns[i].text);
        }
      }
    }
    this.panel.xColumns = ignoreOptions.slice();
    this.panel.ignoreColumns = ignoreOptions.slice();

    this.series = dataList;
    this.data = this.parseSeries(this.series);
    this.render(this.data);
  }

  seriesHandler(seriesData: any) {
    const series = [];
    const yData: any[] = [];
    let xData: any[] = [];
    for (let i = 0; i < seriesData.columns.length; i++) {
      if (seriesData.columns[i].text !== 'Time' && seriesData.columns[i].text !== this.panel.x_axis) {
        yData.push(
          _.map(seriesData.rows, row => {
            return row[i];
          })
        );
      } else {
        yData.push([]);
        if (seriesData.columns[i].text === this.panel.x_axis) {
          xData = _.map(seriesData.rows, row => {
            return row[i];
          });
        }
      }
    }
    for (let i = 0; i < seriesData.columns.length; i++) {
      if (seriesData.columns[i].text !== 'Time' && seriesData.columns[i].text !== this.panel.x_axis) {
        const dataSeries = _.map(yData[i], (y, d: number) => {
          return { x: xData[d], y: y };
        });
        let allIsNull = true;
        for (let j = 0; j < dataSeries.length; j++) {
          if (dataSeries[j].y != null) {
            allIsNull = false;
            break;
          }
        }
        if (allIsNull) {
          continue;
        }
        const groupedSeries = d3
          .nest()
          .key((d: any) => {
            return d.x;
          })
          .sortKeys((a: any, b: any) => {
            const nA = Number(a);
            const nB = Number(b);
            if (isNaN(nA) || isNaN(nB)) {
              return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
            } else {
              return nA < nB ? -1 : nA > nB ? 1 : nA >= nB ? 0 : NaN;
            }
          })
          .entries(dataSeries);
        const groupedData: any[] = [];
        for (let j = 0; j < groupedSeries.length; j++) {
          let dataToPush = [];
          switch (this.panel.aggregation) {
            case 'no':
              dataToPush = _.map(groupedSeries[j].values, (d: any) => {
                return [d.x, d.y];
              })
              break;
            case 'sum':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.sum(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            case 'max':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.max(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            case 'min':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.min(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            case 'mean':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.mean(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            case 'median':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.median(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            case 'variance':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.variance(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            case 'deviation':
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.deviation(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
              break;
            default:
              dataToPush = [
                Number(groupedSeries[j].key),
                d3.sum(
                  _.map(groupedSeries[j].values, (d: any) => {
                    return d.y;
                  })
                ),
              ];
          }
          if (this.panel.aggregation === 'no') {
            groupedData.concat(dataToPush);
          }
          else {
            groupedData.push(dataToPush);
          }
        }

        series.push({
          label: seriesData.columns[i].text,
          data: groupedData,
        });
      }
    }
    return series;
  }

  getDecimalsForValue(value: any) {
    if (_.isNumber(this.panel.decimals)) {
      return { decimals: this.panel.decimals, scaledDecimals: null };
    }

    const delta = value / 2;
    let dec = -Math.floor(Math.log(delta) / Math.LN10);

    const magn = Math.pow(10, -dec);
    const norm = delta / magn; // norm is between 1.0 and 10.0
    let size;

    if (norm < 1.5) {
      size = 1;
    } else if (norm < 3) {
      size = 2;
      // special case for 2.5, requires an extra decimal
      if (norm > 2.25) {
        size = 2.5;
        ++dec;
      }
    } else if (norm < 7.5) {
      size = 5;
    } else {
      size = 10;
    }

    size *= magn;

    // reduce starting decimals if not needed
    if (Math.floor(value) === value) {
      dec = 0;
    }

    const result = {
      decimals: 0,
      scaledDecimals: 0,
    };
    result.decimals = Math.max(0, dec);
    result.scaledDecimals = result.decimals - Math.floor(Math.log(size) / Math.LN10) + 2;

    return result;
  }

  formatValue(value: any) {
    const decimalInfo = this.getDecimalsForValue(value);
    const formatFunc = kbn.valueFormats[this.panel.format];
    if (formatFunc) {
      return formatFunc(value, decimalInfo.decimals, decimalInfo.scaledDecimals);
    }
    return value;
  }

  link(scope: any, elem: any, attrs: any, ctrl: any) {
    rendering(scope, elem, attrs, ctrl);
  }

  toggleSeries(serie: any) {
    if (this.hiddenSeries[serie.label]) {
      delete this.hiddenSeries[serie.label];
    } else {
      this.hiddenSeries[serie.label] = true;
    }
    this.render();
  }

  onLegendTypeChanged() {
    this.setLegendWidthForLegacyBrowser();
    this.render();
  }

  setLegendWidthForLegacyBrowser() {
    // @ts-ignore
    const isIE11 = !!window.MSInputMethodContext && !!document.documentMode;
    if (isIE11 && this.panel.legendType === 'Right side' && !this.panel.legend.sideWidth) {
      this.panel.legend.sideWidth = 150;
    }
  }
}

export { DotChartCtrl, DotChartCtrl as MetricsPanelCtrl };
