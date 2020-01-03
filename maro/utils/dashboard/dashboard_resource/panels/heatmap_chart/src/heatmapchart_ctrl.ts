// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.



import { MetricsPanelCtrl } from 'grafana/app/plugins/sdk';
import _ from 'lodash';
import kbn from 'grafana/app/core/utils/kbn';
// @ts-ignore
import TimeSeries from 'grafana/app/core/time_series';
import rendering from './rendering';

class HeatmapChartCtrl extends MetricsPanelCtrl {
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
      x_axis: 'from',
      xColumns: [],
      y_axis: 'to',
      yColumns: [],
      z_axis: 'heat',
      zColumns: [],

      links: [],
      datasource: null,
      maxDataPoints: 3,
      interval: null,
      targets: [{}],
      cacheTimeout: null,
      nullPointMode: 'connected',
      breakPoint: '10%',
      aliasColors: {},
      fontSize: '80%',
    };

    _.defaults(this.panel, panelDefaults);

    this.events.on('render', this.onRender.bind(this));
    this.events.on('data-received', this.onDataReceived.bind(this));
    this.events.on('data-error', this.onDataError.bind(this));
    this.events.on('data-snapshot-load', this.onDataReceived.bind(this));
    this.events.on('init-edit-mode', this.onInitEditMode.bind(this));
  }

  onInitEditMode() {
    this.addEditorTab('Options', 'public/plugins/grafana-heatmapchart-panel/editor.html', 2);
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
    if (series && series.length > 0) {
      const xProcessed = series.map(this.seriesHandler.bind(this));
      if (xProcessed) {
        for (let j = 0; j < xProcessed.length; j++) {
          const ddd = xProcessed[j];
          const calDict: any = {};
          if (ddd) {
            for (let i = 0; i < ddd.length; i++) {
              if (!calDict[ddd[i].x + '-' + ddd[i].y]) {
                calDict[ddd[i].x + '-' + ddd[i].y] = { x: ddd[i].x, y: ddd[i].y, z: ddd[i].z };
                seriesData.push(calDict[ddd[i].x + '-' + ddd[i].y]);
              } else {
                calDict[ddd[i].x + '-' + ddd[i].y].z += ddd[i].z;
              }
            }
          }
        }
      }
    }
    return seriesData;
  }

  onDataReceived(dataList: any) {
    const columnOptions: string[] = [];
    if (dataList) {
      for (let j = 0; j < dataList.length; j++) {
        for (let i = 0; i < dataList[j].columns.length; i++) {
          columnOptions.push(dataList[j].columns[i].text);
        }
      }
    }
    this.panel.xColumns = columnOptions.slice();
    this.panel.yColumns = columnOptions.slice();
    this.panel.zColumns = columnOptions.slice();

    this.series = dataList;
    this.data = this.parseSeries(this.series);
    this.render(this.data);
  }

  seriesHandler(seriesData: any) {
    const series: any[] = [];

    let xIndex = 0;
    let yIndex = 0;
    let zIndex = 0;
    for (let i = 0; i < seriesData.columns.length; i++) {
      if (seriesData.columns[i].text === this.panel.x_axis) {
        xIndex = i;
      }
      if (seriesData.columns[i].text === this.panel.y_axis) {
        yIndex = i;
      }
      if (seriesData.columns[i].text === this.panel.z_axis) {
        zIndex = i;
      }
    }
    for (let i = 0; i < seriesData.rows.length; i++) {
      series.push({
        x: seriesData.rows[i][xIndex],
        y: seriesData.rows[i][yIndex],
        z: seriesData.rows[i][zIndex],
      });
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
}

export { HeatmapChartCtrl, HeatmapChartCtrl as MetricsPanelCtrl };
