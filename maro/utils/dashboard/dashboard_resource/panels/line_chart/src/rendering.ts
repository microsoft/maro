// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.



import _ from 'lodash';
import $ from 'jquery';

export default function link(scope: any, elem: any, attrs: any, ctrl: any) {
  let data;
  const panel = ctrl.panel;
  elem = elem.find('.linechart-panel__chart');
  const $tooltip = $('<div id="tooltip">') as any;

  ctrl.events.on('render', () => {
    if (panel.legendType === 'Right side') {
      render(false);
      setTimeout(() => {
        render(true);
      }, 50);
    } else {
      render(true);
    }
  });

  function getLegendHeight(panelHeight: any) {
    if (!ctrl.panel.legend.show || ctrl.panel.legendType === 'Right side' || ctrl.panel.legendType === 'On graph') {
      return 20;
    }

    if (ctrl.panel.legendType === 'Under graph') {
      const breakPoint = parseInt(ctrl.panel.breakPoint, 10) / 100;
      const total = 23 + 20 * data.length;
      return Math.min(total, Math.floor(panelHeight * breakPoint));
    }

    return 0;
  }

  function noDataPoints() {
    const html = '<div class="datapoints-warning"><span class="small">No data points</span></div>';
    elem.html(html);
  }

  function addLineChart() {
    const width = elem.width();
    const height = ctrl.height - getLegendHeight(ctrl.height);

    const size = Math.min(width, height);

    const plotCanvas = $('<div></div>');
    const plotCss = {
      margin: 'auto',
      position: 'relative',
      paddingBottom: 20 + 'px',
      height: size + 'px',
    };

    plotCanvas.css(plotCss);

    const options = {
      series: {
        lines: {
          steps: 0,
          show: true,
          lineWidth: 1,
          fill: false,
          // eg: rgba(255, 255, 255, 0.8)
          fillColor: null,
        },
        // 0 = no shadow
        shadowSize: 0,
        // mouse over color
        highlightColor: 1,
      },
      grid: {
        hoverable: true,
        clickable: false,
      },
      legend: {
        show: false,
        backgroundOpacity: 0.5,
        noColumns: 0,
        backgroundColor: 'green',
        position: 'ne',
      },
      xaxes: [{ position: 'bottom' }],
      yaxes: [{ position: 'left' }],
    };

    data = [];

    for (let i = 0; i < ctrl.data.length; i++) {
      const series = ctrl.data[i];

      // if hidden remove points
      if (!(ctrl.hiddenSeries[series.label] || ctrl.panel.ignoreColumn.indexOf(series.label) >= 0)) {
        data.push(series);
      }
    }

    if (panel.legend.sort) {
      if (ctrl.panel.valueName !== panel.legend.sort) {
        panel.legend.sort = ctrl.panel.valueName;
      }
      if (panel.legend.sortDesc === true) {
        data.sort((a: any, b: any) => {
          return b.legendData - a.legendData;
        });
      } else {
        data.sort((a: any, b: any) => {
          return a.legendData - b.legendData;
        });
      }
    }

    elem.html(plotCanvas);

    // @ts-ignore
    $.plot(plotCanvas, data, options);
    plotCanvas.bind('plothover', (event: any, pos: any, item: any) => {
      if (!item) {
        $tooltip.detach();
        return;
      }

      let body;
      const formatted = ctrl.formatValue(item.datapoint[1]);

      body = '<div class="linechart-tooltip-small"><div class="linechart-tooltip-time">';
      body += '<div class="linechart-tooltip-value">' + _.escape(item.series.label) + ': ' + formatted;
      body += '</div>';
      body += '</div></div>';

      $tooltip.html(body).place_tt(pos.pageX + 20, pos.pageY);
    });
  }

  function render(incrementRenderCounter: any) {
    if (!ctrl.data) {
      return;
    }

    data = ctrl.data;

    if (0 === ctrl.data.length) {
      noDataPoints();
    } else {
      addLineChart();
    }

    if (incrementRenderCounter) {
      ctrl.renderingCompleted();
    }
  }
}
