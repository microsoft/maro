// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.



import _ from 'lodash';
import $ from 'jquery';
import * as d3 from 'd3';

export default function link(scope: any, elem: any, attrs: any, ctrl: any) {
  let data;
  const panel = ctrl.panel;
  elem = elem.find('.heatmapchart-panel__chart');
  // const $tooltip = $('<div id="tooltip">') as any;

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

  function noDataPoints() {
    const html = '<div class="datapoints-warning"><span class="small">No data points</span></div>';
    elem.html(html);
  }

  function addHeatmapChart() {
    const width = elem.width();
    const height = ctrl.height;

    const size = Math.min(width, height);

    const plotCanvas = $('<div id="test"></div>');
    const plotCss = {
      margin: 'auto',
      position: 'relative',
      paddingBottom: 20 + 'px',
      height: size + 'px',
    };

    plotCanvas.css(plotCss);

    data = ctrl.data;

    elem.html(plotCanvas);

    const margin = { top: 20, right: 20, bottom: 60, left: 60 },
      chartWidth = width - margin.left - margin.right,
      chartHeight = height - margin.top - margin.bottom;
    const chartRoot = d3.select(plotCanvas.get(0));
    chartRoot.selectAll('*').remove();
    const gRoot = chartRoot
      .append('svg')
      .attr('viewBox', '0,0,' + width + ',' + height)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const xArray = d3
      .nest()
      .key((d: any) => {
        return d.x;
      })
      .entries(data)
      .map((d: any) => {
        return d.key;
      })
      .sort();
    const yArray = d3
      .nest()
      .key((d: any) => {
        return d.y;
      })
      .entries(data)
      .map((d: any) => {
        return d.key;
      })
      .sort();

    let zMin = 0;
    let zMax = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i].z > zMax) {
        zMax = data[i].z;
      }
      if (data[i].z < zMin) {
        zMin = data[i].z;
      }
    }
    const z = d3
      .scaleLinear()
      .domain([zMin, zMax])
      .range([200, 1000]);

    const drawSeries = [];
    for (let i = 0; i < data.length; i++) {
      drawSeries.push({
        x: xArray.indexOf(data[i].x),
        y: yArray.indexOf(data[i].y),
        z: data[i].z,
      });
    }
    let rectWidth = chartWidth / xArray.length - 1;
    let rectHeight = chartHeight / yArray.length - 1;
    if (rectWidth <= 0) {
      rectWidth = 1;
    }
    if (rectHeight <= 0) {
      rectHeight = 1;
    }
    gRoot
      .selectAll('.heatRect')
      .data(drawSeries)
      .enter()
      .append('g')
      .attr('class', 'heatRect')
      .attr('transform', d => {
        return 'translate(' + (rectWidth + 2) * d.x + ',' + (rectHeight + 2) * d.y + ')';
      })
      .append('rect')
      .attr('width', rectWidth)
      .attr('height', rectHeight)
      .style('fill', 'red')
      .style('fill-opacity', d => {
        return z(d.z) / 1000;
      })
      .append('title')
      .text(d => {
        return ctrl.panel.x_axis + ' : ' + xArray[d.x] + '\n' + ctrl.panel.y_axis + ' : ' + yArray[d.y] + '\n' + ctrl.panel.z_axis + ' : ' + d.z;
      });
    gRoot
      .selectAll('.heatX')
      .data(xArray)
      .enter()
      .append('g')
      .attr('class', 'heatX')
      .attr('transform', (d, i) => {
        return 'translate(' + (rectWidth + 2) * (i + 0.5) + ',' + (chartHeight + 20) + ')';
      })
      .append('text')
      .attr('class', 'gf-form-label')
      .attr('fill', d3.rgb(125, 125, 125).toString())
      .attr('transform', () => {
        return 'rotate(-45 ' + 0 + ',' + 0 + ')';
      })
      .attr('text-anchor', 'end')
      .text(d => {
        return d.length > 10 ? d.substring(0, 7) + '...' : d;
      })
      .append('title')
      .text(d => {
        return d;
      });
    gRoot
      .selectAll('.heatY')
      .data(yArray)
      .enter()
      .append('g')
      .attr('class', 'heatY')
      .attr('transform', (d, i) => {
        return 'translate(' + 0 + ',' + (rectHeight + 2) * i + ')';
      })
      .append('text')
      .attr('class', 'gf-form-label')
      .attr('text-anchor', 'end')
      .attr('fill', d3.rgb(125, 125, 125).toString())
      .attr('transform', () => {
        return 'rotate(-45 20,' + rectHeight / 2 + ')';
      })
      .text(d => {
        return d.length > 10 ? d.substring(0, 7) + '...' : d;
      })
      .append('title')
      .text(d => {
        return d;
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
      addHeatmapChart();
    }

    if (incrementRenderCounter) {
      ctrl.renderingCompleted();
    }
  }
}
