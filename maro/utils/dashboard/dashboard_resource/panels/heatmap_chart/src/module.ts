// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { HeatmapChartCtrl } from './heatmapchart_ctrl';
import { loadPluginCss } from 'grafana/app/plugins/sdk';

loadPluginCss({
  dark: 'plugins/grafana-heatmapchart-panel/styles/dark.css',
  light: 'plugins/grafana-heatmapchart-panel/styles/light.css',
});

export { HeatmapChartCtrl as PanelCtrl };
