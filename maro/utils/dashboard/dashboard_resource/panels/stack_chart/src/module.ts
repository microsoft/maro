// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { StackChartCtrl } from './stackchart_ctrl';
import { loadPluginCss } from 'grafana/app/plugins/sdk';

loadPluginCss({
  dark: 'plugins/grafana-stackchart-panel/styles/dark.css',
  light: 'plugins/grafana-stackchart-panel/styles/light.css',
});

export { StackChartCtrl as PanelCtrl };
