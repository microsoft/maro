// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { DotChartCtrl } from './dotchart_ctrl';
import { loadPluginCss } from 'grafana/app/plugins/sdk';

loadPluginCss({
  dark: 'plugins/grafana-dotchart-panel/styles/dark.css',
  light: 'plugins/grafana-dotchart-panel/styles/light.css',
});

export { DotChartCtrl as PanelCtrl };
