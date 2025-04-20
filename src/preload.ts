/* eslint-disable @typescript-eslint/no-var-requires */

// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('API', {
  getSessionUuid: () => process.argv.slice(-1)[0], // gets the session uuid from the additional arguments
  openKeyLoggerFile: () => ipcRenderer.invoke('openKeyLoggerFile'),
})
