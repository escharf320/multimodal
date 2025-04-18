import { uIOhook } from 'uiohook-napi'

export function startLogger() {
  console.log('Starting logger');

  uIOhook.on('keydown', (event) => {
    console.log(event);
  })

  uIOhook.on('keyup', (event) => {
    console.log(event);
  })
  
  uIOhook.start();
}

export function stopLogger() {
  console.log('Stopping logger');
  uIOhook.stop();
}
