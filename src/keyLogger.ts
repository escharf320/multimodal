import { uIOhook, UiohookKeyboardEvent, UiohookKey } from 'uiohook-napi'
import { createLogger, format, transports } from 'winston'

function startLogger(keyLoggerFile: string) {
  const fileTransport = new transports.File({ filename: keyLoggerFile })
  const consoleTransport = new transports.Console()

  const logMessageOnlyFormat = format.printf((info) => {
    return JSON.stringify(info.message)
  })

  const logger = createLogger({
    transports: [fileTransport, consoleTransport],
    format: logMessageOnlyFormat,
  })

  return logger
}

function extractKeyInfo(event: UiohookKeyboardEvent) {
  return {
    time: Date.now(), // overwrite time with current time
    type: event.type,
    shiftKey: event.shiftKey,
    keycode: event.keycode,
  }
}

export function startKeyLogger(keyLoggerFile: string) {
  const logger = startLogger(keyLoggerFile)

  uIOhook.on('keydown', (event) => {
    logger.info(extractKeyInfo(event))
  })

  uIOhook.on('keyup', (event) => {
    logger.info(extractKeyInfo(event))
  })

  uIOhook.start()
}

export function stopKeyLogger() {
  uIOhook.stop()
}
