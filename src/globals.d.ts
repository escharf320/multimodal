declare global {
  interface Window {
    API: {
      getSessionUuid: () => string
      openKeyLoggerFile: () => void
    }
  }
}

export {}
