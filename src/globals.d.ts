declare global {
  interface Window {
    API: {
      getSessionUuid: () => string
      openKeyLoggerDirectory: () => void
    }
  }
}

export {}
