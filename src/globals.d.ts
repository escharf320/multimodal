declare global {
  interface Window {
    API: {
      getSessionUuid: () => string
    }
  }
}

export {}
