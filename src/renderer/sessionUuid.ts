function toHexString(byteArray: Uint8Array) {
  return Array.from(byteArray, function (byte) {
    return ('0' + (byte & 0xff).toString(16)).slice(-2)
  }).join('')
}

export function generateSessionUuid() {
  return toHexString(crypto.getRandomValues(new Uint8Array(4)))
}
