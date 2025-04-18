import crypto from 'crypto'

export function generateSessionUuid() {
  return crypto.randomBytes(4).toString('hex')
}
