import { useState } from 'react'
import StreamingQRCode from './StreamingQRCode'
import { generateSessionUuid } from './sessionUuid'

export default function App() {
  const [sessionUuid, setSessionUuid] = useState(generateSessionUuid())

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <StreamingQRCode uuid={sessionUuid} />
    </div>
  )
}
