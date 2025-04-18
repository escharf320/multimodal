import { useEffect, useState } from 'react'
import { QRCodeSVG } from 'qrcode.react'

const REFRESH_RATE = 25 // hz

interface StreamingQRCodeProps {
  uuid: string
}

export default function StreamingQRCode({ uuid }: StreamingQRCodeProps) {
  // Update a timestamp state every REFRESH_RATE
  const [timestamp, setTimestamp] = useState(Date.now())

  useEffect(() => {
    const interval = setInterval(() => {
      setTimestamp(Date.now())
    }, 1000 / REFRESH_RATE)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="my-2 text-center">
      <QRCodeSVG value={`${uuid}@${timestamp}`} size={290} />
    </div>
  )
}
