import StreamingQRCode from './StreamingQRCode'

export default function App() {
  const sessionUuid = window.API.getSessionUuid()

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <StreamingQRCode uuid={sessionUuid} />
    </div>
  )
}
