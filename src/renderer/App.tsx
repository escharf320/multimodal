import StreamingQRCode from './StreamingQRCode'

export default function App() {
  const sessionUuid = window.API.getSessionUuid()

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <StreamingQRCode uuid={sessionUuid} />
      <button
        className="mt-4 px-4 py-2 bg-gray-100 text-black rounded-lg shadow-sm border border-gray-300 hover:bg-gray-200 transition-colors font-medium text-sm cursor-pointer"
        onClick={() => {
          console.log('Opening key logger directory')
          window.API.openKeyLoggerDirectory()
        }}
      >
        Open KeyLogger Directory
      </button>
    </div>
  )
}
