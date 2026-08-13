import { useEffect, useState } from 'react'

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    fetch('http://127.0.0.1:8000/health')
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'healthy') {
          setBackendStatus('connected')
        } else {
          setBackendStatus('unavailable')
        }
      })
      .catch(() => setBackendStatus('unavailable'))
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white text-center px-4">
      <h1 className="text-4xl font-semibold text-gray-900">MediScribeAI</h1>
      <p className="mt-2 text-lg text-gray-600">
        Fog-Assisted AI Clinical Documentation System
      </p>
      <p className="mt-8 text-base font-medium text-gray-700">
        {backendStatus === 'checking' && 'Checking backend...'}
        {backendStatus === 'connected' && '🟢 Backend Connected'}
        {backendStatus === 'unavailable' && '🔴 Backend Unavailable'}
      </p>
    </div>
  )
}

export default App
