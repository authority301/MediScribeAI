import { useEffect, useRef, useState } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    fetch(`${API_BASE}/health`)
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

  const [token, setToken] = useState(null)
  const [loginEmail, setLoginEmail] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [loginError, setLoginError] = useState(null)

  const [consultationId, setConsultationId] = useState(null)
  const [consultationStatus, setConsultationStatus] = useState(null)
  const [consultationError, setConsultationError] = useState(null)

  const [isRecording, setIsRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState(null)
  const [micError, setMicError] = useState(null)
  const [uploadState, setUploadState] = useState('idle') // idle | uploading | success | error
  const [uploadMessage, setUploadMessage] = useState(null)

  // { id, filename, processingState, processingMessage, transcribeState, transcribeResult }
  const [audioRecords, setAudioRecords] = useState([])

  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])

  async function handleLogin(event) {
    event.preventDefault()
    setLoginError(null)
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: loginEmail, password: loginPassword }),
      })
      if (!res.ok) {
        setLoginError('Login failed. Check your email and password.')
        return
      }
      const data = await res.json()
      setToken(data.access_token)
    } catch {
      setLoginError('Could not reach the server.')
    }
  }

  async function handleCreateConsultation() {
    setConsultationError(null)
    try {
      const res = await fetch(`${API_BASE}/consultations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({}),
      })
      if (!res.ok) {
        setConsultationError('Could not create consultation.')
        return
      }
      const data = await res.json()
      setConsultationId(data.id)
      setConsultationStatus(data.status)
    } catch {
      setConsultationError('Could not reach the server.')
    }
  }

  async function handleActivateConsultation() {
    setConsultationError(null)
    try {
      const res = await fetch(`${API_BASE}/consultations/${consultationId}/status`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ status: 'active' }),
      })
      if (!res.ok) {
        setConsultationError('Could not activate consultation.')
        return
      }
      const data = await res.json()
      setConsultationStatus(data.status)
    } catch {
      setConsultationError('Could not reach the server.')
    }
  }

  async function handleStartRecording() {
    setMicError(null)
    setUploadState('idle')
    setUploadMessage(null)
    setRecordedBlob(null)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setRecordedBlob(blob)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch {
      setMicError('Microphone permission was denied or unavailable. Please allow microphone access and try again.')
    }
  }

  function handleStopRecording() {
    mediaRecorderRef.current?.stop()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    setIsRecording(false)
  }

  async function handleUploadRecording() {
    if (!recordedBlob) return
    setUploadState('uploading')
    setUploadMessage(null)

    const formData = new FormData()
    formData.append('file', recordedBlob, 'recording.webm')

    try {
      const res = await fetch(`${API_BASE}/consultations/${consultationId}/audio`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      })

      if (!res.ok) {
        setUploadState('error')
        setUploadMessage(`Upload failed (HTTP ${res.status}).`)
        return
      }

      const data = await res.json()
      setUploadState('success')
      setUploadMessage(`Uploaded "${data.original_filename}" (${data.file_size} bytes).`)
      setRecordedBlob(null)
      setAudioRecords((prev) => [
        ...prev,
        {
          id: data.id,
          filename: data.original_filename,
          processingState: 'idle',
          processingMessage: null,
          transcribeState: 'idle',
          transcribeResult: null,
        },
      ])
    } catch {
      setUploadState('error')
      setUploadMessage('Could not reach the server.')
    }
  }

  async function handleProcessAudio(audioId) {
    setAudioRecords((prev) =>
      prev.map((record) =>
        record.id === audioId
          ? { ...record, processingState: 'processing', processingMessage: null }
          : record,
      ),
    )

    try {
      const res = await fetch(`${API_BASE}/consultations/${consultationId}/audio/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ audio_id: audioId }),
      })

      if (!res.ok) {
        setAudioRecords((prev) =>
          prev.map((record) =>
            record.id === audioId
              ? { ...record, processingState: 'failed', processingMessage: `Processing failed (HTTP ${res.status}).` }
              : record,
          ),
        )
        return
      }

      const data = await res.json()
      setAudioRecords((prev) =>
        prev.map((record) =>
          record.id === audioId
            ? {
                ...record,
                processingState: 'completed',
                processingMessage: `${data.sample_rate} Hz, ${data.channels} channel(s), ${data.format}`,
              }
            : record,
        ),
      )
    } catch {
      setAudioRecords((prev) =>
        prev.map((record) =>
          record.id === audioId
            ? { ...record, processingState: 'failed', processingMessage: 'Could not reach the server.' }
            : record,
        ),
      )
    }
  }

  async function handleTranscribeAudio(audioId) {
    setAudioRecords((prev) =>
      prev.map((record) =>
        record.id === audioId
          ? { ...record, transcribeState: 'transcribing', transcribeResult: null }
          : record,
      ),
    )

    try {
      const res = await fetch(`${API_BASE}/consultations/${consultationId}/audio/transcribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ audio_id: audioId }),
      })

      if (!res.ok) {
        setAudioRecords((prev) =>
          prev.map((record) =>
            record.id === audioId ? { ...record, transcribeState: 'failed' } : record,
          ),
        )
        return
      }

      const data = await res.json()
      setAudioRecords((prev) =>
        prev.map((record) =>
          record.id === audioId
            ? {
                ...record,
                transcribeState: 'completed',
                transcribeResult: {
                  language: data.language,
                  text: data.text,
                  segmentCount: data.segment_count,
                },
              }
            : record,
        ),
      )
    } catch {
      setAudioRecords((prev) =>
        prev.map((record) =>
          record.id === audioId ? { ...record, transcribeState: 'failed' } : record,
        ),
      )
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white text-center px-4 py-8 gap-6">
      <div>
        <h1 className="text-4xl font-semibold text-gray-900">MediScribeAI</h1>
        <p className="mt-2 text-lg text-gray-600">
          Fog-Assisted AI Clinical Documentation System
        </p>
        <p className="mt-4 text-base font-medium text-gray-700">
          {backendStatus === 'checking' && 'Checking backend...'}
          {backendStatus === 'connected' && '🟢 Backend Connected'}
          {backendStatus === 'unavailable' && '🔴 Backend Unavailable'}
        </p>
      </div>

      {!token && (
        <form onSubmit={handleLogin} className="flex flex-col gap-2 w-64">
          <input
            type="email"
            placeholder="Email"
            value={loginEmail}
            onChange={(e) => setLoginEmail(e.target.value)}
            className="border rounded px-2 py-1"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={loginPassword}
            onChange={(e) => setLoginPassword(e.target.value)}
            className="border rounded px-2 py-1"
            required
          />
          <button type="submit" className="bg-gray-900 text-white rounded px-3 py-1">
            Login
          </button>
          {loginError && <p className="text-sm text-red-600">{loginError}</p>}
        </form>
      )}

      {token && !consultationId && (
        <div className="flex flex-col gap-2 items-center">
          <button
            onClick={handleCreateConsultation}
            className="bg-gray-900 text-white rounded px-3 py-1"
          >
            Create Consultation
          </button>
          {consultationError && <p className="text-sm text-red-600">{consultationError}</p>}
        </div>
      )}

      {token && consultationId && consultationStatus !== 'active' && (
        <div className="flex flex-col gap-2 items-center">
          <p className="text-sm text-gray-600">
            Consultation status: <span className="font-medium">{consultationStatus}</span>
          </p>
          <button
            onClick={handleActivateConsultation}
            className="bg-gray-900 text-white rounded px-3 py-1"
          >
            Activate Consultation
          </button>
          {consultationError && <p className="text-sm text-red-600">{consultationError}</p>}
        </div>
      )}

      {token && consultationId && consultationStatus === 'active' && (
        <div className="flex flex-col gap-3 items-center">
          <p className="text-sm text-gray-600">
            Consultation <span className="font-mono">{consultationId}</span> is active
          </p>

          {!isRecording ? (
            <button
              onClick={handleStartRecording}
              className="bg-gray-900 text-white rounded px-3 py-1"
            >
              Start Recording
            </button>
          ) : (
            <button
              onClick={handleStopRecording}
              className="bg-red-600 text-white rounded px-3 py-1"
            >
              Stop Recording
            </button>
          )}

          {isRecording && <p className="text-sm font-medium text-red-600">Recording...</p>}
          {micError && <p className="text-sm text-red-600">{micError}</p>}

          {recordedBlob && !isRecording && (
            <button
              onClick={handleUploadRecording}
              disabled={uploadState === 'uploading'}
              className="bg-gray-900 text-white rounded px-3 py-1 disabled:opacity-50"
            >
              {uploadState === 'uploading' ? 'Uploading...' : 'Upload Recording'}
            </button>
          )}

          {uploadState === 'success' && (
            <p className="text-sm text-green-600">✅ {uploadMessage}</p>
          )}
          {uploadState === 'error' && <p className="text-sm text-red-600">❌ {uploadMessage}</p>}

          {audioRecords.length > 0 && (
            <div className="flex flex-col gap-2 items-center mt-4 w-80">
              <p className="text-sm text-gray-600">Uploaded audio</p>
              {audioRecords.map((record) => (
                <div
                  key={record.id}
                  className="flex flex-col items-center gap-1 border rounded px-3 py-2 w-full"
                >
                  <p className="text-sm font-mono truncate w-full text-center">{record.filename}</p>
                  <button
                    onClick={() => handleProcessAudio(record.id)}
                    disabled={record.processingState === 'processing'}
                    className="bg-gray-900 text-white rounded px-3 py-1 disabled:opacity-50 text-sm"
                  >
                    {record.processingState === 'processing' ? 'Processing...' : 'Process Audio'}
                  </button>
                  {record.processingState === 'completed' && (
                    <p className="text-sm text-green-600">✅ Processing complete ({record.processingMessage})</p>
                  )}
                  {record.processingState === 'failed' && (
                    <p className="text-sm text-red-600">❌ Processing failed{record.processingMessage ? `: ${record.processingMessage}` : ''}</p>
                  )}

                  {record.processingState === 'completed' && (
                    <>
                      <button
                        onClick={() => handleTranscribeAudio(record.id)}
                        disabled={record.transcribeState === 'transcribing'}
                        className="bg-gray-900 text-white rounded px-3 py-1 disabled:opacity-50 text-sm"
                      >
                        {record.transcribeState === 'transcribing'
                          ? 'Transcribing...'
                          : 'Transcribe Audio'}
                      </button>
                      {record.transcribeState === 'completed' && record.transcribeResult && (
                        <div className="text-sm text-green-600 text-left w-full">
                          <p>✅ Transcription complete</p>
                          <p className="text-gray-700">
                            Language: <span className="font-mono">{record.transcribeResult.language}</span>
                          </p>
                          <p className="text-gray-700">
                            Segments: {record.transcribeResult.segmentCount}
                          </p>
                          <p className="text-gray-700 break-words">
                            "{record.transcribeResult.text}"
                          </p>
                        </div>
                      )}
                      {record.transcribeState === 'failed' && (
                        <p className="text-sm text-red-600">❌ Transcription failed</p>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
