import { useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import type { TrafficCurrent } from '../types'

export function useAutoStep(enabled: boolean, intervalMs = 1000) {
  const [current, setCurrent] = useState<TrafficCurrent | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchCurrent = async () => {
    try {
      const data = await api.trafficCurrent()
      if (data?.step !== undefined) setCurrent(data)
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    }
  }

  const step = async () => {
    try {
      await api.controllerStep()
      await fetchCurrent()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Step error')
    }
  }

  useEffect(() => {
    if (enabled) {
      timerRef.current = setInterval(step, intervalMs)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [enabled, intervalMs])

  return { current, error, fetchCurrent }
}

