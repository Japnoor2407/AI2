import { useEffect, useRef, useState } from 'react'
import type { TrafficCurrent } from '../types'
import { SignalBadge } from './SignalBadge'
import {
  Video, Eye, EyeOff, Radio, ShieldCheck, Zap, Grid,
  Maximize, Sun, Moon, Gauge, Car, Bus as BusIcon, Truck as TruckIcon,
} from 'lucide-react'

interface TrafficVideoFeedProps {
  data: TrafficCurrent | null
}

interface Vehicle {
  id: number
  type: 'car' | 'motorbike' | 'auto' | 'bus' | 'truck'
  lane: number
  xOffset: number // slight lane variance for natural positioning
  y: number
  speed: number
  targetSpeed: number
  color: string
  roofColor: string
  confidence: number
  crossedTripwire: boolean
  isBraking: boolean
}

export function TrafficVideoFeed({ data }: TrafficVideoFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [selectedApproach, setSelectedApproach] = useState<string>('North_Approach')
  const [viewMode, setViewMode] = useState<'single' | 'quad'>('single')
  const [cctvMode, setCctvMode] = useState<'color' | 'ir'>('color')
  const [showYoloBoxes, setShowYoloBoxes] = useState<boolean>(true)
  const [showTripwire, setShowTripwire] = useState<boolean>(true)
  const [showSpeedHud, setShowSpeedHud] = useState<boolean>(true)
  const [tripwireFlash, setTripwireFlash] = useState<number>(0)
  const [countsByType, setCountsByType] = useState({ car: 18, motorbike: 42, auto: 11, bus: 4, truck: 2 })

  // Camera info mapping
  const cameraInfo: Record<string, { label: string; code: string; road: string; direction: string }> = {
    North_Approach: { label: 'CAM 01 - North Approach', code: 'A2-N', road: 'Station Road Arterial', direction: 'Northbound' },
    South_Approach: { label: 'CAM 02 - South Approach', code: 'A2-S', road: 'Camp Corridor', direction: 'Southbound' },
    East_Approach:  { label: 'CAM 03 - East Approach',  code: 'A3-E', road: 'Sassoon Arterial', direction: 'Eastbound' },
    West_Approach:  { label: 'CAM 04 - West Approach',  code: 'A3-W', road: 'Collectorate Feeder', direction: 'Westbound' },
  }

  const currentApproach = data?.approaches?.[selectedApproach]
  const signal = currentApproach?.signal || 'RED'
  const timeStr = data?.timestamp || '09:01:20'

  // Internal vehicle simulation state on canvas
  const vehiclesRef = useRef<Vehicle[]>([])
  const nextIdRef = useRef<number>(201)
  const lastSpawnRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationFrameId: number

    // ─── Procedural Vehicle Drawing Helpers ───
    const drawVehicle = (v: Vehicle, vx: number, vy: number, isIR: boolean) => {
      ctx.save()
      ctx.translate(vx, vy)

      const isBraking = v.isBraking
      const shadowColor = 'rgba(0, 0, 0, 0.45)'

      // ── 1. CAR ──
      if (v.type === 'car') {
        const cw = 28
        const cl = 46

        // Drop shadow
        ctx.fillStyle = shadowColor
        ctx.fillRect(-cw / 2 + 3, -cl / 2 + 4, cw, cl)

        // Tires
        ctx.fillStyle = isIR ? '#475569' : '#0f172a'
        ctx.fillRect(-cw / 2 - 2, -cl / 2 + 8, 3, 9)
        ctx.fillRect(cw / 2 - 1, -cl / 2 + 8, 3, 9)
        ctx.fillRect(-cw / 2 - 2, cl / 2 - 17, 3, 9)
        ctx.fillRect(cw / 2 - 1, cl / 2 - 17, 3, 9)

        // Car Body (Aerodynamic chassis)
        ctx.fillStyle = isIR ? '#cbd5e1' : v.color
        ctx.beginPath()
        ctx.roundRect(-cw / 2, -cl / 2, cw, cl, 6)
        ctx.fill()

        // Front Windshield
        ctx.fillStyle = isIR ? '#1e293b' : '#090d16'
        ctx.beginPath()
        ctx.roundRect(-cw / 2 + 3, -cl / 2 + 13, cw - 6, 9, 2)
        ctx.fill()

        // Roof
        ctx.fillStyle = isIR ? '#94a3b8' : v.roofColor
        ctx.fillRect(-cw / 2 + 4, -cl / 2 + 22, cw - 8, 12)

        // Rear Windshield
        ctx.fillStyle = isIR ? '#1e293b' : '#090d16'
        ctx.beginPath()
        ctx.roundRect(-cw / 2 + 3, cl / 2 - 11, cw - 6, 7, 2)
        ctx.fill()

        // Headlights (glowing yellow/white beams)
        ctx.fillStyle = '#fef08a'
        ctx.fillRect(-cw / 2 + 3, cl / 2 - 2, 6, 3)
        ctx.fillRect(cw / 2 - 9, cl / 2 - 2, 6, 3)

        // Headlight beam casting forward
        const beamGrad = ctx.createLinearGradient(0, cl / 2, 0, cl / 2 + 40)
        beamGrad.addColorStop(0, 'rgba(254, 240, 138, 0.25)')
        beamGrad.addColorStop(1, 'rgba(254, 240, 138, 0)')
        ctx.fillStyle = beamGrad
        ctx.beginPath()
        ctx.moveTo(-cw / 2 + 2, cl / 2)
        ctx.lineTo(-cw / 2 - 8, cl / 2 + 40)
        ctx.lineTo(cw / 2 + 8, cl / 2 + 40)
        ctx.lineTo(cw / 2 - 2, cl / 2)
        ctx.fill()

        // Taillights
        ctx.fillStyle = isBraking ? '#ff0000' : '#991b1b'
        ctx.fillRect(-cw / 2 + 3, -cl / 2, 6, 3)
        ctx.fillRect(cw / 2 - 9, -cl / 2, 6, 3)
        if (isBraking) {
          ctx.shadowColor = '#ff0000'
          ctx.shadowBlur = 10
          ctx.fillRect(-cw / 2 + 3, -cl / 2, 6, 3)
          ctx.fillRect(cw / 2 - 9, -cl / 2, 6, 3)
          ctx.shadowBlur = 0
        }
      }

      // ── 2. AUTO-RICKSHAW (Pune Tuk-Tuk) ──
      else if (v.type === 'auto') {
        const aw = 22
        const al = 34

        // Shadow
        ctx.fillStyle = shadowColor
        ctx.fillRect(-aw / 2 + 2, -al / 2 + 3, aw, al)

        // 3 Wheels (1 front, 2 rear)
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-2, al / 2 - 3, 4, 6)
        ctx.fillRect(-aw / 2 - 2, -al / 2 + 4, 3, 8)
        ctx.fillRect(aw / 2 - 1, -al / 2 + 4, 3, 8)

        // Body (Classic Pune Yellow/Black hood + Green body)
        ctx.fillStyle = isIR ? '#94a3b8' : '#15803d'
        ctx.beginPath()
        ctx.roundRect(-aw / 2, -al / 2, aw, al - 4, 4)
        ctx.fill()

        // Yellow canopy roof
        ctx.fillStyle = isIR ? '#cbd5e1' : '#eab308'
        ctx.beginPath()
        ctx.roundRect(-aw / 2 + 1, -al / 2 + 6, aw - 2, 18, 3)
        ctx.fill()

        // Driver area & windshield
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-aw / 2 + 3, al / 2 - 10, aw - 6, 4)

        // Single front headlight
        ctx.fillStyle = '#fef08a'
        ctx.beginPath()
        ctx.arc(0, al / 2, 3, 0, Math.PI * 2)
        ctx.fill()
      }

      // ── 3. MOTORBIKE / SCOOTER ──
      else if (v.type === 'motorbike') {
        // Shadow
        ctx.fillStyle = shadowColor
        ctx.fillRect(-4, -12, 8, 28)

        // Front & Rear tires
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-1.5, 9, 3, 7)
        ctx.fillRect(-1.5, -15, 3, 7)

        // Chassis body
        ctx.fillStyle = isIR ? '#cbd5e1' : v.color
        ctx.fillRect(-2.5, -8, 5, 18)

        // Handlebars
        ctx.fillStyle = '#64748b'
        ctx.fillRect(-8, 8, 16, 2)

        // Rider body
        ctx.fillStyle = isIR ? '#475569' : '#0369a1'
        ctx.fillRect(-5, -3, 10, 8)

        // Rider helmet
        ctx.fillStyle = isIR ? '#f8fafc' : v.roofColor
        ctx.beginPath()
        ctx.arc(0, 1, 4.5, 0, Math.PI * 2)
        ctx.fill()

        // Headlight
        ctx.fillStyle = '#fef08a'
        ctx.fillRect(-1.5, 14, 3, 2)
      }

      // ── 4. PUNE CITY BUS (PMPML Transit Bus) ──
      else if (v.type === 'bus') {
        const bw = 34
        const bl = 82

        // Shadow
        ctx.fillStyle = shadowColor
        ctx.fillRect(-bw / 2 + 4, -bl / 2 + 4, bw, bl)

        // Dual rear axles & front tires
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-bw / 2 - 2, bl / 2 - 22, 3, 10)
        ctx.fillRect(bw / 2 - 1, bl / 2 - 22, 3, 10)
        ctx.fillRect(-bw / 2 - 2, -bl / 2 + 12, 3, 10)
        ctx.fillRect(bw / 2 - 1, -bl / 2 + 12, 3, 10)

        // Main Bus Body
        ctx.fillStyle = isIR ? '#94a3b8' : '#b91c1c'
        ctx.beginPath()
        ctx.roundRect(-bw / 2, -bl / 2, bw, bl, 4)
        ctx.fill()

        // White/Silver Roof with AC vents
        ctx.fillStyle = isIR ? '#cbd5e1' : '#e2e8f0'
        ctx.fillRect(-bw / 2 + 3, -bl / 2 + 12, bw - 6, bl - 24)

        // AC Unit on roof
        ctx.fillStyle = isIR ? '#64748b' : '#94a3b8'
        ctx.fillRect(-bw / 2 + 7, -8, bw - 14, 18)

        // LED Destination Board (Pune Station)
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-bw / 2 + 4, bl / 2 - 8, bw - 8, 5)
        ctx.fillStyle = '#f59e0b'
        ctx.font = '5px monospace'
        ctx.fillText('PMPML PUNE', -bw / 2 + 5, bl / 2 - 4)

        // Front windshield
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-bw / 2 + 3, bl / 2 - 15, bw - 6, 6)

        // Dual headlights
        ctx.fillStyle = '#fef08a'
        ctx.fillRect(-bw / 2 + 3, bl / 2 - 2, 7, 3)
        ctx.fillRect(bw / 2 - 10, bl / 2 - 2, 7, 3)

        // Taillights
        ctx.fillStyle = isBraking ? '#ff0000' : '#7f1d1d'
        ctx.fillRect(-bw / 2 + 3, -bl / 2, 7, 3)
        ctx.fillRect(bw / 2 - 10, -bl / 2, 7, 3)
      }

      // ── 5. HEAVY COMMERCIAL TRUCK ──
      else if (v.type === 'truck') {
        const tw = 32
        const tl = 76

        // Shadow
        ctx.fillStyle = shadowColor
        ctx.fillRect(-tw / 2 + 4, -tl / 2 + 4, tw, tl)

        // Rear container cargo bed
        ctx.fillStyle = isIR ? '#64748b' : v.color
        ctx.beginPath()
        ctx.roundRect(-tw / 2, -tl / 2, tw, 48, 2)
        ctx.fill()
        // Container rib lines
        ctx.strokeStyle = 'rgba(0,0,0,0.3)'
        ctx.lineWidth = 1
        for (let r = -tl / 2 + 6; r < -tl / 2 + 45; r += 7) {
          ctx.beginPath()
          ctx.moveTo(-tw / 2, r)
          ctx.lineTo(tw / 2, r)
          ctx.stroke()
        }

        // Front Driver Cab
        ctx.fillStyle = isIR ? '#cbd5e1' : '#f59e0b'
        ctx.beginPath()
        ctx.roundRect(-tw / 2 + 1, tl / 2 - 24, tw - 2, 22, 3)
        ctx.fill()

        // Windshield
        ctx.fillStyle = '#0f172a'
        ctx.fillRect(-tw / 2 + 3, tl / 2 - 16, tw - 6, 7)

        // Headlights
        ctx.fillStyle = '#fef08a'
        ctx.fillRect(-tw / 2 + 3, tl / 2 - 2, 6, 3)
        ctx.fillRect(tw / 2 - 9, tl / 2 - 2, 6, 3)
      }

      // ─── High-Tech YOLOv8 HUD Bounding Box ───
      if (showYoloBoxes) {
        const boxW = v.type === 'bus' ? 42 : v.type === 'truck' ? 38 : v.type === 'car' ? 36 : v.type === 'auto' ? 28 : 22
        const boxH = v.type === 'bus' ? 88 : v.type === 'truck' ? 82 : v.type === 'car' ? 52 : v.type === 'auto' ? 40 : 34

        // Box border
        ctx.strokeStyle = '#22c55e'
        ctx.lineWidth = 1.2
        ctx.strokeRect(-boxW / 2, -boxH / 2, boxW, boxH)

        // High-tech corner reticles
        const cLen = 6
        ctx.strokeStyle = '#4ade80'
        ctx.lineWidth = 2.5
        // Top-left
        ctx.beginPath(); ctx.moveTo(-boxW / 2, -boxH / 2 + cLen); ctx.lineTo(-boxW / 2, -boxH / 2); ctx.lineTo(-boxW / 2 + cLen, -boxH / 2); ctx.stroke()
        // Top-right
        ctx.beginPath(); ctx.moveTo(boxW / 2 - cLen, -boxH / 2); ctx.lineTo(boxW / 2, -boxH / 2); ctx.lineTo(boxW / 2, -boxH / 2 + cLen); ctx.stroke()
        // Bottom-left
        ctx.beginPath(); ctx.moveTo(-boxW / 2, boxH / 2 - cLen); ctx.lineTo(-boxW / 2, boxH / 2); ctx.lineTo(-boxW / 2 + cLen, boxH / 2); ctx.stroke()
        // Bottom-right
        ctx.beginPath(); ctx.moveTo(boxW / 2 - cLen, boxH / 2); ctx.lineTo(boxW / 2, boxH / 2); ctx.lineTo(boxW / 2, boxH / 2 - cLen); ctx.stroke()

        // Label Tag
        ctx.fillStyle = 'rgba(10, 15, 30, 0.9)'
        ctx.fillRect(-boxW / 2, -boxH / 2 - 14, boxW + 28, 13)
        ctx.fillStyle = '#4ade80'
        ctx.font = 'bold 9px monospace'
        ctx.fillText(`${v.type.toUpperCase()} ${v.confidence}`, -boxW / 2 + 2, -boxH / 2 - 4)

        // Speed readout
        if (showSpeedHud) {
          const kmh = Math.round(v.speed * 18)
          ctx.fillStyle = v.isBraking ? '#f87171' : '#38bdf8'
          ctx.font = '8px monospace'
          ctx.fillText(`${kmh} km/h`, -boxW / 2 + 2, boxH / 2 + 9)
        }
      }

      ctx.restore()
    }

    // ─── Main Canvas Render Loop ───
    const render = (timestamp: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const w = canvas.width
      const h = canvas.height
      const isIR = cctvMode === 'ir'

      // ─── 1. Road Asphalt & Pavement ───
      // Sidewalk curbs
      const roadLeft = 70
      const roadRight = w - 70
      const roadWidth = roadRight - roadLeft

      // Pavement background (sidewalk tiles)
      ctx.fillStyle = isIR ? '#1e293b' : '#0f172a'
      ctx.fillRect(0, 0, w, h)

      // Sidewalk paving pattern
      ctx.strokeStyle = isIR ? '#334155' : '#1e293b'
      ctx.lineWidth = 1
      for (let y = 0; y < h; y += 25) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(roadLeft, y); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(roadRight, y); ctx.lineTo(w, y); ctx.stroke()
      }

      // Asphalt Road Surface
      ctx.fillStyle = isIR ? '#0f172a' : '#161d2b'
      ctx.fillRect(roadLeft, 0, roadWidth, h)

      // Concrete Curb stone lines
      ctx.strokeStyle = isIR ? '#64748b' : '#cbd5e1'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(roadLeft, 0); ctx.lineTo(roadLeft, h)
      ctx.moveTo(roadRight, 0); ctx.lineTo(roadRight, h)
      ctx.stroke()

      // 3 Lanes marking
      const laneWidth = roadWidth / 3
      ctx.strokeStyle = isIR ? '#475569' : '#64748b'
      ctx.lineWidth = 2
      ctx.setLineDash([20, 16])
      for (let l = 1; l < 3; l++) {
        const lx = roadLeft + l * laneWidth
        ctx.beginPath()
        ctx.moveTo(lx, 0)
        ctx.lineTo(lx, h)
        ctx.stroke()
      }
      ctx.setLineDash([])

      // Lane Arrows (painted on road)
      ctx.fillStyle = isIR ? '#475569' : '#334155'
      ctx.font = 'bold 16px sans-serif'
      ctx.fillText('↰', roadLeft + laneWidth * 0.45, h - 280) // Lane 1: Left turn
      ctx.fillText('↑', roadLeft + laneWidth * 1.45, h - 280) // Lane 2: Straight
      ctx.fillText('↱', roadLeft + laneWidth * 2.45, h - 280) // Lane 3: Right turn

      // Stop Line at intersection entrance
      const stopLineY = h - 130
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 8
      ctx.beginPath()
      ctx.moveTo(roadLeft, stopLineY)
      ctx.lineTo(roadRight, stopLineY)
      ctx.stroke()

      // "STOP" painted on asphalt
      ctx.fillStyle = 'rgba(255, 255, 255, 0.45)'
      ctx.font = 'bold 22px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('S T O P', w / 2, stopLineY - 22)
      ctx.textAlign = 'start'

      // Pedestrian Zebra Crossing
      ctx.fillStyle = isIR ? 'rgba(255, 255, 255, 0.3)' : 'rgba(255, 255, 255, 0.55)'
      for (let z = roadLeft + 8; z < roadRight - 10; z += 30) {
        ctx.fillRect(z, stopLineY + 12, 18, 48)
      }

      // Intersection Crossing Area (beyond stop line)
      ctx.fillStyle = isIR ? '#090d16' : '#111827'
      ctx.fillRect(roadLeft, stopLineY + 70, roadWidth, h - (stopLineY + 70))

      // Yellow Cross-Hatch / Box Junction Lines in intersection
      ctx.strokeStyle = isIR ? 'rgba(255, 255, 255, 0.15)' : 'rgba(234, 179, 8, 0.25)'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let bx = roadLeft; bx < roadRight; bx += 40) {
        ctx.moveTo(bx, stopLineY + 70)
        ctx.lineTo(bx + 40, h)
      }
      ctx.stroke()

      // Virtual Tripwire Counting Boundary
      const tripwireY = h - 210
      if (showTripwire) {
        ctx.strokeStyle = tripwireFlash > 0 ? '#38bdf8' : 'rgba(56, 189, 248, 0.7)'
        ctx.lineWidth = tripwireFlash > 0 ? 3.5 : 2
        ctx.setLineDash([8, 6])
        ctx.beginPath()
        ctx.moveTo(roadLeft, tripwireY)
        ctx.lineTo(roadRight, tripwireY)
        ctx.stroke()
        ctx.setLineDash([])

        // Glow when vehicle crosses
        if (tripwireFlash > 0) {
          ctx.shadowColor = '#38bdf8'
          ctx.shadowBlur = 12
          ctx.stroke()
          ctx.shadowBlur = 0
        }

        ctx.fillStyle = '#38bdf8'
        ctx.font = 'bold 9px monospace'
        ctx.fillText('⚡ YOLOv8 VIRTUAL DETECTION TRIPWIRE', roadLeft + 10, tripwireY - 7)
      }

      // ─── 2. Spawning Vehicles from Real Dataset Rates ───
      const rate = currentApproach?.arrival_rate_pcu_per_min || 18
      const spawnInterval = Math.max(650, 60000 / (rate * 3.2 + 1))
      if (timestamp - lastSpawnRef.current > spawnInterval) {
        lastSpawnRef.current = timestamp
        const lane = Math.floor(Math.random() * 3)
        const laneBlocked = vehiclesRef.current.some(v => v.lane === lane && v.y < 85)

        if (!laneBlocked) {
          const types: Vehicle['type'][] = ['motorbike', 'motorbike', 'auto', 'car', 'car', 'car', 'bus', 'truck']
          const type = types[Math.floor(Math.random() * types.length)]

          const colors = {
            car: ['#38bdf8', '#ef4444', '#f1f5f9', '#a855f7', '#10b981', '#f59e0b'],
            roofs: ['#0284c7', '#b91c1c', '#cbd5e1', '#7e22ce', '#047857', '#d97706'],
            motorbike: ['#0284c7', '#d97706', '#dc2626', '#16a34a'],
            auto: ['#15803d'],
            bus: ['#b91c1c'],
            truck: ['#475569', '#334155'],
          }
          const cIdx = Math.floor(Math.random() * colors.car.length)
          const baseColor = type === 'car' ? colors.car[cIdx] : type === 'auto' ? '#15803d' : colors[type][0]
          const roofColor = type === 'car' ? colors.roofs[cIdx] : '#cbd5e1'

          vehiclesRef.current.push({
            id: nextIdRef.current++,
            type,
            lane,
            xOffset: type === 'motorbike' ? (Math.random() - 0.5) * (laneWidth * 0.4) : (Math.random() - 0.5) * 4,
            y: -80,
            speed: type === 'motorbike' ? 2.9 : type === 'bus' ? 1.7 : 2.4,
            targetSpeed: type === 'motorbike' ? 2.9 : type === 'bus' ? 1.7 : 2.4,
            color: baseColor,
            roofColor,
            confidence: Number((0.89 + Math.random() * 0.09).toFixed(2)),
            crossedTripwire: false,
            isBraking: false,
          })
        }
      }

      // ─── 3. Vehicle Physics & Queuing ───
      const vehicles = vehiclesRef.current
      const isRed = signal === 'RED'
      const isYellow = signal === 'YELLOW'

      vehicles.sort((a, b) => b.y - a.y)

      for (let i = 0; i < vehicles.length; i++) {
        const v = vehicles[i]
        const laneCenterX = roadLeft + v.lane * laneWidth + laneWidth / 2 + v.xOffset

        // Find nearest vehicle ahead
        let vehAhead: Vehicle | null = null
        for (let j = 0; j < vehicles.length; j++) {
          if (i !== j && vehicles[j].lane === v.lane && vehicles[j].y > v.y) {
            if (!vehAhead || vehicles[j].y < vehAhead.y) {
              vehAhead = vehicles[j]
            }
          }
        }

        const vehLength = v.type === 'bus' ? 82 : v.type === 'truck' ? 76 : v.type === 'car' ? 46 : v.type === 'auto' ? 34 : 26
        const stopTargetY = stopLineY - 14

        // Deceleration / Stopping logic
        if (vehAhead) {
          const safeDist = vehAhead.y - vehLength - 16
          if (v.y < safeDist) {
            v.targetSpeed = Math.min(v.speed, vehAhead.speed)
            v.isBraking = vehAhead.speed < 1.0
          } else {
            v.targetSpeed = 0
            v.y = Math.min(v.y, safeDist)
            v.isBraking = true
          }
        } else if ((isRed || isYellow) && v.y < stopTargetY) {
          const dist = stopTargetY - v.y
          if (dist < 180) {
            v.targetSpeed = Math.max(0, (dist / 180) * 1.8)
            v.isBraking = true
          }
          if (dist <= 4) {
            v.targetSpeed = 0
            v.y = stopTargetY
            v.isBraking = true
          }
        } else {
          v.targetSpeed = v.type === 'motorbike' ? 2.9 : v.type === 'bus' ? 1.8 : 2.4
          v.isBraking = false
        }

        v.speed += (v.targetSpeed - v.speed) * 0.12
        v.y += v.speed

        // Tripwire crossing event
        if (!v.crossedTripwire && v.y >= tripwireY) {
          v.crossedTripwire = true
          setTripwireFlash(18)
          setCountsByType(prev => ({ ...prev, [v.type]: prev[v.type] + 1 }))
        }

        // Draw vehicle sprite
        drawVehicle(v, laneCenterX, v.y, isIR)
      }

      // Purge offscreen vehicles
      vehiclesRef.current = vehicles.filter(v => v.y < h + 90)

      // ─── 4. Gantry Signal Head on Stop Line ───
      const poleX = roadRight + 18
      const poleY = stopLineY - 20

      // Signal pole
      ctx.strokeStyle = '#475569'
      ctx.lineWidth = 4
      ctx.beginPath()
      ctx.moveTo(poleX, poleY + 40)
      ctx.lineTo(poleX, poleY - 45)
      ctx.stroke()

      // Signal Head Housing
      ctx.fillStyle = '#0f172a'
      ctx.strokeStyle = '#334155'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.roundRect(poleX - 10, poleY - 45, 20, 68, 4)
      ctx.fill()
      ctx.stroke()

      // Red Lens
      const isLensRed = signal === 'RED' || signal === 'ALL_RED'
      ctx.fillStyle = isLensRed ? '#ef4444' : '#450a0a'
      ctx.beginPath(); ctx.arc(poleX, poleY - 32, 6, 0, Math.PI * 2); ctx.fill()
      if (isLensRed) {
        ctx.shadowColor = '#ef4444'; ctx.shadowBlur = 14; ctx.fill(); ctx.shadowBlur = 0
      }

      // Yellow Lens
      const isLensYellow = signal === 'YELLOW'
      ctx.fillStyle = isLensYellow ? '#eab308' : '#422006'
      ctx.beginPath(); ctx.arc(poleX, poleY - 11, 6, 0, Math.PI * 2); ctx.fill()
      if (isLensYellow) {
        ctx.shadowColor = '#eab308'; ctx.shadowBlur = 14; ctx.fill(); ctx.shadowBlur = 0
      }

      // Green Lens
      const isLensGreen = signal === 'GREEN'
      ctx.fillStyle = isLensGreen ? '#22c55e' : '#052e16'
      ctx.beginPath(); ctx.arc(poleX, poleY + 10, 6, 0, Math.PI * 2); ctx.fill()
      if (isLensGreen) {
        ctx.shadowColor = '#22c55e'; ctx.shadowBlur = 14; ctx.fill(); ctx.shadowBlur = 0
      }

      // ─── 5. Infrared CCTV Overlay (if IR mode is active) ───
      if (isIR) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.08)' // Green night vision wash
        ctx.fillRect(0, 0, w, h)
      }

      animationFrameId = requestAnimationFrame(render)
    }

    animationFrameId = requestAnimationFrame(render)
    return () => cancelAnimationFrame(animationFrameId)
  }, [signal, currentApproach, cctvMode, showYoloBoxes, showTripwire, showSpeedHud, tripwireFlash])

  useEffect(() => {
    if (tripwireFlash > 0) {
      const t = setTimeout(() => setTripwireFlash(f => f - 1), 25)
      return () => clearTimeout(t)
    }
  }, [tripwireFlash])

  const cam = cameraInfo[selectedApproach] || cameraInfo.North_Approach
  const totalDetected = countsByType.car + countsByType.motorbike + countsByType.auto + countsByType.bus + countsByType.truck

  return (
    <div className="card space-y-4">
      {/* ─── Top Control Toolbar ─── */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-700/60 flex items-center justify-center text-cyan-400">
            <Video size={18} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-white text-sm tracking-tight">Real-Time CCTV Traffic Surveillance Feed</span>
              <span className="flex items-center gap-1 text-[11px] bg-red-950 text-red-400 border border-red-700 px-2 py-0.5 rounded font-mono font-bold animate-pulse">
                <Radio size={11} /> 25.0 FPS LIVE
              </span>
            </div>
            <div className="text-xs text-slate-400">
              Pune Traffic Police ICCC · Alankar Chowk CCTV Network
            </div>
          </div>
        </div>

        {/* View Mode & Camera Selectors */}
        <div className="flex flex-wrap items-center gap-2">
          {/* View Mode Switcher (Single / Quad) */}
          <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setViewMode('single')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded font-semibold transition-all ${
                viewMode === 'single' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Maximize size={12} /> Focus Cam
            </button>
            <button
              onClick={() => setViewMode('quad')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded font-semibold transition-all ${
                viewMode === 'quad' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Grid size={12} /> Quad 4-Cam Grid
            </button>
          </div>

          {/* CCTV Color / Night Infrared Switch */}
          <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setCctvMode('color')}
              className={`p-1.5 rounded transition-all ${cctvMode === 'color' ? 'bg-slate-700 text-amber-300' : 'text-slate-400 hover:text-white'}`}
              title="Day Color CCTV"
            >
              <Sun size={13} />
            </button>
            <button
              onClick={() => setCctvMode('ir')}
              className={`p-1.5 rounded transition-all ${cctvMode === 'ir' ? 'bg-emerald-800 text-emerald-200' : 'text-slate-400 hover:text-white'}`}
              title="Night Infrared CCTV"
            >
              <Moon size={13} />
            </button>
          </div>
        </div>
      </div>

      {/* ─── Camera Selector Pills ─── */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-slate-900/70 p-2 rounded-lg border border-slate-800/80 text-xs">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-slate-400 font-semibold mr-1 font-mono text-[11px]">APPROACH CAM:</span>
          {Object.entries(cameraInfo).map(([appId, info]) => {
            const isAppGreen = data?.approaches?.[appId]?.signal === 'GREEN'
            return (
              <button
                key={appId}
                onClick={() => setSelectedApproach(appId)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md font-mono text-xs font-semibold transition-all ${
                  selectedApproach === appId
                    ? 'bg-cyan-600 text-white shadow-md shadow-cyan-950 border border-cyan-400/50'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                }`}
              >
                <span>{info.code}</span>
                <span className={`w-2 h-2 rounded-full ${isAppGreen ? 'bg-emerald-400' : 'bg-rose-500'}`} />
                <span className="hidden sm:inline font-sans text-[11px] text-slate-300">{info.direction}</span>
              </button>
            )
          })}
        </div>

        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="text-slate-400">Signal:</span>
          <SignalBadge state={signal} />
        </div>
      </div>

      {/* ─── CCTV Video Player / Quad Grid ─── */}
      {viewMode === 'single' ? (
        <div className="relative rounded-xl overflow-hidden border-2 border-slate-700 bg-black shadow-2xl">
          <canvas
            ref={canvasRef}
            width={840}
            height={460}
            className="w-full h-auto block max-h-[500px] object-cover"
          />

          {/* CCTV Scanline Overlay */}
          <div className="absolute inset-0 cctv-scanline pointer-events-none" />

          {/* OSD: Top-Left Police Department Banner */}
          <div className="absolute top-3.5 left-3.5 bg-black/85 backdrop-blur-md border border-slate-700/80 px-3 py-2 rounded-lg text-left shadow-lg">
            <div className="flex items-center gap-2 text-white font-mono text-xs font-bold">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping inline-block" />
              <span>PUNE POLICE · ICCC TRAFFIC SURVEILLANCE</span>
            </div>
            <div className="text-cyan-400 font-mono text-xs font-semibold mt-0.5">
              {cam.label} · {cam.road}
            </div>
          </div>

          {/* OSD: Top-Right Timestamp & Quality */}
          <div className="absolute top-3.5 right-3.5 bg-black/85 backdrop-blur-md border border-slate-700/80 px-3 py-2 rounded-lg text-right font-mono text-xs shadow-lg">
            <div className="text-amber-400 font-bold">2023-01-11 {timeStr}</div>
            <div className="text-slate-400 text-[10px] mt-0.5">RES: 1920x1080 · FPS: 25.0 · H.264</div>
          </div>

          {/* OSD: Bottom-Left AI Tracker Specs */}
          <div className="absolute bottom-3.5 left-3.5 bg-black/85 backdrop-blur-md border border-slate-700/80 px-3 py-2 rounded-lg text-left font-mono text-xs shadow-lg">
            <div className="flex items-center gap-1.5 text-emerald-400 font-bold text-xs">
              <ShieldCheck size={14} /> YOLOv8n Multi-Object Detector
            </div>
            <div className="text-slate-300 text-[11px] mt-0.5">
              Tripwire Total: <span className="text-cyan-400 font-bold">+{totalDetected} vehicles</span>
            </div>
          </div>

          {/* OSD: Bottom-Right Approach Signal & Telemetry */}
          <div className="absolute bottom-3.5 right-3.5 bg-black/85 backdrop-blur-md border border-slate-700/80 px-3.5 py-2 rounded-lg text-right font-mono shadow-lg">
            <div className="flex items-center justify-end gap-2 mb-1">
              <span className="text-xs text-slate-300">Live Approach Signal:</span>
              <SignalBadge state={signal} />
            </div>
            <div className="text-xs text-slate-300">
              Queue: <span className="text-amber-400 font-bold">{currentApproach?.queue_pcu.toFixed(1) || '0.0'} PCU</span>
              <span className="text-slate-500 mx-1.5">|</span>
              Flow Rate: <span className="text-white font-bold">{currentApproach?.arrival_rate_pcu_per_min.toFixed(1) || '0.0'}/min</span>
            </div>
          </div>
        </div>
      ) : (
        /* ─── QUAD 4-CAM SPLIT SCREEN GRID ─── */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {Object.entries(cameraInfo).map(([appId, info]) => {
            const app = data?.approaches?.[appId]
            const appSig = app?.signal || 'RED'
            const isGreen = appSig === 'GREEN'
            return (
              <div
                key={appId}
                onClick={() => { setSelectedApproach(appId); setViewMode('single') }}
                className={`cursor-pointer rounded-xl overflow-hidden border-2 transition-all relative bg-slate-950 p-2 space-y-2 hover:border-cyan-500 ${
                  selectedApproach === appId ? 'border-cyan-500 shadow-lg shadow-cyan-950' : 'border-slate-800'
                }`}
              >
                <div className="flex items-center justify-between font-mono text-xs px-1">
                  <div className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${isGreen ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
                    <span className="font-bold text-white">{info.code}</span>
                    <span className="text-slate-400 text-[11px]">{info.road}</span>
                  </div>
                  <SignalBadge state={appSig} />
                </div>

                {/* Simulated Mini CCTV view */}
                <div className="h-44 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between p-3 relative overflow-hidden">
                  <div className="absolute inset-0 cctv-scanline pointer-events-none" />
                  <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 z-10">
                    <span className="text-cyan-400 font-bold">● LIVE 25 FPS</span>
                    <span>2023-01-11 {timeStr}</span>
                  </div>

                  {/* Visual Lane Animation Graphic */}
                  <div className="flex-1 flex items-center justify-around py-3 z-10">
                    <div className="flex flex-col items-center gap-1">
                      <Car size={20} className={isGreen ? 'text-emerald-400 animate-bounce' : 'text-slate-400'} />
                      <span className="text-[10px] font-mono text-slate-300">Cars: {app?.arrivals?.car ?? 0}</span>
                    </div>
                    <div className="flex flex-col items-center gap-1">
                      <Zap size={20} className={isGreen ? 'text-emerald-400' : 'text-amber-400'} />
                      <span className="text-[10px] font-mono text-slate-300">Bikes: {app?.arrivals?.motorbike ?? 0}</span>
                    </div>
                    <div className="flex flex-col items-center gap-1">
                      <BusIcon size={20} className={isGreen ? 'text-emerald-400' : 'text-rose-400'} />
                      <span className="text-[10px] font-mono text-slate-300">Buses: {app?.arrivals?.bus ?? 0}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs font-mono bg-black/60 p-1.5 rounded z-10">
                    <span className="text-slate-300">Queue: <strong className="text-amber-400">{app?.queue_pcu.toFixed(1) || '0.0'} PCU</strong></span>
                    <span className="text-slate-300">Rate: <strong className="text-white">{app?.arrival_rate_pcu_per_min.toFixed(1) || '0.0'}/min</strong></span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ─── Real-Time AI Detection Statistics Bar ─── */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1">
        <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Cars Detected</div>
            <div className="text-lg font-bold font-mono text-cyan-400">+{countsByType.car}</div>
          </div>
          <Car size={18} className="text-cyan-500/60" />
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">2-Wheelers</div>
            <div className="text-lg font-bold font-mono text-amber-400">+{countsByType.motorbike}</div>
          </div>
          <Zap size={18} className="text-amber-500/60" />
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Auto-Rickshaws</div>
            <div className="text-lg font-bold font-mono text-emerald-400">+{countsByType.auto}</div>
          </div>
          <Gauge size={18} className="text-emerald-500/60" />
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Buses (PMPML)</div>
            <div className="text-lg font-bold font-mono text-rose-400">+{countsByType.bus}</div>
          </div>
          <BusIcon size={18} className="text-rose-500/60" />
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Commercial Trucks</div>
            <div className="text-lg font-bold font-mono text-slate-300">+{countsByType.truck}</div>
          </div>
          <TruckIcon size={18} className="text-slate-400/60" />
        </div>
      </div>

      {/* ─── CCTV Display Controls ─── */}
      <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 pt-1 border-t border-slate-800/80">
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-1.5 cursor-pointer hover:text-white">
            <input
              type="checkbox"
              checked={showYoloBoxes}
              onChange={e => setShowYoloBoxes(e.target.checked)}
              className="accent-cyan-500 rounded"
            />
            {showYoloBoxes ? <Eye size={13} className="text-emerald-400" /> : <EyeOff size={13} />}
            <span>YOLOv8 Bounding Boxes</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer hover:text-white">
            <input
              type="checkbox"
              checked={showTripwire}
              onChange={e => setShowTripwire(e.target.checked)}
              className="accent-cyan-500 rounded"
            />
            <Zap size={13} className="text-cyan-400" />
            <span>Virtual Counting Tripwire</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer hover:text-white">
            <input
              type="checkbox"
              checked={showSpeedHud}
              onChange={e => setShowSpeedHud(e.target.checked)}
              className="accent-cyan-500 rounded"
            />
            <Gauge size={13} className="text-amber-400" />
            <span>Vehicle Speed Telemetry (km/h)</span>
          </label>
        </div>

        <div className="text-[11px] text-slate-500 font-mono">
          Model: YOLOv8n-COCO (Trained on Indian Heterogeneous Traffic)
        </div>
      </div>
    </div>
  )
}
