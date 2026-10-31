import React, { useState } from 'react'
import {
  Clock,
  Activity,
  Layers,
  Brain,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  Zap,
  HelpCircle,
  Gauge,
  Workflow,
  Sparkles,
} from 'lucide-react'

interface StrategyDetail {
  id: string
  name: string
  badge: string
  badgeColor: string
  borderColor: string
  accentColor: string
  oneLiner: string
  icon: React.ReactNode
  howItWorks: string[]
  analogy: string
  decisionFlow: { step: string; detail: string }[]
  pros: string[]
  cons: string[]
  bestFor: string
  keyDifference: string
  intelligence: 'Baseline (0%)' | 'Reactive (50%)' | 'Dynamic (75%)' | 'Physics-Optimal (85%)' | 'Self-Learning AI (90%+)'
  sensorNeed: 'None (Timer only)' | 'Basic Loop / Camera Detectors' | 'Live Queue & Waiting Sensors' | 'Network-Wide Queue Detectors' | 'High-Density Feature Telemetry'
}

const STRATEGIES: StrategyDetail[] = [
  {
    id: 'fixed_time',
    name: 'Fixed-Time Control',
    badge: 'Traditional Baseline',
    badgeColor: 'bg-amber-950/80 text-amber-300 border-amber-600/40',
    borderColor: 'border-amber-700/40',
    accentColor: 'text-amber-400',
    oneLiner: 'Traffic lights cycle through predetermined green and red intervals on a fixed clock, completely oblivious to actual traffic on the road.',
    icon: <Clock size={20} className="text-amber-400" />,
    howItWorks: [
      'Traffic engineers survey an intersection once and count historical vehicle volume.',
      'A static timing plan is programmed into the local signal cabinet (e.g., 30s North-South, 30s East-West).',
      'The controller cycles continuously through the fixed schedule, regardless of whether 50 cars or 0 cars are waiting.',
      'Clearance intervals (3s yellow, 2s all-red) run between every phase change according to standard road safety codes.',
    ],
    analogy: 'Like an old-fashioned kitchen egg timer set to ring every 3 minutes. Even if your toast popped up after 1 minute, the timer cannot see it and keeps ticking blindly until the full time elapses.',
    decisionFlow: [
      { step: '1. Countdown', detail: 'Pre-timed internal clock counts down remaining seconds of current phase.' },
      { step: '2. Clearance', detail: 'Runs 3s yellow warning + 2s all-red clearance to clear intersection.' },
      { step: '3. Next Phase', detail: 'Activates next scheduled direction for fixed pre-programmed duration.' },
    ],
    pros: [
      'Extremely predictable and simple to maintain.',
      'Requires zero road sensors, cameras, or computer processors.',
      'Lowest possible hardware installation and maintenance costs.',
      'Provides a dependable baseline failsafe if all smart sensors fail.',
    ],
    cons: [
      'Forces motorists to wait at red lights facing completely empty cross-streets.',
      'Cannot adapt to rush hour surges, rainstorms, accidents, or sports events.',
      'Generates the highest overall commuter delay, idling fuel burn, and tailpipe emissions.',
    ],
    bestFor: 'Predictable intersections with completely stable, balanced traffic around the clock, or as an emergency failsafe mode.',
    keyDifference: 'The only controller that does not observe the road at all. All other four strategies observe actual traffic conditions.',
    intelligence: 'Baseline (0%)',
    sensorNeed: 'None (Timer only)',
  },
  {
    id: 'actuated',
    name: 'Vehicle-Actuated Control',
    badge: 'Gap-Out Sensing',
    badgeColor: 'bg-emerald-950/80 text-emerald-300 border-emerald-600/40',
    borderColor: 'border-emerald-700/40',
    accentColor: 'text-emerald-400',
    oneLiner: 'Smart vehicle sensors detect arriving cars at the stop bar and dynamically extend or terminate the green light based on vehicle gaps.',
    icon: <Activity size={20} className="text-emerald-400" />,
    howItWorks: [
      'Road loops or overhead video cameras detect vehicles arriving in the approaching lane.',
      'The green light is granted for an initial guaranteed minimum time (e.g. 10 seconds) to ensure pedestrian safety.',
      'Every time another car crosses the detector, the green light extends by a unit extension (typically 2-3 seconds).',
      'If a gap between cars exceeds the threshold ("gap-out") or maximum green is reached ("max-out"), the signal safely transitions to the next road.',
    ],
    analogy: 'Like an automatic sliding door at a supermarket entrance. It slides open when you walk up, stays open as long as a continuous stream of shoppers walks through, and closes promptly once everyone has passed.',
    decisionFlow: [
      { step: '1. Vehicle Detection', detail: 'Sensor registers waiting car; grants safe minimum green (10s).' },
      { step: '2. Unit Extension', detail: 'Each newly detected car extends green by 2-3s up to maximum green.' },
      { step: '3. Gap-Out Yield', detail: 'Once vehicle flow pauses (gap-out), immediately switches to waiting cross-traffic.' },
    ],
    pros: [
      'Dramatically reduces wasted green time on empty approaches.',
      'Exceptionally responsive during off-peak hours and fluctuating traffic.',
      'Proven, industry-standard technology widely deployed across modern signal systems.',
      'Significantly lowers commuter wait times compared to fixed timers.',
    ],
    cons: [
      'Only reacts to cars in the immediate sensor zone; cannot anticipate platoons arriving from down the road.',
      'Cannot coordinate network-wide queues across multiple consecutive intersections.',
      'If a sensor breaks or misidentifies vehicles, timings can degrade.',
    ],
    bestFor: 'Intersections with fluctuating volumes, isolated junctions, and major roads intersected by smaller side streets.',
    keyDifference: 'Reacts locally to gaps between arriving cars rather than calculating mathematical queue scores or running neural networks.',
    intelligence: 'Reactive (50%)',
    sensorNeed: 'Basic Loop / Camera Detectors',
  },
  {
    id: 'adaptive',
    name: 'Weighted Adaptive Control',
    badge: 'Dynamic Multi-Factor',
    badgeColor: 'bg-cyan-950/80 text-cyan-300 border-cyan-600/40',
    borderColor: 'border-cyan-700/40',
    accentColor: 'text-cyan-400',
    oneLiner: 'Continuously measures queues, arrival rates, and wait times across all approaches, computing an urgency score to serve the most congested direction.',
    icon: <Layers size={20} className="text-cyan-400" />,
    howItWorks: [
      'Live CCTV cameras and detection models track queue lengths (PCU) and time elapsed since last green for every approach.',
      'A mathematical objective function calculates a demand urgency score balancing queue length and waiting duration.',
      'The phase with the highest urgency score receives green next, with duration dynamically matched to queue depth.',
      'An anti-starvation ceiling preempts any phase if a low-traffic side street has been waiting past a set threshold (e.g. 65-80s).',
    ],
    analogy: 'Like an experienced traffic police officer standing in the center of the junction, looking around in all four directions, and directing traffic flow to clear whichever lane is backing up the most while making sure no side road is forgotten.',
    decisionFlow: [
      { step: '1. Multi-Camera Scan', detail: 'Continuously tracks queue depth (PCU), arrival rate, and delay across all lanes.' },
      { step: '2. Demand Scoring', detail: 'Calculates Score = Queue × Weight + Wait Time × Starvation Urgency.' },
      { step: '3. Fair Allocation', detail: 'Grants green to highest-score road while enforcing anti-starvation limits.' },
    ],
    pros: [
      'Handles complex, asymmetric traffic surges (e.g., morning inbound vs evening outbound) smoothly.',
      'Balances overall junction throughput with strict fairness to prevent side-street starvation.',
      '100% transparent and explainable: every timing adjustment can be audited and understood.',
      'Robust against unpredictable traffic spikes.',
    ],
    cons: [
      'Requires continuous, accurate video detection or radar feeds on all approaches.',
      'More complex to configure than simple actuated controllers.',
      'Needs junction layout calibration for optimal weighting coefficients.',
    ],
    bestFor: 'High-volume urban intersections, major commuting corridors, and junctions with high vehicle diversity (cars, buses, motorbikes).',
    keyDifference: 'Combines multiple real-time indicators (queue size + waiting duration + starvation timers) into an explainable mathematical formula.',
    intelligence: 'Dynamic (75%)',
    sensorNeed: 'Live Queue & Waiting Sensors',
  },
  {
    id: 'max_pressure',
    name: 'Max-Pressure Control',
    badge: 'Network Physics',
    badgeColor: 'bg-blue-950/80 text-blue-300 border-blue-600/40',
    borderColor: 'border-blue-700/40',
    accentColor: 'text-blue-400',
    oneLiner: 'Treats traffic like pressurized water in a pipe network, prioritizing green lights for roads with the greatest difference between incoming queues and downstream space.',
    icon: <Gauge size={20} className="text-blue-400" />,
    howItWorks: [
      'Measures incoming vehicle queue depth on upstream entry approaches.',
      'Measures available capacity and vehicle queues on the downstream exit links.',
      'Calculates "pressure" for each movement: Pressure = Upstream Queue − Downstream Queue.',
      'Selects the signal phase with the greatest net pressure, maximizing vehicle clearing without causing gridlock downstream.',
    ],
    analogy: 'Like opening valves in a pressurized plumbing network. Water only flows efficiently into a pipe if the receiving chamber has open space. If a downstream pipe is backed up, opening the valve only causes a damaging flood.',
    decisionFlow: [
      { step: '1. Pressure Calculation', detail: 'Measures incoming backlog and subtracts downstream exit congestion.' },
      { step: '2. Network Discharge', detail: 'Activates phase with maximum net pressure to maximize throughput.' },
      { step: '3. Spillback Guard', detail: 'Refuses to send vehicles into exit lanes that are already blocked.' },
    ],
    pros: [
      'Mathematically proven to stabilize queues across dense urban networks.',
      'Prevents intersection "box blocking" and spillback gridlock where cars get trapped in the junction.',
      'Operates autonomously without needing pre-programmed cycle timetables.',
      'Maximizes total vehicle throughput under heavy saturation.',
    ],
    cons: [
      'Requires sensors monitoring downstream exit roads, not just the incoming stop bar.',
      'Can occasionally starve quiet side streets if their net pressure remains consistently lower than busy avenues.',
      'Less intuitive for non-technical traffic managers to understand.',
    ],
    bestFor: 'Dense downtown grids, closely spaced traffic signals, and arterial corridors where intersection spillback is a constant risk.',
    keyDifference: 'The only algorithm that actively inspects downstream exit road capacity before deciding to turn the light green.',
    intelligence: 'Physics-Optimal (85%)',
    sensorNeed: 'Network-Wide Queue Detectors',
  },
  {
    id: 'deep_rl',
    name: 'Deep Reinforcement Learning (PPO)',
    badge: 'Self-Learning AI',
    badgeColor: 'bg-purple-950/80 text-purple-300 border-purple-600/40',
    borderColor: 'border-purple-700/40',
    accentColor: 'text-purple-400',
    oneLiner: 'A neural network AI policy that learns optimal signal timing through simulated trial-and-error across hundreds of thousands of traffic scenarios.',
    icon: <Brain size={20} className="text-purple-400" />,
    howItWorks: [
      'The AI policy receives a 21-dimensional observation vector of the junction (queues, arrival velocities, delays, and elapsed times).',
      'A deep neural network (Proximal Policy Optimization) evaluates the state and selects the optimal signal phase and green duration.',
      'The decision passes through an independent safety supervisor enforcing pedestrian minimum greens, clearance intervals, and starvation limits.',
      'The model receives positive rewards for discharging vehicles and heavy penalties for queues and delays, continually refining its policy.',
    ],
    analogy: 'Like a master chess engine that has played thousands of practice matches against a simulator. Instead of following rigid manual rules, it discovers strategic combinations that minimize city delays.',
    decisionFlow: [
      { step: '1. State Observation', detail: 'Processes 21 normalized metrics capturing full intersection state.' },
      { step: '2. Policy Inference', detail: 'Deep neural network predicts phase and green window for maximum cumulative reward.' },
      { step: '3. Safety Supervisor', detail: 'Independent safety layer audits decision, enforcing yellow/all-red clearances.' },
    ],
    pros: [
      'Can discover subtle, non-linear timing strategies that human engineers might overlook.',
      'Learns and adapts to multi-modal traffic distributions (cars, heavy trucks, two-wheelers).',
      'Constantly improves as it trains against diverse traffic demand scenarios and weather profiles.',
      'Capable of coordinating complex intersections with non-standard topologies.',
    ],
    cons: [
      'Requires substantial training data, simulation compute, and reward calibration.',
      'Operates as an internal "black box", requiring external safety supervisory layers to guarantee road safety codes.',
      'Performance can degrade if traffic conditions drastically diverge from training environments.',
    ],
    bestFor: 'Complex intersections with multi-directional surges, smart city corridors, connected vehicle environments, and future autonomous traffic.',
    keyDifference: 'Learns how to control the intersection through reward-driven simulation rather than following human-written mathematical rules.',
    intelligence: 'Self-Learning AI (90%+)',
    sensorNeed: 'High-Density Feature Telemetry',
  },
]

export default function StrategyGuide() {
  const [expandedId, setExpandedId] = useState<string | null>('adaptive')
  const [selectedFilter, setSelectedFilter] = useState<string>('all')

  const toggleExpand = (id: string) => {
    setExpandedId(prev => (prev === id ? null : id))
  }

  const filteredStrategies = selectedFilter === 'all'
    ? STRATEGIES
    : STRATEGIES.filter(s => s.id === selectedFilter)

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* ── Header Banner ── */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                <Sparkles size={18} />
              </span>
              <h2 className="text-xl font-bold text-white">Traffic Signal Strategy Guide</h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-cyan-950/80 border border-cyan-600/40 text-cyan-300">
                Plain English Reference
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1.5 max-w-3xl leading-relaxed">
              Explore how all five traffic signal control strategies operate at Alankar Chowk, Pune.
              Written in simple, plain language for traffic engineers, city planners, and everyday commuters.
            </p>
          </div>

          {/* Quick Filter Pills */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              onClick={() => setSelectedFilter('all')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                selectedFilter === 'all'
                  ? 'bg-cyan-600 text-white border-cyan-500'
                  : 'bg-slate-800/60 border-slate-700/80 text-slate-400 hover:text-white'
              }`}
            >
              All 5 Strategies
            </button>
            {STRATEGIES.map(s => (
              <button
                key={s.id}
                onClick={() => setSelectedFilter(s.id)}
                className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                  selectedFilter === s.id
                    ? 'bg-slate-700 text-white border-slate-500'
                    : 'bg-slate-800/40 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {s.name.split(' ')[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Quick Highlights Row */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-5">
          {STRATEGIES.map(s => (
            <div
              key={s.id}
              onClick={() => {
                setSelectedFilter('all')
                setExpandedId(s.id)
              }}
              className="bg-slate-950/50 border border-slate-800/80 hover:border-slate-700 p-3 rounded-xl cursor-pointer transition-all hover:bg-slate-800/30 group"
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                  {s.badge}
                </span>
                {s.icon}
              </div>
              <div className="text-xs font-bold text-white mt-1 group-hover:text-cyan-400 transition-colors truncate">
                {s.name}
              </div>
              <div className="text-[11px] text-slate-500 mt-1 truncate">
                {s.intelligence}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Deep Dive Cards ── */}
      <div className="space-y-4">
        {filteredStrategies.map((strat, idx) => {
          const isExpanded = expandedId === strat.id
          return (
            <div
              key={strat.id}
              className={`bg-slate-900 border rounded-2xl transition-all shadow-md overflow-hidden ${
                isExpanded ? strat.borderColor : 'border-slate-800/80 hover:border-slate-700'
              }`}
            >
              {/* Card Header / Toggle */}
              <div
                onClick={() => toggleExpand(strat.id)}
                className="p-5 flex items-center justify-between gap-4 cursor-pointer hover:bg-slate-800/30 transition-colors select-none"
              >
                <div className="flex items-center gap-3.5 min-w-0">
                  <span className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center font-mono text-xs font-bold text-slate-300 shrink-0">
                    #{idx + 1}
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="text-base font-bold text-white">{strat.name}</h3>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${strat.badgeColor}`}>
                        {strat.badge}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">
                      {strat.oneLiner}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <div className="text-right hidden sm:block">
                    <div className="text-[10px] text-slate-500 uppercase font-semibold">Adaptability</div>
                    <div className={`text-xs font-bold font-mono ${strat.accentColor}`}>
                      {strat.intelligence}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition-colors"
                  >
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                </div>
              </div>

              {/* Expandable Body */}
              {isExpanded && (
                <div className="px-5 pb-6 pt-2 border-t border-slate-800/80 space-y-6">
                  {/* Real World Analogy Banner */}
                  <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 flex items-start gap-3">
                    <span className="text-2xl mt-0.5 shrink-0">💡</span>
                    <div>
                      <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                        Relatable Real-World Analogy
                      </h4>
                      <p className="text-xs text-slate-300 mt-1 leading-relaxed italic">
                        "{strat.analogy}"
                      </p>
                    </div>
                  </div>

                  {/* Step-by-Step How It Works */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <Workflow size={14} className={strat.accentColor} />
                      <span>How It Works (Step-by-Step)</span>
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                      {strat.howItWorks.map((step, sIdx) => (
                        <div
                          key={sIdx}
                          className="bg-slate-950/40 border border-slate-800/60 rounded-xl p-3 flex items-start gap-2.5 text-xs text-slate-300"
                        >
                          <span className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center font-mono font-bold text-[10px] text-cyan-400 shrink-0 mt-0.5">
                            {sIdx + 1}
                          </span>
                          <span className="leading-relaxed">{step}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Decision Flow Pipeline */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <Zap size={14} className={strat.accentColor} />
                      <span>Decision-Making Pipeline</span>
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {strat.decisionFlow.map((df, dIdx) => (
                        <div
                          key={dIdx}
                          className="bg-slate-950/40 border border-slate-800/60 rounded-xl p-3 space-y-1 relative"
                        >
                          <div className="text-[11px] font-bold text-white font-mono flex items-center justify-between">
                            <span>{df.step}</span>
                            <span className="text-[10px] text-slate-500">Stage {dIdx + 1}</span>
                          </div>
                          <p className="text-[11px] text-slate-400 leading-snug">
                            {df.detail}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Pros & Cons Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Advantages */}
                    <div className="bg-emerald-950/20 border border-emerald-800/30 rounded-xl p-4 space-y-2.5">
                      <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                        <CheckCircle2 size={14} />
                        <span>Advantages</span>
                      </div>
                      <ul className="space-y-1.5 text-xs text-slate-300">
                        {strat.pros.map((p, pIdx) => (
                          <li key={pIdx} className="flex items-start gap-2">
                            <span className="text-emerald-400 mt-0.5 shrink-0">✓</span>
                            <span>{p}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Disadvantages */}
                    <div className="bg-rose-950/20 border border-rose-800/30 rounded-xl p-4 space-y-2.5">
                      <div className="text-xs font-bold text-rose-400 uppercase tracking-wider flex items-center gap-1.5">
                        <XCircle size={14} />
                        <span>Disadvantages</span>
                      </div>
                      <ul className="space-y-1.5 text-xs text-slate-300">
                        {strat.cons.map((c, cIdx) => (
                          <li key={cIdx} className="flex items-start gap-2">
                            <span className="text-rose-400 mt-0.5 shrink-0">✕</span>
                            <span>{c}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Practical Deployment Context */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-xs">
                    <div className="bg-slate-950/40 border border-slate-800/60 p-3 rounded-xl space-y-1">
                      <span className="font-bold text-slate-300 uppercase tracking-wider text-[11px] block">
                        When It Works Best
                      </span>
                      <p className="text-slate-400 leading-relaxed">
                        {strat.bestFor}
                      </p>
                    </div>

                    <div className="bg-slate-950/40 border border-slate-800/60 p-3 rounded-xl space-y-1">
                      <span className="font-bold text-slate-300 uppercase tracking-wider text-[11px] block">
                        Key Difference
                      </span>
                      <p className="text-slate-400 leading-relaxed">
                        {strat.keyDifference}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* ── Comprehensive Side-by-Side Comparison Matrix ── */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="px-6 py-4 border-b border-slate-800">
          <h3 className="font-bold text-white text-sm flex items-center gap-2">
            <Gauge size={16} className="text-cyan-400" />
            <span>Complete Strategy Comparison Matrix</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Compare intelligence, responsiveness, hardware requirements, and trade-offs side by side
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-800/60 text-[11px] text-slate-400 uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 font-semibold">Strategy</th>
                <th className="py-3 px-4 font-semibold">Intelligence</th>
                <th className="py-3 px-4 font-semibold">Sensor Requirement</th>
                <th className="py-3 px-4 font-semibold">Primary Strength</th>
                <th className="py-3 px-4 font-semibold">Primary Limitation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {STRATEGIES.map(s => (
                <tr key={s.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-4">
                    <div className="font-bold text-white text-xs">{s.name}</div>
                    <div className="text-[11px] text-slate-400">{s.badge}</div>
                  </td>
                  <td className="py-3.5 px-4 font-mono font-medium text-slate-200">
                    <span className={s.accentColor}>{s.intelligence}</span>
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">
                    {s.sensorNeed}
                  </td>
                  <td className="py-3.5 px-4 text-emerald-300 font-medium">
                    {s.pros[0]}
                  </td>
                  <td className="py-3.5 px-4 text-rose-300">
                    {s.cons[0]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Plain-English FAQ & Terms ── */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
          <HelpCircle size={16} className="text-cyan-400" />
          <h3 className="font-bold text-white text-sm">Frequently Asked Traffic Terms</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3 bg-slate-950/40 border border-slate-800/60 rounded-xl space-y-1">
            <span className="font-bold text-white">What is a PCU (Passenger Car Unit)?</span>
            <p className="text-slate-400 leading-relaxed">
              Different vehicles consume different road space. A car equals 1.0 PCU, a motorbike equals 0.5 PCU, and a heavy bus or truck equals 2.5 to 3.0 PCU. PCU allows fair comparison of traffic regardless of vehicle type.
            </p>
          </div>

          <div className="p-3 bg-slate-950/40 border border-slate-800/60 rounded-xl space-y-1">
            <span className="font-bold text-white">What is "Gap-Out"?</span>
            <p className="text-slate-400 leading-relaxed">
              In actuated signals, gap-out happens when the time between consecutive arriving cars exceeds a threshold (e.g. 3 seconds). The controller detects that the queue has cleared and immediately terminates green to avoid idling.
            </p>
          </div>

          <div className="p-3 bg-slate-950/40 border border-slate-800/60 rounded-xl space-y-1">
            <span className="font-bold text-white">What is "Starvation"?</span>
            <p className="text-slate-400 leading-relaxed">
              When a small side road is ignored for too long because a massive main avenue has non-stop traffic. AI2 enforces an anti-starvation ceiling (65–80s) to guarantee fair service for every driver.
            </p>
          </div>

          <div className="p-3 bg-slate-950/40 border border-slate-800/60 rounded-xl space-y-1">
            <span className="font-bold text-white">Why is Safety Supervisor Independent?</span>
            <p className="text-slate-400 leading-relaxed">
              No matter how advanced an AI or algorithm is, traffic lights must strictly obey physical safety codes: non-negotiable yellow clearance (3s), all-red clearance (2s), and no conflicting greens. The Safety Supervisor has veto power over all algorithms.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
