import { useStore } from '@nanostores/react'
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceRadial,
  forceSimulation,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum
} from 'd3-force'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { $learningError, $learningGraph, $learningLoading, loadLearningGraph } from '@/store/learning'
import type { LearningGraph, LearningNode } from '@/types/hermes'

import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

// ── The one view ────────────────────────────────────────────────────────────
// A tilted, top-down EVE-style star map of what Hermes has learned. Time is
// RADIAL: oldest learning sits at the galactic core, newer memories/skills
// accrete onto outer rings. The disk is tilted for depth; recent systems burn
// hot while old ones cool toward the core. Hover lights a constellation and
// shows a dated tooltip; click locks focus. Canvas-rendered. No modes.

const RING_INNER = 78
const RING_OUTER = 340
const ZOOM_MIN = 0.3
const ZOOM_MAX = 5
const FIT_PADDING = 80
const TILT = 0.6 // vertical squash → "looking down at a tilted disk"

interface Viewport {
  k: number
  x: number
  y: number
}

interface SimNode extends LearningNode, SimulationNodeDatum {
  rec: number // recency 0 (oldest) → 1 (newest)
  tr: number // time-anchored target radius
  x: number
  y: number
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  source: SimNode | string
  target: SimNode | string
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

function hash(input: string): number {
  let h = 2166136261

  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }

  return h >>> 0
}

// Monotone, theme-driven palette: every shade is derived from the theme's
// `foreground` color (read off the canvas at draw time), so the map is a single
// ink on the overlay and adapts to light/dark. Recency is conveyed by opacity,
// memory vs skill by shape — no hues.
//
// Theme tokens come through `color-mix()`/oklch, so getComputedStyle returns a
// non-rgb() string. Rasterize it through a 1x1 canvas to get real sRGB bytes —
// naive string parsing of oklab()/color(srgb …) silently yields black.
let _probe: CanvasRenderingContext2D | null = null

function resolveRgb(color: string): { b: number; g: number; r: number } {
  if (!_probe) {
    const c = document.createElement('canvas')
    c.width = 1
    c.height = 1
    _probe = c.getContext('2d', { willReadFrequently: true })
  }

  if (!_probe) {
    return { b: 184, g: 163, r: 148 }
  }

  _probe.clearRect(0, 0, 1, 1)
  _probe.fillStyle = '#888888'
  _probe.fillStyle = color
  _probe.fillRect(0, 0, 1, 1)
  const d = _probe.getImageData(0, 0, 1, 1).data

  return { b: d[2], g: d[1], r: d[0] }
}

function luminance(r: number, g: number, b: number): number {
  return (0.2126 * r + 0.7152 * g + 0.114 * b) / 255
}

function nodeRadius(n: LearningNode): number {
  if (n.kind === 'memory') {
    return 4.4
  }

  const base = n.state === 'archived' || n.state === 'stale' ? 2.4 : 3

  return base + Math.sqrt(Math.max(0, n.useCount)) * 0.55 + (n.pinned ? 0.8 : 0)
}

function formatDate(ts?: null | number): string {
  if (!ts) {
    return 'unknown'
  }

  try {
    return new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
  } catch {
    return 'unknown'
  }
}

// Tag-style badge items for the hover tooltip — date first. Use-count is NOT a
// badge (rendered separately, right-aligned) so it's excluded here.
function metaBadges(n: LearningNode): string[] {
  const out: string[] = [formatDate(n.timestamp)]

  if (n.kind === 'memory') {
    out.push(n.memorySource === 'profile' ? 'profile memory' : 'memory')
  } else {
    out.push(n.category)

    if (n.createdBy === 'agent') {
      out.push('learned')
    }

    if (n.pinned) {
      out.push('pinned')
    }
  }

  return out.filter(Boolean)
}

// Bare "xN" use-count, last in the badge row. Null when never used.
function countLabel(n: LearningNode): null | string {
  return n.kind === 'skill' && n.useCount > 0 ? `x${n.useCount}` : null
}

// Footer-row content for the tooltip. Reserved primitive — returns nothing for
// now (skills have no UUID; their id is just the name). Wire real detail here
// later and the tooltip will lay it out automatically.
function nodeFooter(node: LearningNode): null | string {
  void node

  return null
}

// Greedy word-wrap for the tooltip title so long memory lines don't blow out.
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  const lines: string[] = []
  let line = ''

  for (const word of words) {
    const next = line ? `${line} ${word}` : word

    if (!line || ctx.measureText(next).width <= maxW) {
      line = next
    } else {
      lines.push(line)
      line = word
    }
  }

  if (line) {
    lines.push(line)
  }

  return lines
}

function fitViewport(w: number, h: number): Viewport {
  if (w <= 0 || h <= 0) {
    return { k: 1, x: w / 2, y: h / 2 }
  }

  const spanX = (RING_OUTER + 30) * 2
  const spanY = spanX * TILT
  const k = clamp(Math.min((w - FIT_PADDING * 2) / spanX, (h - FIT_PADDING * 2) / spanY, 2.2), ZOOM_MIN, ZOOM_MAX)

  return { k, x: w / 2, y: h / 2 }
}

function StarMap({ graph }: { graph: LearningGraph }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  const simRef = useRef<null | Simulation<SimNode, SimLink>>(null)
  const nodesRef = useRef<SimNode[]>([])
  const linksRef = useRef<SimLink[]>([])
  const byIdRef = useRef(new Map<string, SimNode>())
  const adjacencyRef = useRef(new Map<string, Set<string>>())
  const memByIdRef = useRef(new Map<string, LearningGraph['memory'][number]>())
  const ringsRef = useRef<Array<{ label: null | string; r: number; ratio: number }>>([])
  const ringLabelRectsRef = useRef<Array<{ h: number; i: number; w: number; x: number; y: number }>>([])
  const starsRef = useRef<Array<{ a: number; r: number; x: number; y: number }>>([])

  const viewportRef = useRef<Viewport>({ k: 1, x: 0, y: 0 })
  const hoverRef = useRef<null | string>(null)
  const hoveredRingRef = useRef<null | number>(null)
  const selectedRingRef = useRef<null | number>(null)
  const selectedIdRef = useRef<null | string>(null)
  const sizeRef = useRef({ h: 0, w: 0 })
  const dprRef = useRef(1)
  const dirtyRef = useRef(true)

  const dragRef = useRef<{
    id: null | string
    mode: 'none' | 'pan'
    moved: boolean
    ring: null | number
    sx: number
    sy: number
    vp: Viewport
  }>({ id: null, mode: 'none', moved: false, ring: null, sx: 0, sy: 0, vp: { k: 1, x: 0, y: 0 } })

  const [selectedId, setSelectedId] = useState<null | string>(null)
  const [size, setSize] = useState({ h: 0, w: 0 })
  const loading = useStore($learningLoading)

  const memById = useMemo(() => {
    const m = new Map<string, LearningGraph['memory'][number]>()
    graph.memory.forEach((card, i) => m.set(`memory:${card.source}:${i}`, card))

    return m
  }, [graph.memory])

  const adjacency = useMemo(() => {
    const m = new Map<string, Set<string>>()

    for (const n of graph.nodes) {
      m.set(n.id, new Set())
    }

    for (const e of graph.edges) {
      m.get(e.source)?.add(e.target)
      m.get(e.target)?.add(e.source)
    }

    return m
  }, [graph.edges, graph.nodes])

  useEffect(() => {
    const el = wrapRef.current

    if (!el) {
      return
    }

    const sync = () => setSize({ h: el.clientHeight, w: el.clientWidth })
    const ro = new ResizeObserver(sync)
    ro.observe(el)
    sync()

    return () => ro.disconnect()
  }, [])

  // Build the simulation: time → radius, oldest at the core.
  useEffect(() => {
    sizeRef.current = size

    if (size.w === 0 || size.h === 0) {
      return
    }

    const known = graph.nodes
      .map(n => (typeof n.timestamp === 'number' && Number.isFinite(n.timestamp) ? Number(n.timestamp) : null))
      .filter((v): v is number => v !== null)

    const minTs = known.length ? Math.min(...known) : null
    const maxTs = known.length ? Math.max(...known) : null
    const timed = minTs !== null && maxTs !== null && maxTs > minTs

    const ordered = [...graph.nodes].sort((a, b) => {
      const at = typeof a.timestamp === 'number' ? a.timestamp : Infinity
      const bt = typeof b.timestamp === 'number' ? b.timestamp : Infinity

      return at === bt ? a.id.localeCompare(b.id) : at - bt
    })

    const ordRatio = new Map(ordered.map((n, i) => [n.id, ordered.length > 1 ? i / (ordered.length - 1) : 0]))

    const ratioFor = (n: LearningNode): number => {
      if (timed && typeof n.timestamp === 'number' && minTs !== null && maxTs !== null) {
        return (Number(n.timestamp) - minTs) / (maxTs - minTs)
      }

      return ordRatio.get(n.id) ?? 0
    }

    const nodes: SimNode[] = graph.nodes.map(n => {
      const rec = ratioFor(n)
      const tr = RING_INNER + rec * (RING_OUTER - RING_INNER)
      const seed = hash(n.id)
      const angle = ((seed % 3600) / 3600) * Math.PI * 2

      return { ...n, rec, tr, vx: 0, vy: 0, x: Math.cos(angle) * tr, y: Math.sin(angle) * tr }
    })

    const byId = new Map(nodes.map(n => [n.id, n]))

    const links: SimLink[] = graph.edges
      .filter(e => byId.has(e.source) && byId.has(e.target))
      .map(e => ({ source: e.source, target: e.target }))

    // Radial force dominates so a node's distance from the core faithfully
    // encodes its time (it sits on/near its date ring); charge + collide only
    // spread nodes around that ring, they don't drag them off it.
    const sim = forceSimulation(nodes)
      .alphaDecay(0.05)
      .velocityDecay(0.62)
      .force('charge', forceManyBody<SimNode>().strength(-12))
      .force(
        'link',
        forceLink<SimNode, SimLink>(links)
          .id(n => n.id)
          .distance(26)
          .strength(0.06)
      )
      .force(
        'collide',
        forceCollide<SimNode>()
          .radius(n => nodeRadius(n) + 2)
          .iterations(2)
      )
      .force('radial', forceRadial<SimNode>(n => (n as SimNode).tr, 0, 0).strength(0.92))
      .on('tick', () => {
        dirtyRef.current = true
      })

    const rings: Array<{ label: null | string; r: number; ratio: number }> = []
    const steps = 4

    for (let i = 0; i <= steps; i += 1) {
      const ratio = i / steps
      const r = RING_INNER + ratio * (RING_OUTER - RING_INNER)

      const label =
        timed && minTs !== null && maxTs !== null ? formatDate(Math.round(minTs + (maxTs - minTs) * ratio)) : null

      rings.push({ label, r, ratio })
    }

    simRef.current = sim
    nodesRef.current = nodes
    linksRef.current = links
    byIdRef.current = byId
    ringsRef.current = rings
    viewportRef.current = fitViewport(size.w, size.h)
    dirtyRef.current = true

    if (selectedIdRef.current && !byId.has(selectedIdRef.current)) {
      selectedIdRef.current = null
      setSelectedId(null)
    }

    return () => {
      sim.stop()

      if (simRef.current === sim) {
        simRef.current = null
      }
    }
  }, [graph.edges, graph.nodes, size])

  useEffect(() => {
    const count = clamp(Math.round((size.w * size.h) / 8000), 50, 200)
    starsRef.current = Array.from({ length: count }, (_, i) => {
      const s = hash(`star-${i}-${size.w}x${size.h}`)

      return {
        a: 0.1 + ((s >>> 18) % 55) / 100,
        r: 0.4 + ((s >>> 8) % 12) / 12,
        x: s % Math.max(1, size.w),
        y: (s >>> 12) % Math.max(1, size.h)
      }
    })
    dirtyRef.current = true
  }, [size])

  useEffect(() => {
    adjacencyRef.current = adjacency
    memByIdRef.current = memById
    dirtyRef.current = true
  }, [adjacency, memById])

  useEffect(() => {
    selectedIdRef.current = selectedId
    dirtyRef.current = true
  }, [selectedId])

  // Repaint when the theme/mode changes (toggles class + inline vars on <html>).
  useEffect(() => {
    const mo = new MutationObserver(() => {
      dirtyRef.current = true
    })

    mo.observe(document.documentElement, {
      attributeFilter: ['class', 'style', 'data-hermes-mode', 'data-hermes-theme'],
      attributes: true
    })

    return () => mo.disconnect()
  }, [])

  // Render loop.
  useEffect(() => {
    let raf = 0

    const loop = () => {
      if (dirtyRef.current) {
        dirtyRef.current = false
        draw()
      }

      raf = requestAnimationFrame(loop)
    }

    const draw = () => {
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')

      if (!canvas || !ctx) {
        return
      }

      const { h, w } = sizeRef.current
      const dpr = dprRef.current
      const vp = viewportRef.current
      const nodes = nodesRef.current
      const byId = byIdRef.current
      const adj = adjacencyRef.current
      // Unified focus precedence (nodes and rings behave 1:1): a SELECTED item
      // locks the view; hover only applies when nothing is selected; node-focus
      // and ring-focus are mutually exclusive, so hovering one kind while the
      // other is selected never switches the highlight (tooltip still shows).
      let focus: null | string = null
      let focusRing: null | number = null

      if (selectedIdRef.current) {
        focus = selectedIdRef.current
      } else if (selectedRingRef.current != null) {
        focusRing = selectedRingRef.current
      } else if (hoverRef.current) {
        focus = hoverRef.current
      } else if (hoveredRingRef.current != null) {
        focusRing = hoveredRingRef.current
      }

      const focusSet = focus ? (adj.get(focus) ?? new Set<string>()) : null

      // Tilted projection: y is squashed for the "looking down at a disk" feel.
      const projX = (wx: number) => wx * vp.k + vp.x
      const projY = (wy: number) => wy * vp.k * TILT + vp.y

      // Stark monotone: pure white ink on dark, pure black on light. The theme
      // foreground is only used to detect which mode we're in.
      const fg = resolveRgb(getComputedStyle(canvas).color)
      const darkTheme = luminance(fg.r, fg.g, fg.b) > 0.55
      const base = darkTheme ? { b: 255, g: 255, r: 255 } : { b: 0, g: 0, r: 0 }
      const shade = (a: number) => `rgba(${base.r},${base.g},${base.b},${a})`
      const chipBg = darkTheme ? 'rgba(0,0,0,0.72)' : 'rgba(255,255,255,0.85)'
      // Inverted ink (background color) for the flipped focused tooltip.
      const inkInv = darkTheme ? 'rgba(0,0,0,1)' : 'rgba(255,255,255,1)'

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      // Starfield backdrop (screen space).
      ctx.fillStyle = shade(1)

      for (const s of starsRef.current) {
        ctx.globalAlpha = s.a * (darkTheme ? 0.32 : 0.5)
        ctx.beginPath()
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2)
        ctx.fill()
      }

      ctx.globalAlpha = 1

      // Tilted world transform for the disk structure + jump routes.
      ctx.setTransform(vp.k * dpr, 0, 0, vp.k * TILT * dpr, vp.x * dpr, vp.y * dpr)

      const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, RING_INNER * 1.5)
      coreGrad.addColorStop(0, shade(0.05))
      coreGrad.addColorStop(0.5, shade(0.02))
      coreGrad.addColorStop(1, shade(0))
      ctx.fillStyle = coreGrad
      ctx.beginPath()
      ctx.arc(0, 0, RING_INNER * 1.5, 0, Math.PI * 2)
      ctx.fill()

      const ringIdx = focusRing
      const ring = ringIdx != null ? (ringsRef.current[ringIdx] ?? null) : null
      const active = !!focus || !!ring
      ctx.lineWidth = 1 / vp.k
      ctx.setLineDash([6 / vp.k, 5 / vp.k])
      ringsRef.current.forEach((rg, i) => {
        ctx.strokeStyle = shade(ringIdx === i ? 0.5 : active ? 0.24 : darkTheme ? 0.16 : 0.1)
        ctx.beginPath()
        ctx.arc(0, 0, rg.r, 0, Math.PI * 2)
        ctx.stroke()
      })
      ctx.setLineDash([])
      ctx.strokeStyle = shade(active ? 0.13 : darkTheme ? 0.08 : 0.05)

      for (let i = 0; i < 6; i += 1) {
        const a = (i / 6) * Math.PI * 2
        ctx.beginPath()
        ctx.moveTo(Math.cos(a) * RING_INNER, Math.sin(a) * RING_INNER)
        ctx.lineTo(Math.cos(a) * RING_OUTER, Math.sin(a) * RING_OUTER)
        ctx.stroke()
      }

      // Switch to screen space for jump routes + glyphs (crisp, easy to trim).
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      // Jump routes. A focused node's links stop at its selection ring, not its
      // glyph, so the ring reads cleanly.
      const focusNode = focus ? (byId.get(focus) ?? null) : null
      const focusRingR = focusNode ? (nodeRadius(focusNode) + focusNode.rec) * vp.k + 4 : 0

      for (const link of linksRef.current) {
        const s = typeof link.source === 'object' ? link.source : byId.get(String(link.source))
        const t = typeof link.target === 'object' ? link.target : byId.get(String(link.target))

        if (!s || !t) {
          continue
        }

        const lit =
          !!focus && (s.id === focus || t.id === focus || (!!focusSet && focusSet.has(s.id) && focusSet.has(t.id)))

        let x1 = projX(s.x)
        let y1 = projY(s.y)
        let x2 = projX(t.x)
        let y2 = projY(t.y)

        if (s.id === focus) {
          const d = Math.hypot(x2 - x1, y2 - y1) || 1
          x1 += ((x2 - x1) / d) * focusRingR
          y1 += ((y2 - y1) / d) * focusRingR
        }

        if (t.id === focus) {
          const d = Math.hypot(x1 - x2, y1 - y2) || 1
          x2 += ((x1 - x2) / d) * focusRingR
          y2 += ((y1 - y2) / d) * focusRingR
        }

        ctx.strokeStyle = lit ? shade(0.72) : shade(darkTheme ? 0.17 : 0.1)
        ctx.lineWidth = lit ? 1.6 : 0.85
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
      }

      // Systems (nodes): opacity encodes recency — newest fully opaque, oldest
      // faded toward a floor (eased), with a slight size bump for recent ones.
      ctx.fillStyle = shade(1)

      for (const n of nodes) {
        const r = (nodeRadius(n) + n.rec) * vp.k
        const isFocus = n.id === focus
        const isNeighbor = !!focusSet && focusSet.has(n.id)
        const inRing = !!ring && Math.abs(n.rec - ring.ratio) <= 0.13
        const dim = focus ? !isFocus && !isNeighbor : ring ? !inRing : false
        const sx = projX(n.x)
        const sy = projY(n.y)

        const recAlpha = 0.35 + 0.65 * (1 - (1 - n.rec) ** 3)
        ctx.globalAlpha = dim ? 0.16 : isFocus ? 1 : recAlpha

        if (n.kind === 'memory') {
          ctx.beginPath()
          ctx.moveTo(sx, sy - r)
          ctx.lineTo(sx + r, sy)
          ctx.lineTo(sx, sy + r)
          ctx.lineTo(sx - r, sy)
          ctx.closePath()
          ctx.fill()
        } else {
          ctx.beginPath()
          ctx.arc(sx, sy, r, 0, Math.PI * 2)
          ctx.fill()
        }

        if (isFocus) {
          ctx.globalAlpha = 1
          ctx.strokeStyle = shade(1)
          ctx.lineWidth = 1.4
          ctx.beginPath()
          ctx.arc(sx, sy, r + 4, 0, Math.PI * 2)
          ctx.stroke()
        }
      }

      ctx.globalAlpha = 1

      // Ring date labels (top of each ellipse) — hoverable to focus the ring.
      ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
      ctx.textAlign = 'center'
      ringLabelRectsRef.current = []
      ringsRef.current.forEach((rg, i) => {
        if (!rg.label) {
          return
        }

        const sx = projX(0)
        const sy = projY(-rg.r)

        if (sy < 8 || sy > h - 8) {
          return
        }

        const tw = ctx.measureText(rg.label).width
        const boxW = tw + 6
        // Any focus (a node OR another date) fades the non-focused dates —
        // label and bg together.
        const isThis = ringIdx === i
        const faded = (focus != null || ringIdx != null) && !isThis
        // EVE-style: labels float in space (no chip). A bg-colored halo keeps
        // them legible over rings/links; everything stays stark monotone.
        ctx.globalAlpha = faded ? 0.33 : 1
        ctx.lineJoin = 'round'
        ctx.lineWidth = 3
        ctx.strokeStyle = inkInv
        ctx.strokeText(rg.label, sx, sy + 3)
        ctx.fillStyle = shade(isThis ? 1 : 0.55)
        ctx.fillText(rg.label, sx, sy + 3)
        ctx.globalAlpha = 1
        ringLabelRectsRef.current.push({ h: 18, i, w: boxW + 6, x: sx - boxW / 2 - 3, y: sy - 10 })
      })

      // Constellation labels for the focus + its neighbors (name only).
      ctx.font = '11px ui-sans-serif, system-ui, sans-serif'

      for (const id of focusSet ?? []) {
        if (id === hoverRef.current) {
          continue
        }

        const n = byId.get(id)

        if (!n) {
          continue
        }

        const sx = projX(n.x)
        const sy = projY(n.y) - (nodeRadius(n) * vp.k + 7)
        const tw = ctx.measureText(n.label).width
        ctx.fillStyle = chipBg
        ctx.fillRect(sx - tw / 2 - 4, sy - 11, tw + 8, 15)
        ctx.fillStyle = shade(0.85)
        ctx.fillText(n.label, sx, sy)
      }

      // Tooltip on focus (hover OR selection). The metabar and the title each
      // get their own background that fills to their own width — no shared box,
      // so a long title never leaves blank gaps in the metabar.
      const tip = focus ? byId.get(focus) : null

      if (tip) {
        const PADX = 6
        const PADY = 4
        const BADGE_H = 14
        const ROW_GAP = 3
        const LINE_H = 16
        const badgeFont = '9px ui-sans-serif, system-ui, sans-serif'
        const monoFont = '9px ui-monospace, SFMono-Regular, Menlo, monospace'
        const titleFont = '600 11px ui-sans-serif, system-ui, sans-serif'
        // The date (index 0) stays sans; the rest of the tags are monospace.
        const badgeFontFor = (i: number) => (i === 0 ? badgeFont : monoFont)

        const badges = metaBadges(tip)
        const use = countLabel(tip)

        const titleText =
          tip.kind === 'memory' ? memByIdRef.current.get(tip.id)?.body.split('\n')[0]?.trim() || tip.label : tip.label

        // Metabar metrics — plain text: no background, no chips, no padding.
        const ITEM_GAP = 8

        const badgeW = badges.map((b, i) => {
          ctx.font = badgeFontFor(i)

          return ctx.measureText(b).width
        })

        const rowW = badgeW.reduce((a, b) => a + b, 0) + ITEM_GAP * Math.max(0, badges.length - 1)
        ctx.font = monoFont
        const useW = use ? ctx.measureText(use).width : 0
        const metaW = rowW + (use ? ITEM_GAP + useW : 0)

        // Title metrics (wrapped) — title keeps its own filled (inverted) bg.
        ctx.font = titleFont
        const maxTitleW = Math.min(380, w - 16) - PADX * 2
        const titleLines = wrapText(ctx, titleText, maxTitleW)
        const titleW = Math.max(0, ...titleLines.map(l => ctx.measureText(l).width))
        const titleBgW = titleW + PADX * 2
        const titleBgH = titleLines.length * LINE_H + PADY * 2

        // Footer primitive — reserved for future per-node detail; nothing now.
        const footerText = nodeFooter(tip)
        const FOOTER_H = 13
        const footerFont = '9px ui-sans-serif, system-ui, sans-serif'
        ctx.font = footerFont
        const footerW = footerText ? ctx.measureText(footerText).width : 0

        const contentW = Math.max(metaW, footerW)
        const totalW = Math.max(contentW, titleBgW)
        const totalH = BADGE_H + ROW_GAP + titleBgH + (footerText ? ROW_GAP + FOOTER_H : 0)
        const bx = clamp(projX(tip.x) - totalW / 2, 4, Math.max(4, w - totalW - 4))
        const by = clamp(projY(tip.y) - (nodeRadius(tip) * vp.k + 8) - totalH, 4, Math.max(4, h - totalH - 4))

        const textX = bx + PADX

        ctx.textAlign = 'left'
        ctx.textBaseline = 'middle'

        const badgeMidY = by + BADGE_H / 2

        // Metadata sits flush at the left edge (no padding).
        ctx.fillStyle = shade(0.7)
        let cx = bx
        badges.forEach((label, i) => {
          ctx.font = badgeFontFor(i)
          ctx.fillText(label, cx, badgeMidY)
          cx += badgeW[i] + ITEM_GAP
        })

        if (use) {
          ctx.font = monoFont
          ctx.fillStyle = shade(0.5)
          ctx.fillText(use, cx, badgeMidY)
        }

        // Title: inverted (fg/bg flipped) so the focused tooltip pops.
        const ty = by + BADGE_H + ROW_GAP
        ctx.fillStyle = shade(1)
        ctx.fillRect(bx, ty, titleBgW, titleBgH)
        ctx.font = titleFont
        ctx.fillStyle = inkInv
        titleLines.forEach((line, i) => {
          ctx.fillText(line, textX, ty + PADY + LINE_H * i + LINE_H / 2)
        })

        // Footer primitive (renders only when nodeFooter() returns content).
        if (footerText) {
          ctx.font = footerFont
          ctx.fillStyle = shade(0.45)
          ctx.fillText(footerText, bx, ty + titleBgH + ROW_GAP + FOOTER_H / 2)
        }

        ctx.textBaseline = 'alphabetic'
      }
    }

    raf = requestAnimationFrame(loop)

    return () => cancelAnimationFrame(raf)
  }, [])

  // Size the backing canvas (DPR-aware).
  useEffect(() => {
    sizeRef.current = size
    dprRef.current = Math.min(2, window.devicePixelRatio || 1)
    const canvas = canvasRef.current

    if (canvas && size.w > 0 && size.h > 0) {
      canvas.width = Math.round(size.w * dprRef.current)
      canvas.height = Math.round(size.h * dprRef.current)
      canvas.style.width = `${size.w}px`
      canvas.style.height = `${size.h}px`
    }

    dirtyRef.current = true
  }, [size])

  // ── Interactions (invert the tilted projection for hit-testing) ───────────
  const pickNode = (cssX: number, cssY: number): null | SimNode => {
    const vp = viewportRef.current
    const wx = (cssX - vp.x) / vp.k
    const wy = (cssY - vp.y) / (vp.k * TILT)
    let best: null | SimNode = null
    let bestD = Infinity

    for (const n of nodesRef.current) {
      const r = nodeRadius(n) + 6
      const d = (n.x - wx) ** 2 + (n.y - wy) ** 2

      if (d < r * r && d < bestD) {
        bestD = d
        best = n
      }
    }

    return best
  }

  const pickRingLabel = (cssX: number, cssY: number): null | number => {
    for (const r of ringLabelRectsRef.current) {
      if (cssX >= r.x && cssX <= r.x + r.w && cssY >= r.y && cssY <= r.y + r.h) {
        return r.i
      }
    }

    return null
  }

  const localXY = (e: React.MouseEvent): { x: number; y: number } => {
    const rect = canvasRef.current?.getBoundingClientRect()

    return { x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) }
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) {
      return
    }

    const { x, y } = localXY(e)
    const ringHit = pickRingLabel(x, y)
    hoveredRingRef.current = null
    // Nodes aren't draggable (static map) — remember which was pressed so a
    // click (press without movement) can select it; any drag just pans.
    const nodeId = ringHit == null ? (pickNode(x, y)?.id ?? null) : null
    dragRef.current = {
      id: nodeId,
      mode: 'pan',
      moved: false,
      ring: ringHit,
      sx: e.clientX,
      sy: e.clientY,
      vp: viewportRef.current
    }
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current

    if (drag.mode === 'none') {
      const { x, y } = localXY(e)
      const ringHit = pickRingLabel(x, y)
      const id = ringHit == null ? (pickNode(x, y)?.id ?? null) : null

      if (id !== hoverRef.current || ringHit !== hoveredRingRef.current) {
        hoverRef.current = id
        hoveredRingRef.current = ringHit
        dirtyRef.current = true
      }

      const canvas = canvasRef.current

      if (canvas) {
        canvas.style.cursor = id || ringHit != null ? 'pointer' : 'grab'
      }

      return
    }

    const dx = e.clientX - drag.sx
    const dy = e.clientY - drag.sy

    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      drag.moved = true
    }

    if (drag.mode === 'pan') {
      viewportRef.current = { ...drag.vp, x: drag.vp.x + dx, y: drag.vp.y + dy }
      dirtyRef.current = true
    }
  }

  const endDrag = () => {
    const drag = dragRef.current

    // A click (press without movement) selects a ring date, a node, or clears.
    if (drag.mode === 'pan' && !drag.moved) {
      if (drag.ring != null) {
        selectedRingRef.current = selectedRingRef.current === drag.ring ? null : drag.ring
        setSelectedId(null)
      } else if (drag.id) {
        selectedRingRef.current = null
        setSelectedId(prev => (prev === drag.id ? null : drag.id))
      } else {
        selectedRingRef.current = null
        setSelectedId(null)
      }

      dirtyRef.current = true
    }

    dragRef.current = { id: null, mode: 'none', moved: false, ring: null, sx: 0, sy: 0, vp: viewportRef.current }
  }

  const onMouseLeave = () => {
    hoverRef.current = null
    hoveredRingRef.current = null
    dirtyRef.current = true
    endDrag()
  }

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()

    if (!rect) {
      return
    }

    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const vp = viewportRef.current
    const k = clamp(vp.k * (e.deltaY > 0 ? 0.9 : 1.1), ZOOM_MIN, ZOOM_MAX)
    viewportRef.current = { k, x: px - ((px - vp.x) / vp.k) * k, y: py - ((py - vp.y) / vp.k) * k }
    dirtyRef.current = true
  }

  const resetView = () => {
    viewportRef.current = fitViewport(sizeRef.current.w, sizeRef.current.h)
    selectedRingRef.current = null
    dirtyRef.current = true
    setSelectedId(null)
  }

  return (
    <div className="relative min-h-0 flex-1 overflow-hidden" ref={wrapRef}>
      <canvas
        className="block touch-none select-none text-foreground"
        onDoubleClick={resetView}
        onMouseDown={onMouseDown}
        onMouseLeave={onMouseLeave}
        onMouseMove={onMouseMove}
        onMouseUp={endDrag}
        onWheel={onWheel}
        ref={canvasRef}
      />

      <div className="pointer-events-none absolute left-2 top-2 flex flex-col gap-1 bg-background/40 px-2 py-1.5 text-[0.62rem] text-muted-foreground backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block size-2 rounded-full bg-foreground/70" /> skill
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block size-2 rotate-45 bg-foreground/70" /> memory
          </span>
        </div>
        <div className="text-[0.58rem] text-muted-foreground/65">core = oldest · outer rings = newer</div>
      </div>

      <div className="absolute right-3 top-2 flex items-center gap-3 text-[0.65rem]">
        <button className="text-muted-foreground hover:text-foreground" onClick={resetView} type="button">
          Reset view
        </button>
        <button
          className="text-muted-foreground underline underline-offset-2 hover:text-foreground disabled:opacity-50"
          disabled={loading}
          onClick={() => void loadLearningGraph(true)}
          type="button"
        >
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
    </div>
  )
}

export function LearningView({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const graph = useStore($learningGraph)
  const loading = useStore($learningLoading)
  const error = useStore($learningError)

  useEffect(() => {
    void loadLearningGraph()
  }, [])

  const skillCount = graph ? graph.nodes.filter(n => n.kind === 'skill').length : 0
  const memoryCount = graph ? graph.nodes.filter(n => n.kind === 'memory').length : 0
  const subtitle = graph ? `${skillCount} learned skills · ${memoryCount} memories, over time` : undefined

  return (
    <Panel closeLabel={t.learning.close} onClose={onClose}>
      <PanelHeader subtitle={subtitle} title={t.learning.title} />

      {error ? (
        <PanelEmpty description={error} icon="warning" title={t.learning.loadFailed} />
      ) : !graph && loading ? (
        <div aria-label={t.learning.loading} className="grid flex-1 place-items-center" role="status">
          <div className="flex flex-col items-center gap-3">
            <Loader className="size-12 text-muted-foreground" strokeScale={0.72} type="spiral-search" />
            <span className="text-xs text-muted-foreground/70">{t.learning.loading}</span>
          </div>
        </div>
      ) : graph && graph.nodes.length === 0 ? (
        <PanelEmpty description={t.learning.emptyDesc} icon="lightbulb" title={t.learning.emptyTitle} />
      ) : graph ? (
        <StarMap graph={graph} />
      ) : null}
    </Panel>
  )
}
