import { scaleBand, scaleLinear, select } from "d3"
import { InteractiveFigureDefinition } from "./core"
import { readNumber } from "./math"
import { expandButtonHtml } from "./ui"

type ExampleKey = "pronoun" | "local" | "longRange" | "causal"
type HeadKey = "semantic" | "local" | "syntax"
type MaskKey = "full" | "causal"

type AttentionRoutingState = {
  exampleKey: ExampleKey
  headKey: HeadKey
  maskKey: MaskKey
  queryId: string
  temperature: number
}

type TokenSpec = {
  features: number[]
  id: string
  label: string
  value: number[]
}

type RelationBoost = {
  heads: HeadKey[]
  query: string
  strength: number
  target: string
}

type ExampleDefinition = {
  defaultMask: MaskKey
  defaultQueryId: string
  key: ExampleKey
  label: string
  outputLabels: string[]
  relations: RelationBoost[]
  tokens: TokenSpec[]
}

type HeadDefinition = {
  featureWeights: number[]
  key: HeadKey
  label: string
  localBias: number
}

type AttentionModel = {
  outputs: number[][]
  rows: AttentionRow[]
}

type AttentionCell = {
  column: number
  keyId: string
  masked: boolean
  rawScore: number
  weight: number
}

type AttentionRow = {
  cells: AttentionCell[]
  entropy: number
  output: number[]
  queryId: string
}

const exampleKeys: ExampleKey[] = ["pronoun", "local", "longRange", "causal"]
const headKeys: HeadKey[] = ["semantic", "local", "syntax"]
const temperatureDomain: [number, number] = [0.35, 2.5]
const defaultState: AttentionRoutingState = {
  exampleKey: "pronoun",
  headKey: "semantic",
  maskKey: "full",
  queryId: "p-it",
  temperature: 1,
}

const headDefinitions: Record<HeadKey, HeadDefinition> = {
  local: {
    featureWeights: [0.35, 0.25, 0.25, 0.2],
    key: "local",
    label: "local",
    localBias: 0.72,
  },
  semantic: {
    featureWeights: [1.1, 0.95, 0.65, 0.8],
    key: "semantic",
    label: "semantic",
    localBias: 0.08,
  },
  syntax: {
    featureWeights: [0.55, 0.7, 1.1, 0.6],
    key: "syntax",
    label: "syntax",
    localBias: 0.16,
  },
}

const exampleDefinitions: Record<ExampleKey, ExampleDefinition> = {
  causal: {
    defaultMask: "causal",
    defaultQueryId: "c-predict",
    key: "causal",
    label: "causal next-token demo",
    outputLabels: ["subject", "action", "object"],
    relations: [
      { heads: ["semantic"], query: "c-predict", strength: 1.35, target: "c-models" },
      { heads: ["syntax"], query: "c-predict", strength: 1.1, target: "c-to" },
      { heads: ["semantic"], query: "c-token", strength: 1.45, target: "c-next" },
      { heads: ["local"], query: "c-token", strength: 0.85, target: "c-next" },
    ],
    tokens: [
      { features: [0.7, 0.15, 0.25, 0.2], id: "c-we", label: "We", value: [0.78, 0.05, 0.08] },
      {
        features: [0.15, 0.9, 0.45, 0.28],
        id: "c-train",
        label: "train",
        value: [0.12, 0.9, 0.18],
      },
      {
        features: [0.95, 0.18, 0.55, 0.52],
        id: "c-models",
        label: "models",
        value: [0.28, 0.12, 0.9],
      },
      { features: [0.02, 0.08, 0.72, 0.25], id: "c-to", label: "to", value: [0.02, 0.12, 0.05] },
      {
        features: [0.22, 0.95, 0.82, 0.68],
        id: "c-predict",
        label: "predict",
        value: [0.12, 0.88, 0.4],
      },
      { features: [0.02, 0.08, 0.1, 0.15], id: "c-the", label: "the", value: [0.04, 0.02, 0.04] },
      {
        features: [0.3, 0.16, 0.38, 0.98],
        id: "c-next",
        label: "next",
        value: [0.16, 0.16, 0.68],
      },
      {
        features: [0.8, 0.1, 0.38, 0.86],
        id: "c-token",
        label: "token",
        value: [0.2, 0.08, 0.95],
      },
    ],
  },
  local: {
    defaultMask: "full",
    defaultQueryId: "l-chased",
    key: "local",
    label: "local syntax",
    outputLabels: ["actor", "action", "object"],
    relations: [
      { heads: ["syntax"], query: "l-chased", strength: 1.45, target: "l-cat" },
      { heads: ["syntax"], query: "l-chased", strength: 1.25, target: "l-toy" },
      { heads: ["semantic"], query: "l-small", strength: 0.95, target: "l-cat" },
      { heads: ["semantic"], query: "l-bright", strength: 0.95, target: "l-toy" },
    ],
    tokens: [
      { features: [0.02, 0.02, 0.02, 0.1], id: "l-the", label: "The", value: [0.03, 0.02, 0.03] },
      {
        features: [0.12, 0.08, 0.34, 0.92],
        id: "l-small",
        label: "small",
        value: [0.12, 0.02, 0.22],
      },
      { features: [0.92, 0.16, 0.62, 0.25], id: "l-cat", label: "cat", value: [0.9, 0.08, 0.22] },
      {
        features: [0.24, 0.95, 0.82, 0.42],
        id: "l-chased",
        label: "chased",
        value: [0.18, 0.95, 0.35],
      },
      { features: [0.02, 0.02, 0.02, 0.1], id: "l-a", label: "a", value: [0.02, 0.02, 0.04] },
      {
        features: [0.16, 0.08, 0.36, 0.95],
        id: "l-bright",
        label: "bright",
        value: [0.08, 0.02, 0.3],
      },
      { features: [0.9, 0.12, 0.58, 0.28], id: "l-toy", label: "toy", value: [0.22, 0.05, 0.92] },
    ],
  },
  longRange: {
    defaultMask: "full",
    defaultQueryId: "r-wrote",
    key: "longRange",
    label: "long-range dependency",
    outputLabels: ["person", "action", "modifier"],
    relations: [
      { heads: ["syntax"], query: "r-wrote", strength: 1.85, target: "r-researcher" },
      { heads: ["syntax"], query: "r-praised", strength: 1.15, target: "r-committee" },
      { heads: ["semantic"], query: "r-praised", strength: 1.25, target: "r-researcher" },
      { heads: ["semantic"], query: "r-clearly", strength: 1.05, target: "r-wrote" },
    ],
    tokens: [
      { features: [0.02, 0.02, 0.02, 0.08], id: "r-the1", label: "The", value: [0.03, 0.02, 0.02] },
      {
        features: [0.95, 0.14, 0.74, 0.42],
        id: "r-researcher",
        label: "researcher",
        value: [0.95, 0.08, 0.28],
      },
      { features: [0.08, 0.06, 0.52, 0.12], id: "r-who", label: "who", value: [0.12, 0.02, 0.08] },
      { features: [0.02, 0.02, 0.02, 0.08], id: "r-the2", label: "the", value: [0.03, 0.02, 0.02] },
      {
        features: [0.82, 0.12, 0.7, 0.36],
        id: "r-committee",
        label: "committee",
        value: [0.82, 0.06, 0.22],
      },
      {
        features: [0.25, 0.9, 0.78, 0.44],
        id: "r-praised",
        label: "praised",
        value: [0.22, 0.82, 0.24],
      },
      {
        features: [0.35, 0.92, 0.86, 0.5],
        id: "r-wrote",
        label: "wrote",
        value: [0.18, 0.9, 0.3],
      },
      {
        features: [0.08, 0.18, 0.24, 0.94],
        id: "r-clearly",
        label: "clearly",
        value: [0.08, 0.14, 0.82],
      },
    ],
  },
  pronoun: {
    defaultMask: "full",
    defaultQueryId: "p-it",
    key: "pronoun",
    label: "pronoun resolution",
    outputLabels: ["entity", "action", "state"],
    relations: [
      { heads: ["semantic"], query: "p-it", strength: 2.05, target: "p-box" },
      { heads: ["syntax"], query: "p-dropped", strength: 1.2, target: "p-robot" },
      { heads: ["syntax"], query: "p-dropped", strength: 1.1, target: "p-box" },
      { heads: ["semantic"], query: "p-broke", strength: 1.45, target: "p-box" },
    ],
    tokens: [
      { features: [0.02, 0.02, 0.02, 0.1], id: "p-the1", label: "The", value: [0.03, 0.02, 0.02] },
      {
        features: [0.66, 0.28, 0.78, 0.28],
        id: "p-robot",
        label: "robot",
        value: [0.76, 0.08, 0.28],
      },
      {
        features: [0.18, 0.92, 0.66, 0.4],
        id: "p-dropped",
        label: "dropped",
        value: [0.12, 0.9, 0.22],
      },
      { features: [0.02, 0.02, 0.02, 0.1], id: "p-the2", label: "the", value: [0.03, 0.02, 0.02] },
      { features: [0.96, 0.12, 0.62, 0.52], id: "p-box", label: "box", value: [0.95, 0.06, 0.74] },
      {
        features: [0.08, 0.04, 0.16, 0.1],
        id: "p-because",
        label: "because",
        value: [0.04, 0.03, 0.04],
      },
      { features: [0.9, 0.08, 0.56, 0.48], id: "p-it", label: "it", value: [0.72, 0.04, 0.46] },
      {
        features: [0.68, 0.82, 0.58, 0.9],
        id: "p-broke",
        label: "broke",
        value: [0.38, 0.84, 0.82],
      },
    ],
  },
}

function clamp(value: number, minValue: number, maxValue: number) {
  return Math.min(maxValue, Math.max(minValue, value))
}

function formatNumber(value: number, digits = 2) {
  if (!Number.isFinite(value)) return "-"

  return Number.parseFloat(value.toFixed(digits)).toString()
}

function readExampleKey(value: string | undefined): ExampleKey {
  return exampleKeys.includes(value as ExampleKey) ? (value as ExampleKey) : "pronoun"
}

function readHeadKey(value: string | undefined): HeadKey {
  return headKeys.includes(value as HeadKey) ? (value as HeadKey) : "semantic"
}

function readMaskKey(value: string | undefined): MaskKey {
  return value === "causal" ? "causal" : "full"
}

function readTemperature(value: string | undefined) {
  return clamp(
    readNumber(value, defaultState.temperature),
    temperatureDomain[0],
    temperatureDomain[1],
  )
}

function resolveQueryId(example: ExampleDefinition, value: string | undefined) {
  if (!value) return example.defaultQueryId

  const normalized = value.toLowerCase()
  const direct = example.tokens.find((token) => token.id === value)
  if (direct) return direct.id

  const byLabel = example.tokens.find((token) => token.label.toLowerCase() === normalized)
  return byLabel?.id ?? example.defaultQueryId
}

function dot(a: number[], b: number[], weights: number[]) {
  let total = 0
  for (let index = 0; index < Math.min(a.length, b.length, weights.length); index++) {
    total += weights[index] * a[index] * b[index]
  }

  return total
}

function relationBoost(
  example: ExampleDefinition,
  headKey: HeadKey,
  queryId: string,
  targetId: string,
) {
  let boost = 0
  for (const relation of example.relations) {
    if (
      relation.query === queryId &&
      relation.target === targetId &&
      relation.heads.includes(headKey)
    ) {
      boost += relation.strength
    }
  }

  return boost
}

function rawScore(
  example: ExampleDefinition,
  head: HeadDefinition,
  headKey: HeadKey,
  queryIndex: number,
  keyIndex: number,
) {
  const query = example.tokens[queryIndex]
  const key = example.tokens[keyIndex]
  const dimensionScale = Math.sqrt(Math.max(1, head.featureWeights.length))
  const contentScore = dot(query.features, key.features, head.featureWeights) / dimensionScale
  const distance = Math.abs(queryIndex - keyIndex)
  const locality = -head.localBias * distance
  const selfBias = queryIndex === keyIndex ? 0.22 : 0

  return contentScore + locality + selfBias + relationBoost(example, headKey, query.id, key.id)
}

function softmax(scores: number[], temperature: number) {
  const finiteScores = scores.filter(Number.isFinite)
  if (finiteScores.length === 0) return scores.map(() => 0)

  const maxScore = Math.max(...finiteScores)
  const exponentials = scores.map((score) =>
    Number.isFinite(score) ? Math.exp((score - maxScore) / temperature) : 0,
  )
  const total = exponentials.reduce((sum, value) => sum + value, 0)

  if (total <= 0) return scores.map(() => 0)

  return exponentials.map((value) => value / total)
}

function entropy(weights: number[]) {
  return weights.reduce((total, weight) => {
    if (weight <= 0) return total
    return total - weight * Math.log(weight)
  }, 0)
}

function outputForRow(example: ExampleDefinition, weights: number[]) {
  const dimensionCount = example.outputLabels.length
  const output = new Array<number>(dimensionCount).fill(0)

  for (let tokenIndex = 0; tokenIndex < example.tokens.length; tokenIndex++) {
    const token = example.tokens[tokenIndex]
    const weight = weights[tokenIndex]

    for (let dimension = 0; dimension < dimensionCount; dimension++) {
      output[dimension] += weight * (token.value[dimension] ?? 0)
    }
  }

  return output
}

function buildAttentionModel(state: AttentionRoutingState): AttentionModel {
  const example = exampleDefinitions[state.exampleKey]
  const head = headDefinitions[state.headKey]
  const rows: AttentionRow[] = []

  for (let queryIndex = 0; queryIndex < example.tokens.length; queryIndex++) {
    const scores = example.tokens.map((_, keyIndex) => {
      const masked = state.maskKey === "causal" && keyIndex > queryIndex
      return masked ? -Infinity : rawScore(example, head, state.headKey, queryIndex, keyIndex)
    })
    const weights = softmax(scores, state.temperature)

    rows.push({
      cells: scores.map((score, keyIndex) => ({
        column: keyIndex,
        keyId: example.tokens[keyIndex].id,
        masked: !Number.isFinite(score),
        rawScore: score,
        weight: weights[keyIndex],
      })),
      entropy: entropy(weights),
      output: outputForRow(example, weights),
      queryId: example.tokens[queryIndex].id,
    })
  }

  return {
    outputs: rows.map((row) => row.output),
    rows,
  }
}

function strongestCell(row: AttentionRow) {
  return row.cells.reduce((best, cell) => (cell.weight > best.weight ? cell : best), row.cells[0])
}

function selectedRow(example: ExampleDefinition, model: AttentionModel, queryId: string) {
  const index = example.tokens.findIndex((token) => token.id === queryId)
  return model.rows[Math.max(0, index)] ?? model.rows[0]
}

function selectedQueryIndex(example: ExampleDefinition, queryId: string) {
  const index = example.tokens.findIndex((token) => token.id === queryId)
  return Math.max(0, index)
}

function maxOutputAbs(outputs: number[][]) {
  return Math.max(0.1, ...outputs.flatMap((output) => output.map((value) => Math.abs(value))))
}

function readAttentionRoutingState(figure: HTMLElement): AttentionRoutingState {
  const exampleKey = readExampleKey(figure.dataset.example)
  const example = exampleDefinitions[exampleKey]

  return {
    exampleKey,
    headKey: readHeadKey(figure.dataset.head),
    maskKey: readMaskKey(figure.dataset.mask),
    queryId: resolveQueryId(example, figure.dataset.query),
    temperature: readTemperature(figure.dataset.temperature),
  }
}

function cloneAttentionRoutingState(state: AttentionRoutingState): AttentionRoutingState {
  return { ...state }
}

function resetAttentionRouting(state: AttentionRoutingState) {
  Object.assign(state, defaultState)
}

function selectOptions<T extends string>(
  keys: T[],
  selected: T,
  definitions: Record<T, { label: string }>,
) {
  return keys
    .map((key) => {
      const selectedAttribute = key === selected ? " selected" : ""
      return `<option value="${key}"${selectedAttribute}>${definitions[key].label}</option>`
    })
    .join("")
}

function attentionRoutingToolbarHtml(canExpand: boolean) {
  return `<div class="interactive-figure-toolbar attention-routing-toolbar">
    <div class="interactive-figure-title">Attention routing: match, normalize, mix</div>
    <div class="interactive-figure-actions">
      <button class="interactive-figure-resample" type="button" aria-label="Reset attention routing">
        <svg aria-hidden="true" viewBox="0 0 24 24">
          <path d="M20 12a8 8 0 1 1-2.34-5.66" />
          <path d="M20 4v6h-6" />
        </svg>
      </button>
      ${canExpand ? expandButtonHtml() : ""}
    </div>
  </div>`
}

function attentionRoutingTemplate({
  canExpand,
  id,
  state,
}: {
  canExpand: boolean
  id: string
  state: AttentionRoutingState
}) {
  return `
    ${attentionRoutingToolbarHtml(canExpand)}
    <div class="interactive-figure-plot attention-routing-plot"></div>
    <div class="attention-routing-footer" data-no-expand>
      <div class="attention-routing-readout" aria-live="polite"></div>
      <div class="attention-routing-controls">
        <label class="attention-routing-select-control">
          <span>example</span>
          <select class="attention-routing-example" aria-label="Attention example">
            ${selectOptions(exampleKeys, state.exampleKey, exampleDefinitions)}
          </select>
        </label>
        <label class="attention-routing-select-control">
          <span>head</span>
          <select class="attention-routing-head" aria-label="Attention head pattern">
            ${selectOptions(headKeys, state.headKey, headDefinitions)}
          </select>
        </label>
        <label class="attention-routing-temperature-control">
          <span>temperature</span>
          <input class="attention-routing-temperature" type="range" min="${temperatureDomain[0]}" max="${temperatureDomain[1]}" step="0.05" value="${state.temperature}" aria-describedby="${id}-temperature" />
          <output class="attention-routing-temperature-output" id="${id}-temperature">${formatNumber(state.temperature)}</output>
        </label>
        <label class="attention-routing-mask-control">
          <input class="attention-routing-mask" type="checkbox" ${state.maskKey === "causal" ? "checked" : ""} />
          <span>causal mask</span>
        </label>
      </div>
    </div>
  `
}

function syncAttentionRoutingControls(figure: HTMLElement, state: AttentionRoutingState) {
  const exampleSelect = figure.querySelector<HTMLSelectElement>(".attention-routing-example")
  const headSelect = figure.querySelector<HTMLSelectElement>(".attention-routing-head")
  const temperatureSlider = figure.querySelector<HTMLInputElement>(".attention-routing-temperature")
  const temperatureOutput = figure.querySelector<HTMLOutputElement>(
    ".attention-routing-temperature-output",
  )
  const maskCheckbox = figure.querySelector<HTMLInputElement>(".attention-routing-mask")

  if (exampleSelect) exampleSelect.value = state.exampleKey
  if (headSelect) headSelect.value = state.headKey
  if (temperatureSlider) temperatureSlider.value = String(state.temperature)
  if (temperatureOutput) temperatureOutput.value = formatNumber(state.temperature)
  if (maskCheckbox) maskCheckbox.checked = state.maskKey === "causal"

  figure.dataset.example = state.exampleKey
  figure.dataset.head = state.headKey
  figure.dataset.mask = state.maskKey
  figure.dataset.query = state.queryId
  figure.dataset.temperature = String(state.temperature)
}

function updateReadout(
  figure: HTMLElement,
  example: ExampleDefinition,
  selected: AttentionRow,
  state: AttentionRoutingState,
) {
  const readout = figure.querySelector<HTMLElement>(".attention-routing-readout")
  if (!readout) return

  const query = example.tokens.find((token) => token.id === selected.queryId) ?? example.tokens[0]
  const strongest = strongestCell(selected)
  const strongestToken = example.tokens[strongest.column]
  const outputMaxIndex = selected.output.reduce(
    (best, value, index) => (value > selected.output[best] ? index : best),
    0,
  )
  const outputLabel = example.outputLabels[outputMaxIndex]

  readout.innerHTML = `
    <span>query ${query.label}</span>
    <span>top key ${strongestToken.label} ${formatNumber(strongest.weight * 100, 1)}%</span>
    <span>entropy ${formatNumber(selected.entropy)}</span>
    <span>mix ${outputLabel} ${formatNumber(selected.output[outputMaxIndex])}</span>
    <span>${state.maskKey === "causal" ? "causal" : "full"} mask</span>
  `
}

function renderTokenRibbon(
  root: ReturnType<typeof select<SVGGElement, unknown>>,
  example: ExampleDefinition,
  selected: AttentionRow,
  queryIndex: number,
  width: number,
  y: number,
  isCompact: boolean,
) {
  const tokenCount = example.tokens.length
  const gap = isCompact ? 5 : 7
  const chipWidth = Math.max(38, Math.min(84, (width - gap * (tokenCount - 1)) / tokenCount))
  const startX = (width - (chipWidth * tokenCount + gap * (tokenCount - 1))) / 2
  const chipHeight = isCompact ? 24 : 28
  const centers = example.tokens.map(
    (_, index) => startX + index * (chipWidth + gap) + chipWidth / 2,
  )
  const queryCenter = centers[queryIndex]

  const arcs = root.append("g").attr("class", "attention-routing-arcs")
  selected.cells.forEach((cell, index) => {
    if (cell.masked || cell.weight < 0.02 || index === queryIndex) return

    const targetCenter = centers[index]
    const distance = Math.abs(targetCenter - queryCenter)
    const lift = Math.max(18, Math.min(48, distance * 0.28))
    const mid = (targetCenter + queryCenter) / 2

    arcs
      .append("path")
      .attr("class", "attention-routing-arc")
      .attr("d", `M${queryCenter},${y + 4} Q${mid},${y - lift} ${targetCenter},${y + 4}`)
      .style("stroke-width", 0.8 + cell.weight * 6)
      .style("opacity", 0.16 + cell.weight * 0.58)
  })

  const chips = root
    .append("g")
    .attr("class", "attention-routing-token-row")
    .selectAll("g")
    .data(example.tokens)
    .join("g")
    .attr("class", (_, index) =>
      [
        "attention-routing-token",
        index === queryIndex ? "is-query" : "",
        selected.cells[index]?.weight > 0.15 ? "is-attended" : "",
      ]
        .filter(Boolean)
        .join(" "),
    )
    .attr("data-attention-query", (token) => token.id)
    .attr("transform", (_, index) => `translate(${startX + index * (chipWidth + gap)},${y})`)
    .style("cursor", "pointer")

  chips
    .append("rect")
    .attr("width", chipWidth)
    .attr("height", chipHeight)
    .attr("rx", 5)
    .style("fill-opacity", (_, index) => 0.16 + (selected.cells[index]?.weight ?? 0) * 0.62)

  chips
    .append("text")
    .attr("x", chipWidth / 2)
    .attr("y", chipHeight / 2 + 4)
    .attr("text-anchor", "middle")
    .text((token) => token.label)
}

function renderHeatmap(
  root: ReturnType<typeof select<SVGGElement, unknown>>,
  example: ExampleDefinition,
  model: AttentionModel,
  selected: AttentionRow,
  state: AttentionRoutingState,
  x: number,
  y: number,
  width: number,
  height: number,
  isCompact: boolean,
) {
  const tokens = example.tokens
  const queryIndex = selectedQueryIndex(example, state.queryId)
  const labelWidth = isCompact ? 42 : 58
  const topLabelHeight = isCompact ? 18 : 24
  const gridWidth = width - labelWidth
  const gridHeight = height - topLabelHeight
  const cellSize = Math.min(gridWidth / tokens.length, gridHeight / tokens.length)
  const heatWidth = cellSize * tokens.length
  const heatHeight = cellSize * tokens.length
  const xScale = scaleBand<string>()
    .domain(tokens.map((token) => token.id))
    .range([0, heatWidth])
    .paddingInner(0.06)
  const yScale = scaleBand<string>()
    .domain(tokens.map((token) => token.id))
    .range([0, heatHeight])
    .paddingInner(0.06)
  const panel = root
    .append("g")
    .attr("class", "attention-routing-heatmap")
    .attr("transform", `translate(${x},${y})`)

  panel
    .append("text")
    .attr("class", "attention-routing-panel-label")
    .attr("x", 0)
    .attr("y", -8)
    .text("QK score -> softmax weight")

  panel
    .append("rect")
    .attr("class", "attention-routing-panel-bg")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", width)
    .attr("height", height)
    .attr("rx", 5)

  const grid = panel.append("g").attr("transform", `translate(${labelWidth},${topLabelHeight})`)

  grid
    .selectAll("rect")
    .data(model.rows.flatMap((row, rowIndex) => row.cells.map((cell) => ({ cell, rowIndex }))))
    .join("rect")
    .attr("class", ({ cell, rowIndex }) =>
      [
        "attention-routing-cell",
        cell.masked ? "is-masked" : "",
        rowIndex === queryIndex ? "is-selected-row" : "",
      ]
        .filter(Boolean)
        .join(" "),
    )
    .attr("data-attention-query", ({ rowIndex }) => tokens[rowIndex].id)
    .attr("x", ({ cell }) => xScale(cell.keyId) ?? 0)
    .attr("y", ({ rowIndex }) => yScale(tokens[rowIndex].id) ?? 0)
    .attr("width", xScale.bandwidth())
    .attr("height", yScale.bandwidth())
    .style("fill-opacity", ({ cell }) => (cell.masked ? 0.08 : 0.1 + cell.weight * 0.82))

  grid
    .selectAll("text.attention-routing-cell-value")
    .data(selected.cells)
    .join("text")
    .attr("class", "attention-routing-cell-value")
    .attr("x", (cell) => (xScale(cell.keyId) ?? 0) + xScale.bandwidth() / 2)
    .attr("y", () => (yScale(selected.queryId) ?? 0) + yScale.bandwidth() / 2 - 3)
    .attr("text-anchor", "middle")
    .each(function (cell) {
      if (cell.masked || isCompact) return

      const xPosition = (xScale(cell.keyId) ?? 0) + xScale.bandwidth() / 2
      const text = select(this)
      text.append("tspan").attr("x", xPosition).text(formatNumber(cell.rawScore, 1))
      text
        .append("tspan")
        .attr("x", xPosition)
        .attr("dy", "1.15em")
        .text(`${formatNumber(cell.weight * 100, 0)}%`)
    })

  panel
    .selectAll("text.attention-routing-row-label")
    .data(tokens)
    .join("text")
    .attr(
      "class",
      (_, index) => `attention-routing-row-label ${index === queryIndex ? "is-selected" : ""}`,
    )
    .attr("data-attention-query", (token) => token.id)
    .attr("x", labelWidth - 8)
    .attr("y", (token) => topLabelHeight + (yScale(token.id) ?? 0) + yScale.bandwidth() / 2 + 4)
    .attr("text-anchor", "end")
    .text((token) => token.label)

  panel
    .selectAll("text.attention-routing-column-label")
    .data(tokens)
    .join("text")
    .attr("class", "attention-routing-column-label")
    .attr("x", (token) => labelWidth + (xScale(token.id) ?? 0) + xScale.bandwidth() / 2)
    .attr("y", topLabelHeight - 7)
    .attr("text-anchor", "middle")
    .text((token) => (isCompact ? token.label.slice(0, 4) : token.label))

  const rowY = topLabelHeight + (yScale(selected.queryId) ?? 0)
  panel
    .append("rect")
    .attr("class", "attention-routing-selected-row-frame")
    .attr("x", labelWidth - 2)
    .attr("y", rowY - 2)
    .attr("width", heatWidth + 4)
    .attr("height", yScale.bandwidth() + 4)
    .attr("rx", 4)
}

function renderMixer(
  root: ReturnType<typeof select<SVGGElement, unknown>>,
  example: ExampleDefinition,
  model: AttentionModel,
  selected: AttentionRow,
  x: number,
  y: number,
  width: number,
  height: number,
  isCompact: boolean,
) {
  const strongest = strongestCell(selected)
  const query = example.tokens.find((token) => token.id === selected.queryId) ?? example.tokens[0]
  const barHeight = isCompact ? 12 : 14
  const rowGap = isCompact ? 5 : 7
  const labelWidth = isCompact ? 54 : 66
  const weightWidth = width - labelWidth - 18
  const outputTop = y + height - (isCompact ? 82 : 92)
  const outputMax = maxOutputAbs(model.outputs)
  const outputScale = scaleLinear()
    .domain([0, outputMax])
    .range([0, Math.max(54, width - 92)])
  const panel = root
    .append("g")
    .attr("class", "attention-routing-mixer")
    .attr("transform", `translate(${x},${y})`)

  panel
    .append("text")
    .attr("class", "attention-routing-panel-label")
    .attr("x", 0)
    .attr("y", -8)
    .text(`value mix for "${query.label}"`)

  panel
    .append("rect")
    .attr("class", "attention-routing-panel-bg")
    .attr("width", width)
    .attr("height", height)
    .attr("rx", 5)

  const rows = panel
    .append("g")
    .attr("transform", "translate(12,18)")
    .selectAll("g")
    .data(example.tokens)
    .join("g")
    .attr("class", (_, index) =>
      index === strongest.column
        ? "attention-routing-weight-row is-strongest"
        : "attention-routing-weight-row",
    )
    .attr("transform", (_, index) => `translate(0,${index * (barHeight + rowGap)})`)

  rows
    .append("text")
    .attr("x", labelWidth - 8)
    .attr("y", barHeight - 2)
    .attr("text-anchor", "end")
    .text((token) => token.label)

  rows
    .append("rect")
    .attr("class", "attention-routing-weight-track")
    .attr("x", labelWidth)
    .attr("y", 0)
    .attr("width", weightWidth)
    .attr("height", barHeight)
    .attr("rx", 4)

  rows
    .append("rect")
    .attr("class", "attention-routing-weight-fill")
    .attr("x", labelWidth)
    .attr("y", 0)
    .attr("width", (_, index) => weightWidth * selected.cells[index].weight)
    .attr("height", barHeight)
    .attr("rx", 4)
    .style("opacity", (_, index) => 0.32 + selected.cells[index].weight * 0.68)

  rows
    .append("text")
    .attr("class", "attention-routing-weight-value")
    .attr("x", labelWidth + weightWidth + 6)
    .attr("y", barHeight - 2)
    .text((_, index) => formatNumber(selected.cells[index].weight, 2))

  const output = root
    .append("g")
    .attr("class", "attention-routing-output")
    .attr("transform", `translate(${x + 12},${outputTop})`)

  output
    .append("text")
    .attr("class", "attention-routing-output-label")
    .attr("x", 0)
    .attr("y", 0)
    .text("mixed output vector")

  const outputRows = output
    .selectAll("g")
    .data(example.outputLabels)
    .join("g")
    .attr("transform", (_, index) => `translate(0,${18 + index * 21})`)

  outputRows
    .append("text")
    .attr("x", labelWidth - 8)
    .attr("y", 12)
    .attr("text-anchor", "end")
    .text((label) => label)

  outputRows
    .append("rect")
    .attr("class", "attention-routing-output-track")
    .attr("x", labelWidth)
    .attr("y", 2)
    .attr("width", outputScale.range()[1])
    .attr("height", 12)
    .attr("rx", 4)

  outputRows
    .append("rect")
    .attr("class", "attention-routing-output-fill")
    .attr("x", labelWidth)
    .attr("y", 2)
    .attr("width", (_, index) => outputScale(selected.output[index]))
    .attr("height", 12)
    .attr("rx", 4)
}

function renderAttentionRouting(figure: HTMLElement, state: AttentionRoutingState) {
  const plot = figure.querySelector<HTMLElement>(".attention-routing-plot")
  if (!plot) return

  const example = exampleDefinitions[state.exampleKey]
  state.queryId = resolveQueryId(example, state.queryId)

  const model = buildAttentionModel(state)
  const selected = selectedRow(example, model, state.queryId)
  const queryIndex = selectedQueryIndex(example, state.queryId)
  const width = Math.max(plot.clientWidth, 320)
  const isExpanded = figure.classList.contains("is-expanded")
  const isCompact = width < 720 && !isExpanded
  const margin = { bottom: 18, left: isCompact ? 12 : 18, right: isCompact ? 12 : 18, top: 42 }
  const innerWidth = width - margin.left - margin.right
  const ribbonHeight = isCompact ? 82 : 92
  const panelGap = isCompact ? 18 : 20
  const heatmapHeight = isCompact ? Math.min(330, innerWidth * 0.86) : 330
  const mixerHeight = isCompact ? 300 : 330
  const heatmapWidth = isCompact ? innerWidth : Math.min(430, innerWidth * 0.48)
  const mixerWidth = isCompact ? innerWidth : innerWidth - heatmapWidth - panelGap
  const contentTop = margin.top + ribbonHeight
  const contentHeight = isCompact
    ? heatmapHeight + panelGap + mixerHeight
    : Math.max(heatmapHeight, mixerHeight)
  const height = contentTop + contentHeight + margin.bottom

  updateReadout(figure, example, selected, state)
  plot.replaceChildren()

  const svg = select(plot)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img")
    .attr(
      "aria-label",
      "Interactive attention routing figure with tokens, a softmax attention heatmap, and a weighted value mixture",
    )

  const root = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`)

  renderTokenRibbon(root, example, selected, queryIndex, innerWidth, 18, isCompact)
  renderHeatmap(
    root,
    example,
    model,
    selected,
    state,
    0,
    ribbonHeight,
    heatmapWidth,
    heatmapHeight,
    isCompact,
  )
  renderMixer(
    root,
    example,
    model,
    selected,
    isCompact ? 0 : heatmapWidth + panelGap,
    isCompact ? ribbonHeight + heatmapHeight + panelGap : ribbonHeight,
    mixerWidth,
    mixerHeight,
    isCompact,
  )
}

export const attentionRoutingFigure: InteractiveFigureDefinition<AttentionRoutingState> = {
  bindControls: ({ addCleanup, figure, render, state }) => {
    const exampleSelect = figure.querySelector<HTMLSelectElement>(".attention-routing-example")
    const headSelect = figure.querySelector<HTMLSelectElement>(".attention-routing-head")
    const temperatureSlider = figure.querySelector<HTMLInputElement>(
      ".attention-routing-temperature",
    )
    const maskCheckbox = figure.querySelector<HTMLInputElement>(".attention-routing-mask")
    const plot = figure.querySelector<HTMLElement>(".attention-routing-plot")

    const updateExample = () => {
      if (!exampleSelect) return

      state.exampleKey = readExampleKey(exampleSelect.value)
      const example = exampleDefinitions[state.exampleKey]
      state.queryId = example.defaultQueryId
      state.maskKey = example.defaultMask
      render()
    }
    const updateHead = () => {
      if (!headSelect) return

      state.headKey = readHeadKey(headSelect.value)
      render()
    }
    const updateTemperature = () => {
      if (!temperatureSlider) return

      state.temperature = readTemperature(temperatureSlider.value)
      render()
    }
    const updateMask = () => {
      if (!maskCheckbox) return

      state.maskKey = maskCheckbox.checked ? "causal" : "full"
      render()
    }
    const updateQuery = (event: MouseEvent) => {
      const target = event.target
      if (!(target instanceof Element)) return

      const queryTarget = target.closest<SVGElement>("[data-attention-query]")
      const queryId = queryTarget?.dataset.attentionQuery
      if (!queryId) return

      state.queryId = resolveQueryId(exampleDefinitions[state.exampleKey], queryId)
      render()
    }

    exampleSelect?.addEventListener("change", updateExample)
    headSelect?.addEventListener("change", updateHead)
    temperatureSlider?.addEventListener("input", updateTemperature)
    maskCheckbox?.addEventListener("change", updateMask)
    plot?.addEventListener("click", updateQuery)
    addCleanup(() => {
      exampleSelect?.removeEventListener("change", updateExample)
      headSelect?.removeEventListener("change", updateHead)
      temperatureSlider?.removeEventListener("input", updateTemperature)
      maskCheckbox?.removeEventListener("change", updateMask)
      plot?.removeEventListener("click", updateQuery)
    })
  },
  classNames: ["attention-routing-figure"],
  cloneState: cloneAttentionRoutingState,
  expandIgnoreSelector: "button, input, label, select, a, [data-no-expand]",
  readState: readAttentionRoutingState,
  render: renderAttentionRouting,
  resample: resetAttentionRouting,
  serializeState: (state) => ({
    example: state.exampleKey,
    head: state.headKey,
    mask: state.maskKey,
    query: state.queryId,
    temperature: state.temperature,
  }),
  syncControls: syncAttentionRoutingControls,
  template: attentionRoutingTemplate,
  type: "attention-routing",
}
