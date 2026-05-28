export const sampleTicks = [10, 100, 1000, 10000]
export const sampleMin = sampleTicks[0]
export const sampleMax = sampleTicks[sampleTicks.length - 1]
export const sampleSliderMax = 1000

export function readNumber(value: string | undefined, fallback: number) {
  if (!value) return fallback

  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function clampSampleSize(value: number) {
  return Math.min(sampleMax, Math.max(sampleMin, Math.round(value)))
}

export function sampleSizeToSliderValue(value: number) {
  const clamped = clampSampleSize(value)
  const minLog = Math.log10(sampleMin)
  const maxLog = Math.log10(sampleMax)
  const ratio = (Math.log10(clamped) - minLog) / (maxLog - minLog)
  return Math.round(ratio * sampleSliderMax)
}

export function sliderValueToSampleSize(value: number) {
  const minLog = Math.log10(sampleMin)
  const maxLog = Math.log10(sampleMax)
  const ratio = Math.min(1, Math.max(0, value / sampleSliderMax))
  return clampSampleSize(10 ** (minLog + ratio * (maxLog - minLog)))
}

export function seededRandom(seed: number) {
  let state = seed >>> 0

  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

export function sampleExtent(values: number[]) {
  let min = Infinity
  let max = -Infinity

  for (const value of values) {
    if (value < min) min = value
    if (value > max) max = value
  }

  return [min, max] as const
}

export function makeGaussianSamples(mu: number, sigma: number, n: number, seed: number) {
  const random = seededRandom(seed)
  const samples: number[] = []

  while (samples.length < n) {
    const u = Math.max(random(), Number.EPSILON)
    const v = random()
    const radius = Math.sqrt(-2 * Math.log(u))
    const angle = 2 * Math.PI * v

    samples.push(mu + sigma * radius * Math.cos(angle))
    if (samples.length < n) {
      samples.push(mu + sigma * radius * Math.sin(angle))
    }
  }

  return samples
}

export function makeLogNormalSamples(mu: number, sigma: number, n: number, seed: number) {
  return makeGaussianSamples(mu, sigma, n, seed).map((sample) => Math.exp(sample))
}

export function mean(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length
}

export function standardDeviation(values: number[], center: number) {
  const variance = values.reduce((total, value) => total + (value - center) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

export function normalPdf(x: number, mu: number, sigma: number) {
  const z = (x - mu) / sigma
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI))
}

export function logNormalPdf(x: number, mu: number, sigma: number) {
  if (x <= 0) return 0

  const z = (Math.log(x) - mu) / sigma
  return Math.exp(-0.5 * z * z) / (x * sigma * Math.sqrt(2 * Math.PI))
}

export function sortedQuantile(values: number[], probability: number) {
  if (values.length === 0) return 0

  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor(probability * sorted.length)))
  return sorted[index]
}
