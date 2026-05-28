import { sampleTicks } from "./math"

export function resampleButtonHtml() {
  return `<button class="interactive-figure-resample" type="button" aria-label="Draw new samples">
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M20 12a8 8 0 1 1-2.34-5.66" />
      <path d="M20 4v6h-6" />
    </svg>
  </button>`
}

export function expandButtonHtml() {
  return `<button class="interactive-figure-expand" type="button" aria-label="Open interactive figure">
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M8 3H5a2 2 0 0 0-2 2v3" />
      <path d="M16 3h3a2 2 0 0 1 2 2v3" />
      <path d="M8 21H5a2 2 0 0 1-2-2v-3" />
      <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
    </svg>
  </button>`
}

export function toolbarHtml(title: string, canExpand: boolean) {
  return `<div class="interactive-figure-toolbar">
    <div class="interactive-figure-title">${title}</div>
    <div class="interactive-figure-actions">
      ${resampleButtonHtml()}
      ${canExpand ? expandButtonHtml() : ""}
    </div>
  </div>`
}

export function sampleTicksHtml(id: string) {
  return `<div class="gaussian-histogram-ticks" id="${id}" aria-hidden="true">
    ${sampleTicks.map((stop) => `<span>${stop}</span>`).join("")}
  </div>`
}

export function tooltipHtml() {
  return `<div class="gaussian-histogram-tooltip" hidden></div>`
}
