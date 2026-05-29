export type InteractiveFigureLayout = "inline" | "side" | "modal"

export type InteractiveFigureDefinition<State extends object> = {
  bindControls?: (context: InteractiveFigureControlContext<State>) => void
  classNames: string[]
  cloneState: (state: State) => State
  expandIgnoreSelector?: string
  readState: (figure: HTMLElement) => State
  render: (figure: HTMLElement, state: State) => void
  resizeTargetSelector?: string
  resample?: (state: State) => void
  serializeState: (state: State) => Record<string, number | string>
  syncControls: (figure: HTMLElement, state: State) => void
  template: (context: InteractiveFigureTemplateContext<State>) => string
  type: string
}

export type InteractiveFigureControlContext<State extends object> = {
  addCleanup: (cleanup: () => void) => void
  figure: HTMLElement
  render: () => void
  renderNow: () => void
  scheduleRender: () => void
  state: State
}

type InteractiveFigureTemplateContext<State extends object> = {
  canExpand: boolean
  expanded: boolean
  id: string
  layout: InteractiveFigureLayout
  state: State
}

type AnyFigureDefinition = InteractiveFigureDefinition<any>

type AnyFigureController = {
  definition: AnyFigureDefinition
  destroy: () => void
  figure: HTMLElement
  layout: InteractiveFigureLayout
  render: () => void
  renderNow: () => void
  scheduleRender: () => void
  state: any
  syncFrom: (state: object) => void
}

type CreateFigureOptions = {
  expanded?: boolean
  layout?: InteractiveFigureLayout
  onExpand?: (controller: AnyFigureController) => void
  useGlobalCleanup?: boolean
}

const interactiveFigureSelector = "[data-interactive-figure]"
const interactiveLightboxId = "interactive-figure-lightbox"
const defaultExpandIgnoreSelector = "button, input, label, a, [data-no-expand]"
const defaultResizeTargetSelector = ".interactive-figure-plot"
const registry = new Map<string, AnyFigureDefinition>()
const controllers = new WeakMap<HTMLElement, AnyFigureController>()
const lightboxCleanups = new WeakMap<HTMLElement, () => void>()
const lightboxSources = new WeakMap<HTMLElement, AnyFigureController>()
let interactiveFigureCount = 0

function isFigureLayout(value: string | undefined): value is InteractiveFigureLayout {
  return value === "inline" || value === "side" || value === "modal"
}

function inferFigureLayout(
  figure: HTMLElement,
  options: CreateFigureOptions,
): InteractiveFigureLayout {
  if (options.layout) return options.layout
  if (isFigureLayout(figure.dataset.figureLayout)) return figure.dataset.figureLayout
  if (figure.classList.contains("side-figure")) return "side"
  return "inline"
}

function escapeAttribute(value: number | string) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;")
}

function stateAttributes(definition: AnyFigureDefinition, state: object) {
  return Object.entries(definition.serializeState(state)).map(
    ([key, value]) => `data-${key}="${escapeAttribute(value)}"`,
  )
}

function createInteractiveFigure<State extends object>(
  figure: HTMLElement,
  definition: InteractiveFigureDefinition<State>,
  options: CreateFigureOptions = {},
) {
  const id = `${definition.type}-${++interactiveFigureCount}`
  const disposers: (() => void)[] = []
  const expanded = options.expanded === true
  const layout = inferFigureLayout(figure, options)
  const state = definition.readState(figure)
  const expandIgnoreSelector = definition.expandIgnoreSelector ?? defaultExpandIgnoreSelector
  const resizeTargetSelector = definition.resizeTargetSelector ?? defaultResizeTargetSelector
  let controller: AnyFigureController
  let destroyed = false
  let pendingFrame: number | undefined
  const addCleanup = (cleanup: () => void) => disposers.push(cleanup)

  figure.classList.add("interactive-figure", ...definition.classNames)
  figure.dataset.figureLayout = layout
  figure.classList.toggle("is-expanded", expanded)
  figure.classList.toggle("can-expand", options.onExpand !== undefined)
  figure.replaceChildren()
  figure.innerHTML = definition.template({
    canExpand: options.onExpand !== undefined,
    expanded,
    id,
    layout,
    state,
  })

  const runRender = () => {
    pendingFrame = undefined
    if (destroyed) return

    definition.syncControls(figure, state)
    definition.render(figure, state)
  }
  const renderNow = () => {
    if (destroyed) return

    if (pendingFrame !== undefined) {
      cancelAnimationFrame(pendingFrame)
      pendingFrame = undefined
    }
    runRender()
  }
  const scheduleRender = () => {
    if (destroyed || pendingFrame !== undefined) return

    pendingFrame = requestAnimationFrame(runRender)
  }
  const resampleButton = figure.querySelector<HTMLButtonElement>(".interactive-figure-resample")
  const expandButton = figure.querySelector<HTMLButtonElement>(".interactive-figure-expand")
  const resizeTarget = figure.querySelector<HTMLElement>(resizeTargetSelector)
  const resample = () => {
    definition.resample?.(state)
    scheduleRender()
  }
  const expand = () => options.onExpand?.(controller)
  const expandFromFigure = (event: MouseEvent) => {
    const target = event.target
    if (!(target instanceof Element)) return
    if (target.closest(expandIgnoreSelector)) return

    expand()
  }

  resampleButton?.addEventListener("click", resample)
  expandButton?.addEventListener("click", expand)
  if (options.onExpand) figure.addEventListener("click", expandFromFigure)
  addCleanup(() => {
    resampleButton?.removeEventListener("click", resample)
    expandButton?.removeEventListener("click", expand)
    if (options.onExpand) figure.removeEventListener("click", expandFromFigure)
  })

  definition.bindControls?.({
    addCleanup,
    figure,
    render: scheduleRender,
    renderNow,
    scheduleRender,
    state,
  })

  const observer = new ResizeObserver(scheduleRender)
  if (resizeTarget) observer.observe(resizeTarget)
  addCleanup(() => observer.disconnect())

  controller = {
    definition,
    destroy: () => {
      if (destroyed) return

      destroyed = true
      if (pendingFrame !== undefined) {
        cancelAnimationFrame(pendingFrame)
        pendingFrame = undefined
      }
      for (const dispose of disposers) dispose()
      controllers.delete(figure)
    },
    figure,
    layout,
    render: scheduleRender,
    renderNow,
    scheduleRender,
    state,
    syncFrom: (nextState: object) => {
      Object.assign(state, definition.cloneState(nextState as State))
      scheduleRender()
    },
  }
  controllers.set(figure, controller)
  scheduleRender()

  if (options.useGlobalCleanup !== false) {
    window.addCleanup(controller.destroy)
  }

  return controller
}

function closeInteractiveLightbox(immediate = false) {
  const lightbox = document.getElementById(interactiveLightboxId)
  if (!lightbox) return

  const modalFigure = lightbox.querySelector<HTMLElement>(interactiveFigureSelector)
  const modalController = modalFigure ? controllers.get(modalFigure) : undefined
  const sourceController = lightboxSources.get(lightbox)

  if (
    sourceController &&
    modalController &&
    sourceController.definition.type === modalController.definition.type
  ) {
    sourceController.syncFrom(modalController.state)
  }

  modalController?.destroy()
  lightboxCleanups.get(lightbox)?.()
  lightboxCleanups.delete(lightbox)
  lightboxSources.delete(lightbox)
  lightbox.classList.remove("is-open")
  document.body.classList.remove("interactive-figure-lightbox-open")

  if (immediate) {
    lightbox.remove()
  } else {
    window.setTimeout(() => lightbox.remove(), 180)
  }
}

function openInteractiveLightbox(sourceController: AnyFigureController) {
  closeInteractiveLightbox(true)

  const { definition } = sourceController
  const lightbox = document.createElement("div")
  lightbox.id = interactiveLightboxId
  lightbox.className = "interactive-figure-lightbox"
  lightbox.setAttribute("aria-hidden", "true")
  lightboxSources.set(lightbox, sourceController)
  lightbox.innerHTML = `
    <button class="interactive-figure-lightbox-close" type="button" aria-label="Close interactive figure">
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M18 6 6 18M6 6l12 12" />
      </svg>
    </button>
    <div class="interactive-figure-lightbox-frame" role="dialog" aria-modal="true">
      <figure
        class="interactive-figure"
        data-interactive-figure="${definition.type}"
        data-figure-layout="modal"
        ${stateAttributes(definition, sourceController.state).join("\n        ")}
      ></figure>
    </div>
  `

  const closeButton = lightbox.querySelector<HTMLButtonElement>(
    ".interactive-figure-lightbox-close",
  )
  const modalFigure = lightbox.querySelector<HTMLElement>(interactiveFigureSelector)
  const closeFromBackdrop = (event: MouseEvent) => {
    if (event.target === lightbox) closeInteractiveLightbox()
  }
  const closeFromButton = () => closeInteractiveLightbox()
  const closeFromKeyboard = (event: KeyboardEvent) => {
    if (event.key === "Escape") closeInteractiveLightbox()
  }

  lightbox.addEventListener("click", closeFromBackdrop)
  closeButton?.addEventListener("click", closeFromButton)
  document.addEventListener("keydown", closeFromKeyboard)
  lightboxCleanups.set(lightbox, () => {
    lightbox.removeEventListener("click", closeFromBackdrop)
    closeButton?.removeEventListener("click", closeFromButton)
    document.removeEventListener("keydown", closeFromKeyboard)
  })
  document.body.append(lightbox)
  document.body.classList.add("interactive-figure-lightbox-open")

  if (modalFigure) {
    const modalController = createInteractiveFigure(modalFigure, definition, {
      expanded: true,
      layout: "modal",
      useGlobalCleanup: false,
    })
    modalController.syncFrom(sourceController.state)
  }

  requestAnimationFrame(() => {
    lightbox.classList.add("is-open")
    lightbox.setAttribute("aria-hidden", "false")
    closeButton?.focus()
  })
}

export function registerInteractiveFigure<State extends object>(
  definition: InteractiveFigureDefinition<State>,
) {
  registry.set(definition.type, definition)
}

export function setupInteractiveFigures() {
  const figures = document.querySelectorAll<HTMLElement>(`article ${interactiveFigureSelector}`)

  for (const figure of figures) {
    if (figure.dataset.interactiveReady === "true") continue

    const type = figure.dataset.interactiveFigure
    const definition = type ? registry.get(type) : undefined
    if (!definition) continue

    figure.dataset.interactiveReady = "true"
    createInteractiveFigure(figure, definition, {
      onExpand: openInteractiveLightbox,
    })
  }
}

document.addEventListener("prenav", () => closeInteractiveLightbox(true))
