const lightboxId = "side-figure-lightbox"

function closeSideFigureLightbox() {
  const lightbox = document.getElementById(lightboxId)
  if (!lightbox) return

  lightbox.classList.remove("is-open")
  lightbox.setAttribute("aria-hidden", "true")
  document.body.classList.remove("side-figure-lightbox-open")
}

function ensureSideFigureLightbox() {
  const existing = document.getElementById(lightboxId)
  if (existing) return existing

  const lightbox = document.createElement("div")
  lightbox.id = lightboxId
  lightbox.className = "side-figure-lightbox"
  lightbox.setAttribute("aria-hidden", "true")
  lightbox.innerHTML = `
    <button class="side-figure-lightbox-close" type="button" aria-label="Close figure preview">
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path d="M18 6 6 18M6 6l12 12" />
      </svg>
    </button>
    <div class="side-figure-lightbox-frame" role="dialog" aria-modal="true">
      <img alt="" />
      <p class="side-figure-lightbox-caption"></p>
    </div>
  `

  const closeButton = lightbox.querySelector<HTMLButtonElement>(".side-figure-lightbox-close")
  const closeOnBackdrop = (event: MouseEvent) => {
    if (event.target === lightbox) closeSideFigureLightbox()
  }
  const closeOnButton = () => closeSideFigureLightbox()
  const closeOnEscape = (event: KeyboardEvent) => {
    if (event.key === "Escape") closeSideFigureLightbox()
  }

  lightbox.addEventListener("click", closeOnBackdrop)
  closeButton?.addEventListener("click", closeOnButton)
  document.addEventListener("keydown", closeOnEscape)

  document.body.append(lightbox)
  return lightbox
}

function openSideFigureLightbox(image: HTMLImageElement) {
  const lightbox = ensureSideFigureLightbox()
  const preview = lightbox.querySelector<HTMLImageElement>(".side-figure-lightbox-frame img")
  const caption = lightbox.querySelector<HTMLElement>(".side-figure-lightbox-caption")
  const source = image.currentSrc || image.src
  const label = image.alt.trim()

  if (preview) {
    preview.src = source
    preview.alt = label
  }

  if (caption) {
    caption.textContent = label
    caption.toggleAttribute("hidden", label === "")
  }

  lightbox.classList.add("is-open")
  lightbox.setAttribute("aria-hidden", "false")
  document.body.classList.add("side-figure-lightbox-open")
  lightbox.querySelector<HTMLButtonElement>(".side-figure-lightbox-close")?.focus()
}

function setupSideFigureLightbox() {
  ensureSideFigureLightbox()

  const figures = document.querySelectorAll<HTMLElement>("article .side-figure")
  for (const figure of figures) {
    if (figure.dataset.lightboxReady === "true") continue

    const image = figure.querySelector<HTMLImageElement>("img")
    if (!image) continue

    figure.dataset.lightboxReady = "true"
    figure.tabIndex = 0
    figure.setAttribute("role", "button")
    figure.setAttribute("aria-label", image.alt ? `Open ${image.alt}` : "Open figure preview")

    const open = () => openSideFigureLightbox(image)
    const openWithKeyboard = (event: KeyboardEvent) => {
      if (event.key !== "Enter" && event.key !== " ") return
      event.preventDefault()
      open()
    }

    figure.addEventListener("click", open)
    figure.addEventListener("keydown", openWithKeyboard)
    window.addCleanup(() => {
      figure.removeEventListener("click", open)
      figure.removeEventListener("keydown", openWithKeyboard)
    })
  }
}

document.addEventListener("nav", setupSideFigureLightbox)
