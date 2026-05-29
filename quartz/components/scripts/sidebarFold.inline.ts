{
  const sidebarFoldStorageKey = "quartz:left-sidebar-folded"
  const sidebarFoldClass = "sidebar-folded"
  const sidebarFoldButtonSelector = ".sidebar-fold-button"
  const sidebarFoldBoundButtons = new WeakSet<HTMLButtonElement>()

  function isSidebarFolded() {
    return document.documentElement.classList.contains(sidebarFoldClass)
  }

  function syncSidebarFoldButtons() {
    const folded = isSidebarFolded()
    const label = folded ? "Expand sidebar" : "Collapse sidebar"

    for (const button of document.querySelectorAll<HTMLButtonElement>(sidebarFoldButtonSelector)) {
      button.setAttribute("aria-pressed", folded ? "true" : "false")
      button.setAttribute("aria-label", label)
      button.title = label
    }
  }

  function getScrollAnchor() {
    const x = Math.max(0, Math.min(window.innerWidth - 1, window.innerWidth / 2))
    const y = Math.max(0, Math.min(window.innerHeight - 1, window.innerHeight * 0.35))
    return document
      .elementsFromPoint(x, y)
      .find(
        (element): element is HTMLElement =>
          element instanceof HTMLElement && !!element.closest(".center"),
      )
  }

  function jumpToScrollPosition(x: number, y: number) {
    const root = document.documentElement
    const previousScrollBehavior = root.style.scrollBehavior

    root.style.scrollBehavior = "auto"
    window.scrollTo(x, y)
    root.style.scrollBehavior = previousScrollBehavior
  }

  function preserveViewportPosition(change: () => void) {
    const scrollX = window.scrollX
    const scrollY = window.scrollY
    const anchor = getScrollAnchor()
    const anchorTop = anchor?.getBoundingClientRect().top

    change()

    const restore = () => {
      if (anchor && anchor.isConnected && anchorTop !== undefined) {
        const delta = anchor.getBoundingClientRect().top - anchorTop
        jumpToScrollPosition(scrollX, window.scrollY + delta)
      } else {
        jumpToScrollPosition(scrollX, scrollY)
      }
    }

    requestAnimationFrame(() => {
      restore()
      requestAnimationFrame(restore)
    })
    window.setTimeout(restore, 420)
  }

  function setSidebarFolded(folded: boolean, preserveScroll = false) {
    const applyFoldState = () => {
      document.documentElement.classList.toggle(sidebarFoldClass, folded)

      try {
        localStorage.setItem(sidebarFoldStorageKey, folded ? "true" : "false")
      } catch {
        // Keep the in-page interaction working even when storage is unavailable.
      }

      syncSidebarFoldButtons()
    }

    if (preserveScroll) {
      preserveViewportPosition(applyFoldState)
    } else {
      applyFoldState()
    }
  }

  function setupSidebarFoldButtons() {
    for (const button of document.querySelectorAll<HTMLButtonElement>(sidebarFoldButtonSelector)) {
      if (sidebarFoldBoundButtons.has(button)) continue

      sidebarFoldBoundButtons.add(button)
      button.addEventListener("click", () => setSidebarFolded(!isSidebarFolded(), true))
    }

    syncSidebarFoldButtons()
  }

  document.addEventListener("nav", setupSidebarFoldButtons)
  setupSidebarFoldButtons()
}
