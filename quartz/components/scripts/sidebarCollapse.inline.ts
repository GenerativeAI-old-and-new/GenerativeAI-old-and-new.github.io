const sidebarCollapseStorageKey = "quartz:left-sidebar-collapsed"
const sidebarCollapsedClass = "sidebar-left-collapsed"

function persistSidebarCollapsed(collapsed: boolean) {
  try {
    localStorage.setItem(sidebarCollapseStorageKey, String(collapsed))
  } catch {
    // Ignore storage failures in private browsing or restricted contexts.
  }
}

function readSidebarCollapsed() {
  try {
    return localStorage.getItem(sidebarCollapseStorageKey) === "true"
  } catch {
    return false
  }
}

function setSidebarCollapsed(collapsed: boolean) {
  document.documentElement.classList.toggle(sidebarCollapsedClass, collapsed)

  for (const button of document.querySelectorAll<HTMLButtonElement>(".sidebar-collapse-button")) {
    const label = collapsed ? "Show contents" : "Hide contents"
    button.setAttribute("aria-expanded", String(!collapsed))
    button.setAttribute("aria-label", label)
    button.setAttribute("title", label)
  }

  persistSidebarCollapsed(collapsed)
}

function setupSidebarCollapse() {
  const buttons = document.querySelectorAll<HTMLButtonElement>(".sidebar-collapse-button")
  if (buttons.length === 0) return

  setSidebarCollapsed(
    document.documentElement.classList.contains(sidebarCollapsedClass) || readSidebarCollapsed(),
  )

  for (const button of buttons) {
    if (button.dataset.sidebarCollapseBound === "true") continue

    button.dataset.sidebarCollapseBound = "true"
    button.addEventListener("click", () => {
      setSidebarCollapsed(!document.documentElement.classList.contains(sidebarCollapsedClass))
    })
  }
}

setupSidebarCollapse()
document.addEventListener("nav", setupSidebarCollapse)
