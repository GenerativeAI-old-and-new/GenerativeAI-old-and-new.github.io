const sidebarCollapsePrescriptStorageKey = "quartz:left-sidebar-collapsed"
const sidebarCollapsedPrescriptClass = "sidebar-left-collapsed"

try {
  if (localStorage.getItem(sidebarCollapsePrescriptStorageKey) === "true") {
    document.documentElement.classList.add(sidebarCollapsedPrescriptClass)
  }
} catch {
  // Ignore storage failures in private browsing or restricted contexts.
}
