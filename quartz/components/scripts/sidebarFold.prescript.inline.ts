{
  const sidebarFoldStorageKey = "quartz:left-sidebar-folded"

  try {
    if (localStorage.getItem(sidebarFoldStorageKey) === "true") {
      document.documentElement.classList.add("sidebar-folded")
    }
  } catch {
    // Ignore storage access errors in private or locked-down browsing contexts.
  }
}
