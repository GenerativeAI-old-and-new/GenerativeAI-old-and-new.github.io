// @ts-ignore
import sidebarFoldScript from "./scripts/sidebarFold.inline"
// @ts-ignore
import sidebarFoldPrescript from "./scripts/sidebarFold.prescript.inline"
import styles from "./styles/sidebarFold.scss"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const SidebarFold: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
  return (
    <button
      type="button"
      class={classNames(displayClass, "sidebar-fold-button")}
      aria-label="Collapse sidebar"
      aria-pressed="false"
      title="Collapse sidebar"
    >
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v13a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 18.5z" />
        <path d="M9 3v18" />
        <path class="sidebar-fold-arrow" d="m16 9-3 3 3 3" />
      </svg>
    </button>
  )
}

SidebarFold.beforeDOMLoaded = sidebarFoldPrescript
SidebarFold.afterDOMLoaded = sidebarFoldScript
SidebarFold.css = styles

export default (() => SidebarFold) satisfies QuartzComponentConstructor
