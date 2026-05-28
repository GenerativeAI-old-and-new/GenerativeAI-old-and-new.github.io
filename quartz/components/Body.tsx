// @ts-ignore
import clipboardScript from "./scripts/clipboard.inline"
// @ts-ignore
import sidebarCollapseScript from "./scripts/sidebarCollapse.inline"
// @ts-ignore
import sidebarCollapsePrescript from "./scripts/sidebarCollapse.prescript.inline"
import clipboardStyle from "./styles/clipboard.scss"
import { concatenateResources } from "../util/resources"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const Body: QuartzComponent = ({ children }: QuartzComponentProps) => {
  return <div id="quartz-body">{children}</div>
}

Body.beforeDOMLoaded = sidebarCollapsePrescript
Body.afterDOMLoaded = concatenateResources(clipboardScript, sidebarCollapseScript)
Body.css = clipboardStyle

export default (() => Body) satisfies QuartzComponentConstructor
