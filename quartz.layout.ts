import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// === 全站共享：页面级脚本和 footer ===
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [Component.SideFigureLightbox(), Component.InteractiveFigures()],
  footer: Component.Footer({
    showQuartz: false,
    notice: {
      prefix:
        "The site content and visualizations are being actively updated. If you have any questions or suggestions, please email ",
      text: "liaorl@cs.utexas.edu",
      href: "mailto:liaorl@cs.utexas.edu",
      suffix: ".",
    },
    credits: [
      {
        prefix: "Lecture notes by ",
        text: "Qiang Liu",
        href: "https://www.cs.utexas.edu/~lqiang/",
      },
      {
        prefix: "Made by ",
        text: "Runlong Liao",
        href: "https://github.com/CyberDragon93",
      },
    ],
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    // Component.ConditionalRender({
    //   component: Component.Breadcrumbs(),
    //   condition: (page) => page.fileData.slug !== "index",
    // }),
    Component.ArticleTitle(),
    Component.TagList(),
  ],
  left: [
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      gap: "0.5rem",
      components: [
        { Component: Component.SidebarFold(), shrink: false },
        { Component: Component.HomeButton(), shrink: false },
        { Component: Component.Darkmode(), shrink: false },
        { Component: Component.Search(), basis: "10.4rem" },
        // { Component: Component.ReaderMode() },
      ],
    }),
    // 非首页：显示目录（ToC）
    Component.ConditionalRender({
      component: Component.DesktopOnly(Component.TableOfContents()),
      condition: (page) => page.fileData.slug !== "index",
    }),
  ],
  right: [
    // Component.Graph(),
    // Component.DesktopOnly(Component.TableOfContents()),
    // Component.Backlinks(),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle()],
  left: [
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      gap: "0.5rem",
      components: [
        { Component: Component.SidebarFold(), shrink: false },
        { Component: Component.HomeButton(), shrink: false },
        { Component: Component.Darkmode(), shrink: false },
        { Component: Component.Search(), basis: "10.4rem" },
      ],
    }),
  ],
  right: [],
}
