import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

const ExplorerPane = Component.Explorer({
  title: "Contents",
  folderClickBehavior: "link",
  folderDefaultState: "open",
  useSavedState: false,
  // filterFn: (n) => !["tags","assets"].includes(n.name),
  // sortFn: (a,b) => (a.order ?? 1e9) - (b.order ?? 1e9) || a.displayName.localeCompare(b.displayName),
})

// === 全站共享：把站点标题和 ExplorerPane 放到页面最上方（随页面滚动；不是 fixed） ===
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [Component.PageTitle()],
  afterBody: [],
  footer: Component.Footer({
    links: {
      "By Qiang Liu": "https://www.cs.utexas.edu/~lqiang/",
    },
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
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        { Component: Component.Search(), grow: true },
        { Component: Component.Darkmode() },
        // { Component: Component.ReaderMode() },
      ],
    }),
    // 非首页：显示目录（ToC）
    Component.ConditionalRender({
      component: Component.DesktopOnly(Component.TableOfContents()),
      condition: (page) => page.fileData.slug !== "index",
    }),
    // 首页：显示站点“Contents”树（Explorer）
    Component.ConditionalRender({
      component: Component.DesktopOnly(ExplorerPane),
      condition: (page) => page.fileData.slug === "index",
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
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
  ],
  right: [],
}
