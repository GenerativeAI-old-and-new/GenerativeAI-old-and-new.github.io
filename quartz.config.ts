import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Generative AI: Old and New",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "quartz.jzhao.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: { name: "IBM Plex Sans", weights: [500, 600, 700], includeItalic: false },
        body: { name: "STIX Two Text", weights: [400, 500, 600, 700], includeItalic: true },
        code: { name: "IBM Plex Mono", weights: [400, 600], includeItalic: false },
      },
      colors: {
        lightMode: {
          light: "#fffdf8",
          lightgray: "#e5e0d8",
          gray: "#81796f",
          darkgray: "#2e2a25",
          dark: "#111111",
          secondary: "#2c5f9e",
          tertiary: "#2c7d54",
          highlight: "rgba(44, 95, 158, 0.1)",
          textHighlight: "#f1c75b66",
        },
        darkMode: {
          light: "#171612",
          lightgray: "#37322c",
          gray: "#95897d",
          darkgray: "#ded7ce",
          dark: "#f7f2e9",
          secondary: "#8ab7f2",
          tertiary: "#8bcfa8",
          highlight: "rgba(138, 183, 242, 0.14)",
          textHighlight: "#d5a93d66",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({
        renderEngine: "mathjax",
        mathJaxOptions: {
          svg: {
            fontCache: "local",
            mtextInheritFont: false,
            scale: 1,
          },
        },
      }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
