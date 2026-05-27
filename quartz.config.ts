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
        body: { name: "Source Serif 4", weights: [400, 500, 600, 700], includeItalic: true },
        code: { name: "IBM Plex Mono", weights: [400, 600], includeItalic: false },
      },
      colors: {
        lightMode: {
          light: "#fbfaf7",
          lightgray: "#e5ded4",
          gray: "#91877b",
          darkgray: "#403a33",
          dark: "#171512",
          secondary: "#a64f31",
          tertiary: "#2f7468",
          highlight: "rgba(166, 79, 49, 0.12)",
          textHighlight: "#f1c75b66",
        },
        darkMode: {
          light: "#171612",
          lightgray: "#37322c",
          gray: "#95897d",
          darkgray: "#ded7ce",
          dark: "#f7f2e9",
          secondary: "#f0a17a",
          tertiary: "#86c6b5",
          highlight: "rgba(240, 161, 122, 0.15)",
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
      Plugin.Latex({ renderEngine: "katex" }),
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
