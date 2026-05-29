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
    baseUrl: "generativeai-old-and-new.github.io",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-US",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "local",
      cdnCaching: false,
      typography: {
        title: { name: "Latin Modern Sans", weights: [400, 700], includeItalic: false },
        header: { name: "Latin Modern Sans", weights: [400, 700], includeItalic: false },
        body: { name: "Latin Modern Roman", weights: [400, 700], includeItalic: true },
        code: { name: "Latin Modern Mono", weights: [400], includeItalic: false },
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
      Plugin.TheoremReferences(),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({
        renderEngine: "mathjax",
        customMacros: {
          "\\Qiang": ["{\\color{Maroon}{#1}}", 1],
          "\\qiang": ["{\\color{Maroon}{#1}}", 1],
          "\\qq": ["{\\color{Maroon}{#1}}", 1],
          "\\red": ["{\\color{Maroon}{#1}}", 1],
          "\\med": ["{\\color{Maroon}{#1}}", 1],
          "\\blue": ["{\\color{blue}{#1}}", 1],
          "\\gray": ["{\\color{gray}{#1}}", 1],
          "\\green": ["{\\color{green}{#1}}", 1],
          "\\ant": ["\\qquad {\\color{Maroon}{\\text{// }#1}}", 1],
          "\\d": "\\,{\\rm d}",
          "\\dx": "\\,{\\rm d}x",
          "\\myf": ["\\frac{\\displaystyle #1}{\\displaystyle #2}", 2],
          "\\myp": ["\\frac{\\displaystyle \\partial #1}{\\displaystyle \\partial #2}", 2],
          "\\eqnref": ["\\mathrm{Eqn.}\\,\\eqref{#1}", 1],
          "\\figref": ["\\mathrm{Fig.}\\,\\ref{#1}", 1],
          "\\N": "\\mathbb{N}",
          "\\R": "\\mathbb{R}",
          "\\RR": "\\mathbb{R}",
          "\\Z": "\\mathbb{Z}",
          "\\ZZ": "\\mathbb{Z}",
          "\\Q": "\\mathbb{Q}",
          "\\C": "\\mathbb{C}",
          "\\X": "\\mathcal{X}",
          "\\E": "\\mathbb{E}",
          "\\V": "\\mathbf",
          "\\M": "\\mathbb{M}",
          "\\HH": "\\mathcal{H}",
          "\\B": "\\mathcal{B}",
          "\\D": "\\mathcal{D}",
          "\\A": "\\mathbb{A}",
          "\\vv": ["\\boldsymbol{#1}", 1],
          "\\v": ["\\boldsymbol{#1}", 1],
          "\\argmax": "\\operatorname*{arg\\,max}",
          "\\argmin": "\\operatorname*{arg\\,min}",
          "\\la": "\\langle",
          "\\ra": "\\rangle",
          "\\ind": "\\mathbb{I}",
          "\\sign": "\\operatorname{sign}",
          "\\prob": "\\operatorname{Pr}",
          "\\cd": "\\mid",
          "\\T": "\\boldsymbol{T}",
          "\\F": "\\mathcal{F}",
          "\\norm": ["\\left\\lVert#1\\right\\rVert", 1],
          "\\abs": ["\\left\\lvert#1\\right\\rvert", 1],
          "\\entropy": "\\mathbb{H}",
        },
        mathJaxOptions: {
          tex: {
            tags: "ams",
            tagSide: "right",
            tagIndent: "0.8em",
          },
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
      // Keep generated OG images disabled until local font loading is wired for Satori.
      // Plugin.CustomOgImages(),
    ],
  },
}

export default config
