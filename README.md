# Math Notes Quartz Template

This repository is a Quartz-based template for math-intensive notes, lecture notes, and technical blogs. It keeps the current course notes as example content and publishes at <https://generativeai-old-and-new.github.io/>.

## Development

```bash
npm install
npm run dev
```

Useful checks:

```bash
npm test
npm run check
npx quartz build
```

Quartz's upstream documentation is still available in `docs/` and can be previewed with:

```bash
npm run dev:docs
```

## Authoring Features

- Obsidian-style callouts are available for theorem-like writing, including `example`, `proof`, `remark`, `theorem`, `definition`, `lemma`, `proposition`, and `corollary`.
- Numbered theorem-like callouts can be labelled with Pandoc-style ids, for example `> [!theorem|Change of Variables] {#thm:change-vars}`. Reference them in prose with `@thm:change-vars`; unresolved references render visibly and print a build warning.
- Numbered callouts use module-aware numbering when the page slug starts with a number, for example `modules/01-probability-basics` renders `Theorem 1.1` and `Example 1.1`. Pages without a numeric slug use page-local numbering such as `Theorem 1`.
- Markdown images with real captions are normalized into numbered figures. Add labels with `![Caption {#fig:example}](/assets/example.png)` and reference them with `@fig:example`. Placeholder alt text like `image` stays unnumbered.
- Math is rendered with MathJax and configured for AMS-style equation tags.
- D3-style interactive figures can be inserted with `<figure data-interactive-figure="...">`. New figures should be exported from `quartz/components/scripts/interactive-figures/index.ts`, use an `.interactive-figure-plot` element for resize observation, and keep figure-specific styles in `quartz/styles/notes/figures/`.

## Deployment

The published site URL is:

```text
https://generativeai-old-and-new.github.io/
```

Quartz uses this as `configuration.baseUrl: "generativeai-old-and-new.github.io"`, without `https://` and without a trailing slash.

- RSS and sitemap are enabled for this domain.
- `Plugin.CustomOgImages()` is still disabled until generated OG images can use local fonts.
- Configure analytics only if the published site should track visits.
