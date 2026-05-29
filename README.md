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
- Math is rendered with MathJax and configured for AMS-style equation tags.
- D3-style interactive figures can be inserted with `<figure data-interactive-figure="...">`. New figures should use an `.interactive-figure-plot` element for resize observation and `data-no-expand` on controls that should not open the modal view.

## Deployment

The published site URL is:

```text
https://generativeai-old-and-new.github.io/
```

Quartz uses this as `configuration.baseUrl: "generativeai-old-and-new.github.io"`, without `https://` and without a trailing slash.

- RSS and sitemap are enabled for this domain.
- `Plugin.CustomOgImages()` is still disabled until generated OG images can use local fonts.
- Configure analytics only if the published site should track visits.
