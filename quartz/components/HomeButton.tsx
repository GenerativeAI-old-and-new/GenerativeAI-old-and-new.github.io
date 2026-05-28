import { FullSlug, pathToRoot } from "../util/path"
import { classNames } from "../util/lang"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const HomeButton: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
  const href = pathToRoot(fileData.slug ?? ("index" as FullSlug))

  return (
    <a class={classNames(displayClass, "home-button")} href={href} aria-label="Home">
      <svg role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 10.8 12 3l9 7.8" />
        <path d="M5.5 9.5V21h13V9.5" />
        <path d="M9.5 21v-6h5v6" />
      </svg>
    </a>
  )
}

HomeButton.css = `
.home-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.1rem;
  height: 2.1rem;
  padding: 0;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--surface);
  color: var(--darkgray);
  text-decoration: none;
  transition:
    border-color 0.16s ease,
    background-color 0.16s ease,
  color 0.16s ease;
}

.home-button svg {
  width: 1.05rem;
  height: 1.05rem;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  fill: none;
}

.home-button:hover,
.home-button:focus-visible {
  border-color: var(--secondary);
  background: color-mix(in srgb, var(--surface) 82%, var(--highlight));
  color: var(--secondary);
  text-decoration: none;
}
`

export default (() => HomeButton) satisfies QuartzComponentConstructor
