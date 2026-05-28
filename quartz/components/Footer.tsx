import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/footer.scss"
import { version } from "../../package.json"
import { i18n } from "../i18n"

interface FooterCredit {
  text: string
  href?: string
  prefix?: string
  suffix?: string
}

interface Options {
  links?: Record<string, string>
  credits?: FooterCredit[]
  showQuartz?: boolean
}

export default ((opts?: Options) => {
  const Footer: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const legacyLinks = opts?.links
      ? Object.entries(opts.links).map(([text, href]) => ({ text, href }))
      : []
    const credits = opts?.credits ?? legacyLinks
    return (
      <footer class={`${displayClass ?? ""}`}>
        {opts?.showQuartz !== false && (
          <p>
            {i18n(cfg.locale).components.footer.createdWith}{" "}
            <a href="https://quartz.jzhao.xyz/">Quartz v{version}</a> © {year}
          </p>
        )}
        <ul class="footer-credits">
          {credits.map(({ text, href, prefix, suffix }) => (
            <li>
              {prefix}
              {href ? <a href={href}>{text}</a> : text}
              {suffix}
            </li>
          ))}
        </ul>
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
