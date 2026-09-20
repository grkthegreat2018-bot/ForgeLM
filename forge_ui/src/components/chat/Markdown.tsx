// Markdown renderer for model output — marked → DOMPurify → .chat-body.
// Styles live in index.css (.chat-body …).

import { clsx } from 'clsx'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { useMemo } from 'react'

marked.setOptions({ breaks: true, gfm: true })

export function Markdown({ text, className }: {
  text: string
  className?: string
}) {
  const html = useMemo(() => {
    const raw = marked.parse(text, { async: false }) as string
    return DOMPurify.sanitize(raw)
  }, [text])
  return (
    <div
      className={clsx('chat-body text-[13.5px] leading-relaxed break-words min-w-0', className)}
      dangerouslySetInnerHTML={{ __html: html }} />
  )
}
