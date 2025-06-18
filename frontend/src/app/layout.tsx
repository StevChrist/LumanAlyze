import { Montserrat, Lora } from 'next/font/google'
import './globals.css'

const montserrat = Montserrat({
  subsets: ['latin'],
  weight: ['700'],
  display: 'swap',
})

const lora = Lora({
  subsets: ['latin'],
  weight: ['400', '700'],
  display: 'swap',
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Lora:wght@400;700&display=swap" rel="stylesheet" />
      </head>
      <body className={`${montserrat.className} ${lora.className}`}>
        {children}
      </body>
    </html>
  )
}
