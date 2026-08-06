import type { Metadata } from 'next';
import './globals.css';
import ThemeProvider from '@/components/theme/ThemeProvider';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';

export const metadata: Metadata = {
  title: 'AI-Enabled Handwriting Verification System',
  description:
    'Decision Support System for Examination Integrity: Handwriting Verification Using Siamese Networks by Sanusi Shafii',
  keywords: [
    'handwriting verification',
    'siamese network',
    'AI',
    'examination integrity',
    'deep learning',
    'Sanusi Shafii',
  ],
  authors: [{ name: 'Sanusi Shafii' }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased min-h-screen flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange={false}
        >
          <div className="flex flex-col min-h-screen">
            <Header />
            <main className="flex-1 flex flex-col">{children}</main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
