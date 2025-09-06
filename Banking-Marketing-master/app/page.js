import Header from '@/components/Header'
import Hero from '@/components/Hero'
import Services from '@/components/Services'
import Footer from '@/components/Footer'
import ChatbotButton from '@/components/ChatbotButton'
import Chatbot from '@/components/Chatbot'
import LoanCalculator from '@/components/loancalculator'
export default function Home() {
  return (
    <main className="min-h-screen bg-[#000049]">
      <Header />
      <Hero />
      <LoanCalculator />
      <Services />
      <ChatbotButton />
      <Footer />
    </main>
  )
}


