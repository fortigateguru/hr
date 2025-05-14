import streamlit as st

# Set page configuration for Hebrew RTL and title
st.set_page_config(page_title="תיקי בית ספר - המלאי מלא", layout="centered")

# HTML content for the landing page
html_content = """
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-b from-blue-200 to-blue-400">
    <div class="bg-white rounded-lg shadow-xl p-8 max-w-md w-full text-center">
      <h1 class="text-3xl font-bold text-blue-800 mb-4">מלאי תיקי בית ספר מלא!</h1>
      <p class="text-lg text-gray-700 mb-6">
        תודה על התעניינותך! מלאי תיקי בית הספר שלנו מלא כרגע. ניצור איתך קשר בהקדם כאשר יהיו תיקים זמינים.
      </p>
      <img 
        src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" 
        alt="תיק בית ספר" 
        class="w-full h-64 object-cover rounded-md mb-6"
      >
      <p class="text-sm text-gray-500">הישארו מעודכנים! נעדכן אתכם בקרוב.</p>
    </div>
  </div>
</body>
</html>
"""

# Render the HTML content in Streamlit
st.components.v1.html(html_content, height=800, scrolling=True)
