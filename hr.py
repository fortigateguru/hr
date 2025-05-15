import streamlit as st
import sqlite3
from datetime import datetime

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('visit_counter.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS visits 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Increment visit count
def increment_visit():
    conn = sqlite3.connect('visit_counter.db')
    c = conn.cursor()
    c.execute("INSERT INTO visits (timestamp) VALUES (?)", (datetime.now().isoformat(),))
    conn.commit()
    conn.close()

# Get total visits
def get_visit_count():
    conn = sqlite3.connect('visit_counter.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM visits")
    count = c.fetchone()[0]
    conn.close()
    return count

# Initialize database
init_db()

# Increment visit counter on page load
increment_visit()

# Set page configuration
st.set_page_config(page_title="תיקי בית ספר - המלאי מלא", layout="centered")

# HTML content
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

# Render HTML
st.components.v1.html(html_content, height=800, scrolling=True)

# Display visit count (optional, for admin/testing)
if st.checkbox("הצג ספירת מבקרים (למנהלים)"):
    st.write(f"מספר המבקרים הכולל: {get_visit_count()}")
