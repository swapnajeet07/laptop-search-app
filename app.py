import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from groq import Groq
import qrcode
from io import BytesIO
import smtplib
from email.message import EmailMessage
import random
import json
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="💻 AI Laptop Finder", page_icon="💻", layout="wide")

# --- Load Data ---
df = pd.read_csv("laptops.csv")

# --- Load Sentence Transformer Model ---
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# --- FAISS Indexing ---
@st.cache_resource
def create_faiss_index():
    descriptions = df['Model'].astype(str).tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = create_faiss_index()

# --- Search Logic ---
def semantic_search(query, top_k=10):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return df.iloc[indices[0]]

# --- Price Extraction from Query ---
def extract_price_filter(query):
    query = query.lower()
    price_min, price_max = None, None
    if m := re.search(r'under\s?(\d+)', query):
        price_max = int(m.group(1))
    elif m := re.search(r'above|over\s?(\d+)', query):
        price_min = int(m.group(1))
    elif m := re.search(r'between\s?(\d+)\s?(?:and|-|to)\s?(\d+)', query):
        price_min, price_max = int(m.group(1)), int(m.group(2))
    return price_min, price_max

# --- Session State ---
if "cart" not in st.session_state:
    st.session_state.cart = []
if "checkout_mode" not in st.session_state:
    st.session_state.checkout_mode = False
if "show_qr" not in st.session_state:
    st.session_state.show_qr = False
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False
if "otp_verified" not in st.session_state:
    st.session_state.otp_verified = False
if "generated_otp" not in st.session_state:
    st.session_state.generated_otp = None

# --- Sidebar Filters ---
st.sidebar.header("🔧 Refine Your Search")
brand = st.sidebar.multiselect("🏷 Brand", options=df['brand'].dropna().unique())
processor = st.sidebar.multiselect("🧠 Processor", options=df['processor_brand'].dropna().unique())
ram = st.sidebar.multiselect("💾 RAM", options=df['ram_memory'].dropna().unique())
storage = st.sidebar.multiselect("📦 Storage", options=df['storage'].dropna().unique() if 'storage' in df.columns else [])
price_min, price_max = st.sidebar.slider("💰 Price Range", int(df['Price'].min()), int(df['Price'].max()), (int(df['Price'].min()), int(df['Price'].max())), step=1000)
rating_filter = st.sidebar.selectbox("⭐ Minimum Rating", [0, 1, 2, 3, 4, 5], index=3)

# --- Sidebar Cart ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛒 Your Cart")
cart_items = st.session_state.cart
if cart_items:
    total = 0
    for item in cart_items:
        st.sidebar.markdown(f"- **{item['Model']}** - ₹{item['Price']}")
        total += item['Price']
    st.sidebar.markdown(f"**Total: ₹{total}**")
    if st.sidebar.button("🧾 Proceed to Checkout"):
        st.session_state.checkout_mode = True
else:
    st.sidebar.markdown("_Cart is empty_")

# --- App Title ---
st.markdown("""
<h1 style='text-align: center; color: #4A90E2;'>💻 AI Laptop Recommendation System</h1>
<p style='text-align: center;'>Smart search or filter to find the perfect laptop for your needs. Chat with the assistant for help!</p>
""", unsafe_allow_html=True)

# --- QR Payment Generator ---
def generate_qr(text):
    qr = qrcode.make(text)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue()

# --- Send Confirmation Email ---
def send_confirmation_email(to_email, name, items, address):
    EMAIL_USER = st.secrets.get("EMAIL_USER")
    EMAIL_PASS = st.secrets.get("EMAIL_PASS")

    msg = EmailMessage()
    msg['Subject'] = "Order Confirmation - AI Laptop Finder"
    msg['From'] = EMAIL_USER
    msg['To'] = to_email

    total = sum([item['Price'] for item in items])
    item_lines = "\n".join([f"- {item['Model']} (₹{item['Price']})" for item in items])

    msg.set_content(f"""
Hi {name},

Thank you for your purchase! 🎉

Order Summary:
{item_lines}

Total Amount: ₹{total}

Shipping Address:
{address}

Your order will be processed shortly.

Thanks,
AI Laptop Finder Team
    
    """)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

# --- OTP Email ---
def send_otp_email(to_email, otp_code):
    EMAIL_USER = st.secrets.get("EMAIL_USER")
    EMAIL_PASS = st.secrets.get("EMAIL_PASS")

    msg = EmailMessage()
    msg['Subject'] = "Your OTP Code - AI Laptop Finder"
    msg['From'] = EMAIL_USER
    msg['To'] = to_email

    msg.set_content(f"""
Hi,

Your OTP code for completing your laptop purchase is: {otp_code}

If you did not request this, please ignore this email.

Thanks,
AI Laptop Finder Team
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send OTP email: {e}")
        return False

# --- Save order to file ---
def save_order(name, email, address, items):
    order = {
        "name": name,
        "email": email,
        "address": address,
        "items": [{"Model": item["Model"], "Price": item["Price"]} for item in items]
    }
    try:
        with open("orders.json", "a") as f:
            f.write(json.dumps(order) + "\n")
    except Exception as e:
        st.error(f"Failed to save order: {e}")

# --- Generate PDF Invoice ---
def generate_pdf_invoice(name, address, items, total):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Laptop Finder Invoice", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Customer: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Shipping Address: {address}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Order Details:", ln=True)
    for item in items:
        pdf.cell(200, 10, txt=f"- {item['Model']} : ₹{item['Price']}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Amount: ₹{total}", ln=True)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

# --- Mode Toggle ---
mode = st.radio("Choose a search mode:", ["🔍 Smart Search", "🎛 Manual Filter"], horizontal=True)

# --- Checkout Flow ---
if st.session_state.checkout_mode:
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5vw;
        padding-right: 5vw;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧾 Secure Checkout")
    if st.button("🔙 Back to Search"):
        st.session_state.checkout_mode = False
        st.session_state.otp_sent = False
        st.session_state.otp_verified = False
        st.experimental_rerun()

    # Initialize OTP state
    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "generated_otp" not in st.session_state:
        st.session_state.generated_otp = None

    # Show checkout form if OTP not yet sent
    if not st.session_state.otp_sent:
        with st.form("checkout_form", clear_on_submit=True):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            address = st.text_area("Shipping Address")
            payment = st.selectbox("Payment Method", ["UPI QR Payment", "Cash on Delivery"])
            submitted = st.form_submit_button("💳 Confirm Purchase")

            if submitted:
                if not name or not email or not address:
                    st.error("Please fill all fields.")
                else:
                    otp_code = random.randint(100000, 999999)
                    sent = send_otp_email(email, otp_code)
                    if sent:
                        st.success("OTP sent to your email. Please check your inbox.")
                        st.session_state.generated_otp = str(otp_code)
                        st.session_state.otp_sent = True
                        st.session_state.form_data = {"name": name, "email": email, "address": address, "payment": payment}
                    else:
                        st.error("Failed to send OTP. Try again.")

    # OTP Input and verification
    elif not st.session_state.otp_verified:
        otp_input = st.text_input("Enter the OTP sent to your email:")
        if st.button("Verify OTP"):
            if otp_input == st.session_state.generated_otp:
                st.success("OTP verified successfully!")
                st.session_state.otp_verified = True
            else:
                st.error("Incorrect OTP. Please try again.")

    # Payment and confirmation
    else:
        name = st.session_state.form_data["name"]
        email = st.session_state.form_data["email"]
        address = st.session_state.form_data["address"]
        payment = st.session_state.form_data["payment"]

        total = sum([item['Price'] for item in st.session_state.cart])

        if payment == "UPI QR Payment":
            qr_data = f"upi://pay?pa=merchant@upi&pn=AI Laptop Store&am={total}&cu=INR"
            st.info("Scan the QR code below to pay via UPI:")
            st.image(generate_qr(qr_data))
            if st.button("✅ I Have Paid"):
                save_order(name, email, address, st.session_state.cart)
                pdf_bytes = generate_pdf_invoice(name, address, st.session_state.cart, total)
                email_sent = send_confirmation_email(email, name, st.session_state.cart, address)
                if email_sent:
                    st.success(f"✅ Order placed! A confirmation email was sent to {email}.")
                    st.download_button("📄 Download Invoice (PDF)", data=pdf_bytes, file_name="invoice.pdf")
                    st.balloons()
                    st.session_state.cart = []
                    st.session_state.checkout_mode = False
                    st.session_state.otp_sent = False
                    st.session_state.otp_verified = False
                    st.experimental_rerun()

        else:  # Cash on Delivery
            if st.button("💳 Confirm Order (Cash on Delivery)"):
                save_order(name, email, address, st.session_state.cart)
                pdf_bytes = generate_pdf_invoice(name, address, st.session_state.cart, total)
                email_sent = send_confirmation_email(email, name, st.session_state.cart, address)
                if email_sent:
                    st.success(f"✅ Order placed! A confirmation email was sent to {email}.")
                    st.download_button("📄 Download Invoice (PDF)", data=pdf_bytes, file_name="invoice.pdf")
                    st.balloons()
                    st.session_state.cart = []
                    st.session_state.checkout_mode = False
                    st.session_state.otp_sent = False
                    st.session_state.otp_verified = False
                    st.experimental_rerun()

else:
    results = pd.DataFrame()
    if mode == "🔍 Smart Search":
        query = st.text_input("Ask for a laptop:", placeholder="e.g. I need a gaming laptop under 70000")
        if query:
            with st.spinner("Searching smartly..."):
                min_q, max_q = extract_price_filter(query)
                results = semantic_search(query)
                results = results[(results['Price'] >= (min_q or price_min)) & (results['Price'] <= (max_q or price_max))]
                if 'Rating' in df.columns:
                    results = results[results['Rating'] >= rating_filter]
    else:
        results = df[(df['Price'] >= price_min) & (df['Price'] <= price_max)]
        if brand:
            results = results[results['brand'].isin(brand)]
        if processor:
            results = results[results['processor_brand'].isin(processor)]
        if ram:
            results = results[results['ram_memory'].isin(ram)]
        if storage and 'storage' in df.columns:
            results = results[results['storage'].isin(storage)]
        if 'Rating' in df.columns:
            results = results[results['Rating'] >= rating_filter]

    if not results.empty:
        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"""
                <div style='background-color:#e6f0ff;padding:15px;margin:10px 0;border-radius:12px;box-shadow:2px 2px 6px rgba(0,0,0,0.1);'>
                    <h4 style='color:#0a58ca;'>{row['Model']} - ₹{row['Price']}</h4>
                    <p style='color:#000;'>Brand: <b>{row['brand']}</b> | Processor: <b>{row['processor_brand']}</b> | RAM: <b>{row['ram_memory']}</b> | Rating: <b>{row.get('Rating', 'N/A')}</b></p>
                </div>
                """, unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🛒 Add to Cart: {row['Model']}", key=f"add_{row['Model']}"):
                        st.session_state.cart.append(row)
                        st.success("Added to cart!")
                with col2:
                    if st.button(f"⚡ Buy Now: {row['Model']}", key=f"buy_{row['Model']}"):
                        st.session_state.cart = [row]
                        st.session_state.checkout_mode = True
                        st.experimental_rerun()
    else:
        st.warning("No laptops found matching your criteria.")

# --- Chatbot Interface ---
st.divider()
st.markdown("### 🤖 AI Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_msg = st.chat_input("Chat with your laptop assistant...")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.spinner("🤖 Thinking..."):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": "You are a helpful assistant that recommends laptops."}] + st.session_state.chat_history
            )
            bot_reply = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        except Exception as e:
            bot_reply = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

# --- Display Chat ---
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg['role'] == "user" else "assistant"):
        st.markdown(msg['content'])

