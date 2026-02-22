import streamlit as st
import base64

st.set_page_config(layout="wide")

# --------------------------------------------------
# LOAD BACKGROUND IMAGE
# --------------------------------------------------

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_image("assets/apsrtc_bg.png")
people_image = get_base64_image("assets/ap_people.avif")  # add AP people image
logo_image = get_base64_image("assets/apsrtc_logo.png")  # add APSRTC logo

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(f"""
<style>

/* Full Background Map */
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(rgba(255,255,255,0.55),
                                      rgba(255,255,255,0.70)),
                      url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* HERO SECTION CENTERED */
.main-container {{
    height: 85vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}}

/* Hero Text */
.hero-title {{
    font-size: 64px;
    font-weight: 800;
    color: #15803d;
}}

.hero-sub {{
    font-size: 22px;
    color: #1f2937;
    margin-top: 20px;
}}

/* Smooth Transition to Grey */
.stats-section {{
    background: linear-gradient(to bottom,
                rgba(255,255,255,0) 0%,
                #f3f4f6 25%);
    padding-top: 120px;
    padding-bottom: 120px;
}}

/* Card Styling */
.stat-card {{
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    text-align: center;
    transition: 0.3s ease;
}}

.stat-card:hover {{
    transform: translateY(-8px);
}}

.stat-number {{
    font-size: 40px;
    font-weight: 700;
    color: #15803d;
}}

.stat-label {{
    font-size: 18px;
    margin-top: 10px;
    color: #374151;
}}

/* ABOUT SECTION WITH IMAGE BACKGROUND */
.about-section {{
    height: 80vh;
    background-image: linear-gradient(rgba(0,0,0,0.6),
                                      rgba(0,0,0,0.6)),
                      url("data:image/png;base64,{people_image}");
    background-size: cover;
    background-position: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: white;
    text-align: center;
    padding: 0 80px;
}}

.about-title {{
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 25px;
}}

.about-text {{
    font-size: 20px;
    max-width: 800px;
    line-height: 1.6;
}}

.logo-img {{
    width: 120px;
    margin-bottom: 25px;
}}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="main-container">

<div class="hero-title">
APSRTC Serving Since 70+ Years
</div>

<div class="hero-sub">
Connecting Villages. Empowering Cities. Driving Andhra Pradesh Forward.
</div>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SCROLL TRANSITION SECTION
# --------------------------------------------------

st.markdown('<div class="stats-section">', unsafe_allow_html=True)

st.markdown("### Andhra Pradesh State Road Transport Corporation — At a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">11,000+</div>
        <div class="stat-label">Buses Operating Daily</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">14,000+</div>
        <div class="stat-label">Villages Connected</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">39+ Lakhs</div>
        <div class="stat-label">Passengers Per Day</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">120+</div>
        <div class="stat-label">Bus Depots</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT APSRTC SECTION
# --------------------------------------------------

st.markdown(f"""
<div class="about-section">

<img src="data:image/png;base64,{logo_image}" class="logo-img">

<div class="about-title">About APSRTC</div>

<div class="about-text">
Established in 1958, APSRTC has evolved into one of India’s largest public transport networks.<br><br>
It connects thousands of villages and cities, enabling mobility, economic growth, and social development.<br><br>
For over seven decades, APSRTC has been the backbone of Andhra Pradesh’s transportation system.
</div>

</div>
""", unsafe_allow_html=True)