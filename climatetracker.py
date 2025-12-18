
import os
import io
import json
import time
import math
import base64
import requests
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components 

# --- ADDED INI IMPORTS ---
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError
# --- END GEMINI IMPORTS ---

# --- Forecasting Imports with graceful fallbacks ---
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    # Suppress warnings from pmdarima's auto_arima during import for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from pmdarima import auto_arima
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# -------------------------
# Login Credentials (MODIFIED)
# -------------------------
VALID_USER = "aditya01"
VALID_PASS = "vermasingh01"

# --- ADDED GEMINI API KEY ---
# NOTE: In a real app, this should be an environment variable (os.environ.get("GEMINI_API_KEY"))
GEMINI_API_KEY="AIzaSyCW-sKyNFFkWz4hiysMWJR6Fyj342-GhkY" 
# --- END GEMINI API KEY ---

# SustainifyAI — Sustainability & Climate Change Tracker (All-in-One Streamlit App)
# ---------------------------------------------------------------------------------

st.set_page_config(
    page_title="SustainifyAI — Sustainability and Climate Tracker",
    page_icon="🌍", # Changed icon to revolving globe
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Gemini API Initialization (STABLE CHATBOT SETUP)
# -------------------------

SYSTEM_PROMPT = """
You are SustainifyAI's virtual assistant, an expert in sustainability, climate, and the features of the Streamlit dashboard you are embedded in.
Your primary role is to answer questions about:
1.  The dashboard's purpose, features, and various tabs (Overview, Air Quality, Trends, Forecasts, Impact Story, Sustainability Score, Carbon, Green Infra, Swachh, Waste Management, About).
2.  Technical metrics like PM2.5, CO2e, MAE, MAPE, BOD, TPD (Tons Per Day), and the 4R waste model.
3.  Interpreting the data and plots (e.g., Anomaly Tracker, Correlation Matrix, Forecasting).

Keep your answers concise, informative, and friendly. Use relevant emojis and Markdown.
If the question is a greeting or thank you, respond appropriately.
If the question is completely outside the scope of the dashboard (e.g., 'tell me a joke'), gently steer the user back to the dashboard topics.
"""

@st.cache_resource
def initialize_gemini_client():
    """Initializes the Gemini client and the chat session using @st.cache_resource."""
    if not GEMINI_API_KEY:
        return None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Initialize a persistent chat session for history management
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        return client, chat
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client/Chat: {e}")
        return None, None

# Global client and chat object (Cached)
GEMINI_CLIENT, GEMINI_CHAT = initialize_gemini_client()

# ------------------------------ NEW BRAND HERO BANNER FUNCTIONS ------------------------------

def inject_brand_styles() -> None:
    """
    Inject global CSS for the animated hero banner.
    Why: Creates premium look with no dependencies; honors reduced-motion.
    """
    if st.session_state.get("_brand_css_injected"):
        return
    st.session_state["_brand_css_injected"] = True

    st.markdown(
        """
<style>
:root{
  --brand-bg: linear-gradient(135deg,#0bbbc4 0%,#0a7f87 50%,#0b5562 100%);
  --brand-glow: #7cf5ff;
  --brand-ink: #063f42;
  --brand-ink-contrast: #eaffff;
  --brand-card: rgba(255,255,255,.15);
  --brand-card-dark: rgba(8,18,22,.55);
}

/* Wrapper expands to full width */
.brand-hero-wrap{ margin: 8px 0 22px 0; }

/* Glass card with animated gradient border + inner light */
.brand-hero{
  position: relative;
  padding: clamp(18px, 1.6vw + 8px, 28px) clamp(22px, 2.8vw + 12px, 44px);
  border-radius: 22px;
  background: var(--brand-card);
  backdrop-filter: blur(10px) saturate(120%);
  -webkit-backdrop-filter: blur(10px) saturate(120%);
  border: 1px solid rgba(255,255,255,.25);
  overflow: hidden;
  box-shadow:
    0 12px 40px rgba(3, 150, 166, .18),
    inset 0 1px 0 rgba(255,255,255,.35);
}

/* Animated border glow */
.brand-hero::before{
  content:"";
  position:absolute; inset:-2px;
  border-radius: 24px;
  background:
    conic-gradient(from 0deg,
      rgba(124,245,255,.0) 0deg,
      rgba(124,245,255,.9) 120deg,
      rgba(124,245,255,.0) 240deg,
      rgba(124,245,255,.9) 360deg);
  filter: blur(10px);
  animation: spin 12s linear infinite;
  z-index:0;
}

/* Soft gradient sheet behind content */
.brand-hero::after{
  content:"";
  position:absolute; inset:0;
  background:
    radial-gradient(120% 80% at 50% -10%, rgba(255,255,255,.45), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,0));
  z-index:0;
}

/* Floating blobs */
.blobs, .particles{ position:absolute; inset:0; pointer-events:none; }

/* Blobs */
.blobs::before, .blobs::after{
  content:"";
  position:absolute;
  width:40%; height:120%;
  left:-10%; top:-20%;
  background: radial-gradient(50% 50% at 50% 50%, rgba(124,245,255,.35), transparent 60%);
  filter: blur(40px);
  animation: floatY 14s ease-in-out infinite;
}
.blobs::after{
  width:45%; height:120%;
  right:-5%; left:auto; top:-25%;
  background: radial-gradient(50% 50% at 50% 50%, rgba(0, 180, 200, .25), transparent 60%);
  animation: floatY 18s ease-in-out infinite reverse;
}

/* Tiny drifting particles */
.particles{
  background:
    radial-gradient(4px 4px at 20% 30%, rgba(255,255,255,.35), transparent 60%),
    radial-gradient(3px 3px at 70% 60%, rgba(255,255,255,.30), transparent 60%),
    radial-gradient(3px 3px at 40% 80%, rgba(255,255,255,.25), transparent 60%),
    radial-gradient(2px 2px at 85% 35%, rgba(255,255,255,.25), transparent 60%);
  animation: drift 30s linear infinite;
  opacity:.6;
}

/* Content layout */
.brand-row{ position:relative; display:flex; flex-wrap:wrap; align-items:center; gap:18px; z-index:1; }
.brand-icon{
  flex:0 0 auto;
  width: clamp(46px, 4.5vw + 18px, 72px);
  height: clamp(46px, 4.5vw + 18px, 72px);
  border-radius: 18px;
  background: var(--brand-bg);
  box-shadow: 0 8px 28px rgba(0, 150, 166, .35), inset 0 1px 0 rgba(255,255,255,.35);
  display:grid; place-items:center;
  color: var(--brand-ink-contrast);
}

/* Title + subtitle */
.brand-text{ flex:1 1 auto; min-width: 280px; color: var(--brand-ink-contrast); }
.brand-title{
  margin:0;
  font-weight: 900;
  letter-spacing:.3px;
  font-size: clamp(1.25rem, 1.6vw + 1.1rem, 2.2rem);
  text-shadow: 0 2px 8px rgba(0,0,0,.25);
}
.brand-sub{
  margin:.35rem 0 0 0;
  font-weight: 600;
  opacity:.95;
  font-size: clamp(.9rem, .5vw + .7rem, 1.05rem);
}

/* Pills/badges under subtitle */
.brand-badges{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.6rem; }
.brand-badge{
  font-weight:800;
  font-size:.82rem;
  padding:.35rem .7rem;
  border-radius:999px;
  color:#05464b;
  background: rgba(255,255,255,.75);
  box-shadow: 0 6px 18px rgba(255,255,255,.25), inset 0 1px 0 rgba(255,255,255,.8);
}

/* CTA buttons */
.brand-ctas{ display:flex; gap:.6rem; flex-wrap:wrap; margin-left:auto; }
.brand-btn{
  border:0; cursor:pointer;
  padding:.65rem 1rem;
  border-radius:14px;
  font-weight:900;
  color:#043f40;
  background: linear-gradient(180deg, #ddfeff, #bfffff);
  box-shadow: 0 10px 24px rgba(0,150,166,.25);
  transition: transform .1s ease, box-shadow .2s ease, filter .2s ease;
}
.brand-btn:hover{ transform: translateY(-1px); box-shadow: 0 14px 32px rgba(0,150,166,.32); }
.brand-btn.alt{
  background: linear-gradient(180deg, #ffffff, #eaf7f7);
  color:#065b60;
}

/* Shine hover */
.brand-hero:hover{ filter: drop-shadow(0 22px 40px rgba(0,150,166,.18)); }
.brand-hero:hover .brand-title{ text-shadow: 0 3px 12px rgba(0,0,0,.32); }

/* Keyframes */
@keyframes spin{ to{ transform: rotate(360deg);} }
@keyframes floatY{ 0%,100%{ transform: translateY(0px);} 50%{ transform: translateY(18px);} }
@keyframes drift{ 0%{ background-position: 0 0, 0 0, 0 0, 0 0;}
                  100%{ background-position: 200% 0, -150% 0, 120% 0, -180% 0;} }

/* Dark mode */
@media (prefers-color-scheme: dark){
  .brand-hero{ background: var(--brand-card-dark); border-color: rgba(255,255,255,.08); }
  .brand-badge{ background: rgba(0,255,255,.15); color:#caffff; }
  .brand-btn{ color:#022f31; }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce){
  .brand-hero::before, .blobs::before, .blobs::after, .particles{ animation:none !important; }
}
/* --- NEW CSS FOR ANIMATED CHATBOT HEADER --- */
.chatbot-header-wrap {
    display: flex;
    align-items: center;
    gap: 10px; /* Space between icon and text */
    padding: 10px 15px;
    background: linear-gradient(90deg, #e0f2f7, #c1e4f2); /* Light blue gradient */
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    animation: glow-border 1.5s infinite alternate; /* Glowing border */
}

.chatbot-icon {
    font-size: 2.2rem; /* Larger emoji */
    animation: head-shake 2s infinite ease-in-out; /* Shaking animation */
    display: inline-block;
}

.chatbot-title {
    font-size: 1.5rem; /* Main title size */
    font-weight: bold;
    color: #004d40; /* Darker teal text */
    margin: 0;
}

.chatbot-prompt {
    font-size: 1rem; /* Smaller prompt text */
    color: #333;
    margin: 0;
}

.flicker-hand {
    animation: flicker-wave 1s infinite alternate; /* Flickering hand animation */
    display: inline-block;
    font-size: 1.2rem;
    margin-left: 5px;
}

/* Keyframe Animations */
@keyframes head-shake {
    0%, 100% { transform: translateX(0); }
    10% { transform: translateX(-5px) rotate(3deg); }
    20% { transform: translateX(5px) rotate(-3deg); }
    30% { transform: translateX(-5px) rotate(3deg); }
    40% { transform: translateX(5px) rotate(-3deg); }
    50% { transform: translateX(-2px) rotate(1deg); }
    60% { transform: translateX(2px) rotate(-1deg); }
    70%, 100% { transform: translateX(0); }
}

@keyframes glow-border {
    0% { box-shadow: 0 0 5px rgba(0, 123, 255, 0.4); }
    100% { box-shadow: 0 0 15px rgba(0, 123, 255, 0.8), 0 0 20px rgba(0, 123, 255, 0.2); }
}

@keyframes flicker-wave {
    0%, 100% { opacity: 1; transform: rotate(0deg); }
    25% { opacity: 0.8; transform: rotate(10deg); }
    50% { opacity: 1; transform: rotate(-5deg); }
    75% { opacity: 0.8; transform: rotate(10deg); }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def brand_banner(
    title: str,
    subtitle: str,
    badges: Optional[Iterable[str]] = None,
    ctas: Optional[Iterable[Tuple[str, str]]] = None,
    icon: str = "🌿",
) -> None:
    """
    Render the animated brand banner.
    Why: Replaces bland header with premium, theme-aligned hero.
    Args:
        badges: strings shown as feature chips.
        ctas: list of (label, variant) where variant in {"primary","alt"}.
    """
    inject_brand_styles()

    badges = list(badges or [])
    ctas = list(ctas or [("Launch App", "primary"), ("Learn More", "alt")])

    # HTML structure
    html = f"""
<div class="brand-hero-wrap">
  <div class="brand-hero">
    <div class="blobs"></div>
    <div class="particles"></div>
    <div class="brand-row">
      <div class="brand-icon">{icon}</div>
      <div class="brand-text">
        <h1 class="brand-title">{title}</h1>
        <p class="brand-sub">{subtitle}</p>
        {'' if not badges else '<div class="brand-badges">' + ''.join(f'<span class="brand-badge">{b}</span>' for b in badges) + '</div>'}
      </div>
      <div class="brand-ctas">
        {'' .join(f'<button class="brand-btn{" alt" if v=="alt" else ""}">{t}</button>' for t,v in ctas)}
      </div>
    </div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
   

# ------------------------------ PREMIUM MOTION UI CSS OVERHAUL (COMPLETE) ------------------------------
# Removed old .hero, .custom-button-like, .droplet, .hero-title-button, .hero-sub-button CSS
css_block = """
/* *** MODIFICATION: Changing font to a stylized Serif look and updating icon styles *** */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Bebas+Neue&family=Signika:wght@300..700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

:root {
    /* Light & Bright Base */
    --bg: #f8f9fa; /* Very Light Gray/White */ 
    --card: #ffffff; /* Pure White Card */
    --muted: #6c757d; /* Dark Gray Muted Text */
    /* PRIMARY VIBRANT FOCUS - MODIFIED TO GREEN/TEAL FOCUS */
    --brand: #38a3a5; /* Primary Teal (Clean Green) */
    --brand2:#ffc107; /* Amber/Yellow Accent */
    --brand-glow: 0 0 10px rgba(56, 163, 165, 0.4), 0 0 20px rgba(255, 193, 7, 0.2); 
}

/* --- CORE BACKGROUND & LAYOUT (Light) --- */
.stApp { 
    background: var(--bg) !important; 
    font-family: 'Signika', sans-serif; /* New Default Font */
}
 
/* Subtle background animation */
@keyframes subtle-drift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -90;
    /* Subtle Moving Gradient Background */
    background: linear-gradient(135deg, rgba(56, 163, 165, 0.05), rgba(0, 123, 255, 0.05), rgba(255, 193, 7, 0.05));
    background-size: 400% 400%;
    animation: subtle-drift 120s ease infinite;
    pointer-events: none;
    opacity: 1; 
}
 
.main > div {
    background-color: transparent !important; /* Let the body bg show */
    backdrop-filter: none;
}
.stSidebar > div:first-child {
    background: linear-gradient(180deg, #ffffff 0%, #f1f7fd 100%) !important; /* Light gradient */
    border-right: 2px solid var(--brand); /* Clean Green/Teal border */
    box-shadow: 6px 0 30px rgba(0, 0, 0, 0.1), inset -3px 0 10px rgba(56, 163, 165, 0.05);
}

.block-container { padding-top: 1rem; }
 
/* --- NEWS TICKER STYLING (Blue/White) --- */
@keyframes marquee {
    0% 	{ transform: translate(100%, 0); }
    100% { transform: translate(-100%, 0); }
}
.news-pipe-container {
    overflow: hidden;
    width: 100%;
    height: 40px;
    background: #f1f7fd; /* Light blue background */
    border: 1px solid var(--brand); /* Teal border */
    border-radius: 8px;
    margin-bottom: 20px;
    padding: 5px 0;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    position: relative;
}
.news-pipe-content {
    white-space: nowrap;
    color: var(--brand); /* Primary Teal for Alerts */
    font-size: 1.1rem;
    font-weight: 600;
    padding-top: 2px;
    animation: marquee 25s linear infinite;
    text-shadow: 0 0 5px rgba(56, 163, 165, 0.2);
}
 
		/* =========================
			UNIFORM TEAL HEADINGS
			========================= */

		/* All major headings (H1, H2) used for section titles */
		h1, h2 {
			color: #008080 !important; 	 	 	 /* Teal base color */
			text-shadow: 0 0 8px rgba(0,128,128,0.4),
						 0 0 16px rgba(0,128,128,0.25);
			font-weight: 800 !important;
			letter-spacing: 0.5px;
			transition: all 0.4s ease-in-out;
		}

		/* Bullet or dot prefix for "• Heading" style */
		h1::before, h2::before {
			content: "• ";
			color: #008080;
			text-shadow: 0 0 10px rgba(0,128,128,0.6);
		}

		/* Sub-headings and chart titles (H3, H4) */
		h3, h4 {
			color: #008080 !important;
			font-weight: 700 !important;
			text-shadow: 0 0 8px rgba(0,128,128,0.3);
		}

		/* Section title classes sometimes used in markdown */
		.section-title, .highlight-title, .stMarkdown h2 span {
			color: #008080 !important;
			text-shadow: 0 0 12px rgba(0,128,128,0.35);
			font-weight: 800;
		}

		/* Tabs underline highlight */
		[data-baseweb="tab-list"] [aria-selected="true"] {
			border-bottom: 3px solid #008080 !important;
			color: #008080 !important;
		}

		/* Expander headings (used in collapsible sections) */
		[data-testid="stExpander"] > div:first-child {
			color: #008080 !important;
			text-shadow: 0 0 8px rgba(0,128,128,0.35);
		}

		/* Hover glow for interactivity */
		h1:hover, h2:hover, h3:hover {
			text-shadow: 0 0 16px rgba(0,128,128,0.6),
						 0 0 30px rgba(0,128,128,0.4);
		}

		p,li,span,div, label, .stMarkdown { 
			color:#343a40 !important; 
			font-size: 1.0rem !important;
		}

/* --- SIDEBAR SPECIFIC STYLING --- */
.stSidebar h1 { 
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #008080 !important; /* Sidebar heading unified to teal */
    text-shadow: 0 0 10px rgba(0,128,128,0.35); 
    font-family: 'Bebas Neue', sans-serif !important; 
    letter-spacing: 2px;
}
/* --- NEW/MODIFIED: Large Sidebar Icon Container --- */
.sidebar-icon-container {
    text-align: center;
    padding-top: 10px;
    padding-bottom: 5px;
}
.big-earth-icon {
    font-size: 3.5em; /* Bigger icon size */
    color: var(--brand); 
    text-shadow: 0 0 15px rgba(56, 163, 165, 0.8), 0 0 5px #007bff; /* Teal/Blue Glow */
    display: inline-block;
    animation: spin 6s linear infinite;
}
.big-tree-icon {
    font-size: 3.0em; /* Tree icon slightly smaller */
    color: green;
    margin-left: 10px;
    text-shadow: 0 0 10px #38a3a5;
}
 
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.rotating-globe {
    animation: spin 6s linear infinite;
    font-size: 1.5em;
    margin-right: 5px;
    color: var(--brand); 
    text-shadow: 0 0 12px rgba(56, 163, 165, 0.8); 
}

/* *** NEW: CSS for Rotating Windmill (For Max Wind KPI) *** */
@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced Animations and Effects */
.animated-card-good {
    transition: all 0.3s ease;
    border: 2px solid transparent;
    animation: good-glow 4s infinite;
}
 
.animated-card-moderate {
    transition: all 0.3s ease;
    border: 2px solid transparent;
    animation: moderate-glow 4s infinite;
}
 
.animated-card-critical {
    transition: all 0.3s ease;
    border: 2px solid transparent;
    animation: critical-glow 1.5s infinite;
}
 
@keyframes good-glow {
    0%, 100% { box-shadow: 0 0 10px rgba(56, 163, 165, 0.2); }
    50% { box-shadow: 0 0 20px rgba(56, 163, 165, 0.4); }
}
 
@keyframes moderate-glow {
    0%, 100% { box-shadow: 0 0 10px rgba(255, 193, 7, 0.2); }
    50% { box-shadow: 0 0 20px rgba(255, 193, 7, 0.4); }
}
 
@keyframes critical-glow {
    0%, 100% { 
        box-shadow: 0 0 15px rgba(230, 57, 70, 0.3);
        transform: scale(1);
    }
    50% { 
        box-shadow: 0 0 25px rgba(230, 57, 70, 0.6);
        transform: scale(1.02);
    }
}
 
.glowing-text {
    text-shadow: 0 0 10px rgba(var(--accent-rgb), 0.3);
    animation: text-pulse 2s infinite;
}
 
@keyframes text-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}
 
.sliding-text {
    animation: slide-in 0.5s ease-out;
}
 
@keyframes slide-in {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.animated-icon {
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}
 
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}
 
.gradient-border {
    position: relative;
    border: 2px solid var(--border-color);
    background-clip: padding-box;
    box-shadow: 0 0 15px rgba(var(--border-color), 0.3);
    transition: transform 0.3s ease;
}
 
.gradient-border:hover {
    transform: translateY(-2px);
}
 
.animated-increase {
    color: #e63946;
    font-size: 1.2em;
    animation: bounce-up 1s infinite;
}
 
.animated-decrease {
    color: #38a3a5;
    font-size: 1.2em;
    animation: bounce-down 1s infinite;
}
 
@keyframes bounce-up {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}
 
@keyframes bounce-down {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(5px); }
}
 
.rotating-windmill {
    animation: rotate 2s linear infinite;
    display: inline-block;
    margin-right: 5px;
}
 

/* Input Fields (Clean White/Blue Focus) */
.stSidebar [data-baseweb="input"], .stSidebar [data-baseweb="base-input"],
.stSidebar [data-baseweb="select"] > div:first-child,
.stSidebar .stDateInput > div:first-child > div {
    background-color: var(--card) !important; 
    border: 1px solid #ced4da !important; 
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    border-radius: 8px;
    padding: 8px 10px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.stSidebar [data-baseweb="input"]:focus-within, 
.stSidebar [data-baseweb="base-input"]:focus-within,
.stSidebar [data-baseweb="select"]:focus-within > div:first-child,
.stSidebar .stDateInput > div:first-child:focus-within > div {
    border-color: var(--brand) !important; 
    box-shadow: 0 0 0 2px rgba(56, 163, 165, 0.5);
}

/* Slider styling (Yellow/Orange focus) */
.stSidebar .stSlider [data-baseweb="slider"] {
    background-color: #e9ecef;
    height: 8px;
    border-radius: 4px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(2) {
    background-color: var(--brand2); 
    box-shadow: 0 0 8px rgba(255, 193, 7, 0.6);
}
.stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(3) {
    background-color: var(--card);
    border: 2px solid var(--brand2); 
    box-shadow: 0 0 15px rgba(255, 193, 7, 0.8), 0 0 5px rgba(0,0,0,0.2);
    width: 20px;
    height: 20px;
    top: -6px;
}

/* --- HERO HEADING (Custom Button Style with Water Droplets) --- */
/* REMOVED OLD HERO CSS BLOCKS: .hero, .custom-button-like, .hero-title-button, .hero-sub-button, .droplet */
 
/* --- GLASS MORPHISM KPI / CARDS (Header) - Blue/White Focus --- */
.glass-card-header { 
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
 
    border: 1px solid rgba(56, 163, 165, 0.3); 
    padding: 15px; 
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
    transition: all .35s ease; 
    position: relative; 
    color: #212529 !important; 
}
.glass-card-header:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 6px 20px rgba(56, 163, 165, 0.15); 
}
 
/* --- KPI / CARDS (Regular) - White/Teal Focus) --- */
.metric-card { 
    background: var(--card); 
    border:1px solid #e9ecef; 
    padding:20px; 
    border-radius:12px; 
    box-shadow:0 6px 15px rgba(0, 0, 0, 0.1); 
    transition: all .35s ease; 
    position: relative; 
    color: #212529 !important; 
    overflow: hidden; 
}
.metric-card:hover{ 
    transform: translateY(-4px) scale(1.01); 
    background: radial-gradient(800px 220px at 10% 10%, #f1f7fd 0%, #ffffff 70%); 
    box-shadow:0 12px 30px rgba(56, 163, 165, 0.2); 
    border-color:var(--brand); 
}
 
/* --- CUSTOM KPI VALUE STYLING (Primary Blue/Accent) --- */
.kpi-value {
    color: var(--brand) !important; /* Teal/Green KPI Value */
    font-weight: 700; 
    font-size: 2.0rem; 
    text-shadow: 0 0 5px rgba(56, 163, 165, 0.3); 
    margin-top: -10px;
}
.kpi-label {
    color: var(--muted) !important; 
    font-size: 0.9rem !important;
}

/* --- RANK MIRROR BUTTON (New Style) --- */
@keyframes shine {
    0% { background-position: -100% 0; }
    100% { background-position: 100% 0; }
}
.rank-mirror-button {
    display: block;
    padding: 10px 15px;
    background: #f8f9fa; /* Light Background */
    border: 2px solid var(--brand); /* Teal Border */
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    margin-top: 10px;
    overflow: hidden;
    position: relative;
    color: #212529 !important;
    transition: transform 0.3s ease;
}
.rank-mirror-button:hover {
    transform: scale(1.03);
}
.rank-mirror-button::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 200%;
    height: 100%;
    background: linear-gradient(120deg, transparent 30%, rgba(255, 255, 255, 0.5) 50%, transparent 70%);
    transition: opacity 0.3s ease;
    animation: shine 2s infinite linear; /* SHINING EFFECT */
    opacity: 0;
}
.rank-mirror-button:hover::after {
    opacity: 1;
}

/* *** NEW: Flicker and Glow for Swachh Star Crown *** */
@keyframes flicker-glow-zoom {
    0% { transform: scale(1.0) rotate(0deg); opacity: 0.8; text-shadow: 0 0 5px orange; }
    25% { transform: scale(1.1) rotate(5deg); opacity: 1; text-shadow: 0 0 15px gold, 0 0 20px var(--brand2); }
    50% { transform: scale(1.0) rotate(-5deg); opacity: 0.8; text-shadow: 0 0 5px orange; }
    75% { transform: scale(1.1) rotate(5deg); opacity: 1; text-shadow: 0 0 15px gold, 0 0 20px var(--brand2); }
    100% { transform: scale(1.0) rotate(0deg); opacity: 0.8; text-shadow: 0 0 5px orange; }
}
.flicker-crown-icon { /* New class for the second crown */
    animation: flicker-glow-zoom 2s infinite ease-in-out;
    display: inline-block;
    color: gold;
    font-size: 1.2em;
    margin-right: 5px;
    text-shadow: 0 0 5px orange;
}
/* --- TAB BUTTONS (Blue Active) --- */
.stTabs [data-baseweb="tab-list"] { 
    background-color: #e9ecef; 
    border-bottom: 2px solid #ced4da; 
    margin-bottom: 20px; 
}
/* ENHANCEMENT: Subtle background change on hover for better UX */
.stTabs [data-baseweb="tab"] { 
    transition: background-color 0.3s ease; 
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #f1f7fd; /* Light blue on hover */
}
.stTabs [aria-selected="true"] { 
    color: var(--brand) !important; 
    border-bottom: 4px solid var(--brand) !important; 
    font-weight: 700; 
    text-shadow: 0 0 5px rgba(56, 163, 165, 0.5) !important; 
    background-color: #ffffff; /* White background for active tab */
}
 
/* --- BUTTONS (Primary Blue) --- */
.stButton.btn-primary button { 
    background:linear-gradient(90deg,var(--brand),#2a8082)!important; 
    color:white!important; 
    border:2px solid var(--brand)!important; 
    font-weight: 600; 
    box-shadow: 0 4px 15px rgba(56, 163, 165, 0.4); 
    transition: all 0.3s ease;
}
.stButton.btn-primary button:hover {
    background: var(--brand) !important;
    box-shadow: 0 6px 20px rgba(56, 163, 165, 0.5);
}

/* Plot Wrappers (now transparent with subtle corner glow) */
.plot-wrap {
    border: none;
    border-radius:12px;
    padding:8px 12px;
    background: transparent; /* remove big white card look */
    position: relative;
    overflow: visible;
}
/* subtle corner glows for plots */
.plot-wrap::before,
.plot-wrap::after {
    content: '';
    position: absolute;
    width: 120px;
    height: 80px;
    pointer-events: none;
    filter: blur(18px);
    opacity: 0.6;
    transition: opacity 0.3s ease;
}
.plot-wrap::before {
    top: -10px;
    right: -10px;
    background: linear-gradient(135deg, rgba(0,128,128,0.18), rgba(0,163,163,0.06));
    border-top-right-radius: 12px;
}
.plot-wrap::after {
    bottom: -10px;
    left: -10px;
    background: linear-gradient(225deg, rgba(0,128,128,0.12), rgba(0,163,163,0.04));
    border-bottom-left-radius: 12px;
}
.plot-wrap:hover::before, .plot-wrap:hover::after { opacity: 0.95; }
/* --- MINI SPARKLINE (small line chart under headings) --- */
.mini-sparkline {
    height: 68px;
    margin: 6px 0 12px 0;
    padding: 4px 6px;
    background: transparent;
    border-radius: 10px;
    box-shadow: none !important;
}
.mini-sparkline .js-plotly-plot, .mini-sparkline .plotly-graph-div { background: transparent !important; }
 
/* --- AFFORESTATION ENHANCEMENTS (Blue/Pink/Yellow) --- */
.goal-number {
    font-size: 3.0rem !important;
    font-weight: 800;
    color: #ff69b4 !important; 
    text-shadow: 0 0 10px rgba(255, 105, 180, 0.5);
    transition: all 0.5s ease;
}
.current-number {
    font-size: 2.0rem !important;
    font-weight: 600;
    color: var(--brand2) !important; 
    text-shadow: 0 0 5px rgba(255, 193, 7, 0.5); 
}
.goal-card-title {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #212529;
    text-shadow: 0 0 5px rgba(56, 163, 165, 0.2); 
}
 
/* --- TEAM TITLE GLOW (Vibrant Gradient) --- */
@keyframes pop-glow {
    0% { transform: scale(1.0); opacity: 0.8; text-shadow: 0 0 10px rgba(56, 163, 165, 0.3); }
    50% { transform: scale(1.05); opacity: 1; text-shadow: 0 0 20px rgba(255, 193, 7, 0.7), 0 0 40px rgba(56, 163, 165, 0.5); }
    100% { transform: scale(1.0); opacity: 0.8; text-shadow: 0 0 10px rgba(56, 163, 165, 0.3); }
}
.team-title {
    font-size: 2.2rem !important; 
    font-weight: 900;
    line-height: 1.2;
    margin-top: 20px;
    margin-bottom: 20px;
 
    background: linear-gradient(45deg, #ffc107, #ff69b4, #38a3a5); 
    -webkit-background-clip: text; 
    background-clip: text; 
    color: transparent;

    animation: pop-glow 2.5s infinite alternate ease-in-out;
    display: inline-block;
}

/* ------------------------------ LOGIN STYLES (MODIFIED) ------------------------------ */
.login-box {
    max-width:480px;
    margin:6rem auto 2rem auto;
    padding:32px;
    border-radius:16px;
    background:linear-gradient(180deg, rgba(56, 163, 165, 0.1), rgba(255, 255, 255, 0.8));
    box-shadow:0 10px 30px rgba(0, 0, 0, 0.25);
    border:1px solid rgba(56, 163, 165, 0.2);
    position: relative;
    z-index: 10; /* Keep login box above leaves */
}
.login-title {
    margin:0 0 10px 0;
    color: var(--brand);
    font-family: 'Bebas Neue', sans-serif !important; /* New Font */
    font-size: 3.5rem !important;
    letter-spacing: 2px;
    text-shadow: 0 0 5px rgba(56, 163, 165, 0.5);
    transition: all 0.5s ease;
}
.login-sub {
    color:#495057;
    margin:0 0 16px 0;
    font-family: 'Signika', sans-serif;
}

/* --- LOGIN SUCCESS ANIMATION (GLOW & POP/WATER) --- */
@keyframes success-glow {
    0% { transform: scale(1); text-shadow: 0 0 5px var(--brand); }
    50% { transform: scale(1.05); text-shadow: 0 0 20px var(--brand2), 0 0 30px var(--brand); }
    100% { transform: scale(1); text-shadow: 0 0 5px var(--brand); }
}
 
@keyframes splash {
    0% { clip-path: circle(0% at 50% 50%); }
    50% { clip-path: circle(75% at 50% 50%); }
    100% { clip-path: circle(150% at 50% 50%); }
}
 
.login-title.success {
    color: #fff !important;
    background: linear-gradient(45deg, #38a3a5, #007bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: success-glow 0.8s ease-in-out 3; /* Glow/Pulse */
    position: relative;
    overflow: hidden;
}
 
.login-title.success::after {
    content: "💧 Splash! Welcome!";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(56, 163, 165, 0.7); /* Light Teal Water Effect */
    color: white;
    text-align: center;
    padding-top: 10px;
    clip-path: circle(0% at 50% 50%);
    animation: splash 1.5s ease-out 0.2s 1 forwards; /* Water Splash */
    z-index: 20;
    font-size: 1.5rem;
    font-family: 'Signika', sans-serif;
    font-weight: 700;
    letter-spacing: 0;
    -webkit-text-fill-color: white;
}


/* --- FALLING LEAVES ANIMATION --- */
@keyframes fall {
    0% { transform: translateY(-100px) rotate(0deg); opacity: 0.8; }
    100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
}
 
.leaf-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    overflow: hidden;
    z-index: 1; /* Behind login box */
}
 
.leaf {
    position: absolute;
    color: #00b894; /* MODIFIED: Lighter Green/Teal */
    font-size: 1.5em;
    animation-name: fall;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    animation-duration: 15s;
    text-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
}
 
/* Individual leaf styling for randomness */
.leaf:nth-child(1) { left: 5%; animation-delay: 0s; animation-duration: 12s; }
.leaf:nth-child(2) { left: 15%; animation-delay: 2s; animation-duration: 18s; color: var(--brand2); }
.leaf:nth-child(3) { left: 25%; animation-delay: 4s; animation-duration: 14s; }
.leaf:nth-child(4) { left: 35%; animation-delay: 1s; animation-duration: 20s; color: #ff69b4; }
.leaf:nth-child(5) { left: 45%; animation-delay: 3s; animation-duration: 10s; }
.leaf:nth-child(6) { left: 55%; animation-delay: 5s; animation-duration: 16s; color: #007bff; }
.leaf:nth-child(7) { left: 65%; animation-delay: 0.5s; animation-duration: 13s; }
.leaf:nth-child(8) { left: 75%; animation-delay: 2.5s; animation-duration: 17s; color: var(--brand2); }
.leaf:nth-child(9) { left: 85%; animation-delay: 4.5s; animation-duration: 11s; }
.leaf:nth-child(10) { left: 95%; animation-delay: 1.5s; animation-duration: 19s; }
 
 
/* ------------------------------ GLOBAL MOTION UI HEADERS (NEW) ------------------------------ */

/* 1. Primary Glowing/Animated Header (Unified Teal) */
@keyframes header-glow {
    0% { text-shadow: 0 0 6px rgba(0,128,128,0.25), 0 0 14px rgba(0,128,128,0.12); }
    50% { text-shadow: 0 0 12px rgba(0,128,128,0.45), 0 0 24px rgba(0,128,128,0.25); }
    100% { text-shadow: 0 0 6px rgba(0,128,128,0.25), 0 0 14px rgba(0,128,128,0.12); }
}
.animated-main-header {
    font-family: 'Poppins', sans-serif !important; /* SIMPLE FONT */
    font-size: 2.2rem !important;
    font-weight: 800;
    letter-spacing: 1px;
    margin-top: 20px;
    margin-bottom: 20px;
    line-height: 1.2;
    color: #008080 !important; /* Teal text on white pill */
    background: #ffffff !important; /* Make pill background white as requested */
    border: 1px solid rgba(0,128,128,0.12) !important;
    display: inline-block;
    padding: 10px 20px;
    border-radius: 999px; /* pill */
    box-shadow: 0 6px 16px rgba(0,0,0,0.06), 0 0 18px rgba(0,128,128,0.04);
    animation: header-glow 8s infinite alternate ease-in-out;
    transition: transform 0.25s cubic-bezier(.2,.9,.2,1), box-shadow 0.2s ease;
}
.animated-main-header:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 10px 28px rgba(0,0,0,0.08), 0 0 26px rgba(0,128,128,0.06);
    animation-play-state: paused; /* pause subtle glow while hovering */
}
 
/* 2. Animated Emoji Effect (Fade-in and subtle movement) */
@keyframes float-emoji {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}
.animated-emoji {
    display: inline-block;
    font-size: 1.5em;
    margin-right: 10px;
    animation: float-emoji 1s ease-out 1 forwards;
    animation-delay: var(--delay);
}

/* 3. Gradient Separator with Pulsing Dots */
@keyframes pulse-dot {
    0% { box-shadow: 0 0 0 0 rgba(56, 163, 165, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(56, 163, 165, 0); }
    100% { box-shadow: 0 0 0 0 rgba(56, 163, 165, 0); }
}
.gradient-separator {
    height: 3px;
    /* ADJUSTED: Green-focused gradient */
    background: linear-gradient(90deg, transparent, #38a3a5, #00ff9c, #ffc107, transparent);
    border: none;
    margin: 30px 0;
    position: relative;
}
.gradient-separator::after {
    content: '';
    position: absolute;
    top: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: var(--brand);
    animation: pulse-dot 2s infinite;
}

/* 4. Smooth Fade-in on Load (Simulated scroll effect) */
@keyframes fade-up {
    0% { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}
.fade-in-element {
    opacity: 0;
    animation: fade-up 1s ease-out forwards;
    animation-delay: var(--delay);
}
 
/* 5. Sub-Headings (Clean, Bold, Hover Glow) - Poppins/Montserrat */
.animated-sub-header {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700;
    color: #008080 !important;
    margin-top: 12px;
    margin-bottom: 8px;
    background: #ffffff !important; /* white pill */
    border: 1px solid rgba(0,128,128,0.10) !important;
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.animated-sub-header:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 10px 24px rgba(0,0,0,0.08), 0 0 20px rgba(0,128,128,0.05);
}

/* --- NEW/MODIFIED: Glowing/Flickering Dustbin KPI (for Waste) --- */
@keyframes buldge-flicker {
    0% { transform: scale(1.0); box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1); }
    50% { transform: scale(1.02); box-shadow: 0 12px 30px rgba(255, 193, 7, 0.5), 0 0 30px rgba(255, 165, 0, 0.8); } /* Amber/Orange Glow */
    100% { transform: scale(1.0); box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1); }
}
.metric-card.waste-alert {
    background: var(--card); 
    border:2px solid var(--brand2); /* Amber Border */
    padding:20px; 
    border-radius:12px; 
    box-shadow:0 6px 15px rgba(0, 0, 0, 0.1); 
    transition: all .35s ease; 
    position: relative; 
    color: #212529 !important; 
    overflow: hidden; 
    animation: buldge-flicker 3s infinite alternate ease-in-out;
}
.kpi-waste-value {
    color: #e63946 !important; /* Red for Waste */
    font-weight: 700; 
    font-size: 2.0rem; 
    text-shadow: 0 0 5px rgba(230, 57, 70, 0.3); 
    margin-top: -10px;
}
.waste-icon {
    font-size: 1.5em;
    margin-right: 5px;
    color: var(--brand2); /* Amber icon */
}
.waste-increase-icon {
    color: #e63946; /* Red for predicted increase */
    font-size: 1.2em;
    animation: flicker-red 1s infinite alternate;
}
@keyframes flicker-red {
    0% { opacity: 1; text-shadow: 0 0 5px #e63946; }
    100% { opacity: 0.8; text-shadow: none; }
}

/* ------------------------------ Sidebar Slide Effect (New) ------------------------------ */
.stSidebar {
    transform: translateX(-10px);
    opacity: 0.95;
    transition: transform 0.4s ease-in-out, box-shadow 0.4s ease-in-out;
}
.stSidebar:hover {
    transform: translateX(0);
    box-shadow: 4px 0 20px rgba(56, 163, 165, 0.3);
    opacity: 1;
}
/* For the icon animation added in the sidebar */
.sidebar-icon-container {
    text-align: center;
    padding: 12px 0;
}
.big-earth-icon, .big-tree-icon {
    font-size: 2.6em;
    margin: 0 6px;
    display: inline-block;
    animation: pop-fade 3s ease-in-out infinite alternate;
}
@keyframes pop-fade {
    0%  { transform: scale(1); opacity: 0.85; }
    100% { transform: scale(1.15); opacity: 1; }
}

/* For the Hero button style updates */
.hero-sub-button {
    font-family: 'Signika', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    margin-top: 8px;
    color: rgba(255,255,255,0.9);
}

.rank-mirror-button {
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, #38a3a5, #2a8082);
    border-radius: 8px;
    padding: 8px 12px;
    text-shadow: 0 0 5px rgba(255,255,255,0.2);
}
"""


st.markdown(f"<style>{css_block}</style>", unsafe_allow_html=True)


# -------------------------
# Simple Login Function (MODIFIED)
# -# =========================
# Pretty Login (Neon + Shake + Title Leaves)
# =========================
def _inject_login_theme():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Signika:wght@400;600;700&family=Inter:wght@400;600;800&display=swap');

      [data-testid="stAppViewContainer"]{
        background: radial-gradient(1200px 600px at 15% 10%, #bfeee0 0%, #e8fff3 25%, #f4fff7 60%),
                    linear-gradient(135deg, #c6f2df 0%, #f0ffe9 100%);
      }
      .block-container{ padding-top:3rem; max-width:880px; }

      /* Global falling leaves (background) - MODIFIED LEAF COLOR */
      .leaf-container{ position:fixed; inset:0; pointer-events:none; overflow:hidden; z-index:0; }
      .leaf{ position:absolute; top:-10vh; font-size:18px; opacity:.75;
             color: #00b894; /* Lighter Green/Teal */
             filter:drop-shadow(0 2px 2px rgba(0,0,0,.1));
             animation: fall 9s linear infinite, sway 3.6s ease-in-out infinite alternate; }
    .leaf:nth-child(1){left:7%; animation-delay:.2s,.2s; font-size:20px; color: #00b894;}
      .leaf:nth-child(2){left:18%; animation-delay:1.1s,.1s; color: #ffc107;}
      .leaf:nth-child(3){left:28%; animation-delay:2.0s,.4s; font-size:22px; color: #00b894;}
      .leaf:nth-child(4){left:39%; animation-delay:.6s,.3s; color: #ff69b4;}
      .leaf:nth-child(5){left:50%; animation-delay:1.8s,.7s; font-size:24px; color: #00b894;}
      .leaf:nth-child(6){left:61%; animation-delay:.9s,.6s; color: #007bff;}
      .leaf:nth-child(7){left:70%; animation-delay:2.4s,.5s; color: #00b894;}
      .leaf:nth-child(8){left:79%; animation-delay:1.4s,.2s; font-size:21px; color: #ffc107;}
      .leaf:nth-child(9){left:88%; animation-delay:.3s,.8s; font-size:19px; color: #00b894;}
      .leaf:nth-child(10){left:95%; animation-delay:1.5s,.9s; font-size:19px; color: #ff69b4;}
      @keyframes fall { to { transform: translateY(110vh) rotate(360deg); opacity: 0; } } /* Explicitly defined to disappear */
      @keyframes sway { from { transform: translateX(-18px); } to { transform: translateX(18px); } }

      /* Glass card */
      .login-box{
        position:relative; margin:2.2rem auto 0; width:min(520px,92vw);
        padding:32px 28px 26px; border-radius:22px;
        background:linear-gradient(180deg,rgba(255,255,255,.86),rgba(255,255,255,.82));
        border:1px solid rgba(0,0,0,.06); backdrop-filter:blur(6px);
        box-shadow:0 24px 60px rgba(57,139,97,.15); z-index:1;
      }

      /* --- Title: NEON + periodic shake --- */
      .login-title{
        margin:0; text-align:center; font-family:'Inter',system-ui; font-weight:900;
        font-size:clamp(28px,3.6vw,42px); letter-spacing:.05em;
        background:linear-gradient(180deg,#00ff9c 0%, #00d4a6 70%);
        -webkit-background-clip:text; background-clip:text; color:transparent;
        text-shadow:
          0 0 6px rgba(0,255,170,.65),
          0 0 14px rgba(0,255,128,.40),
          0 2px 0 rgba(0,0,0,.20);
        /* 4s cycle: mostly still, shake at the end */
        animation: pulseShake 4s ease-in-out infinite;
        transition: transform .3s ease;
      }
      .login-title.success{ filter:brightness(1.05); }

      /* tiny periodic shake */
              @keyframes pulseShake{
                  0%,86% { transform: translateY(0) rotate(0); }
                  88% { transform: translateY(-1px) rotate(-0.4deg); }
                  90% { transform: translateY(1px) rotate(0.4deg); }
                  92% { transform: translateY(-1px) rotate(-0.3deg); }
                  94% { transform: translateY(1px) rotate(0.3deg); }
                  100% { transform: translateY(0) rotate(0); }
              }

      .login-sub{ margin:6px 0 18px; text-align:center; color:#305a48; font-weight:600;
                  font-family:'Inter',system-ui; }

      /* Inputs */
      .login-box .stTextInput>div>div>input{
        font-family:'Signika',sans-serif; height:46px; border-radius:14px!important;
        border:1px solid #daf1e4; box-shadow:inset 0 1px 0 rgba(17,63,29,.04);
      }
      .login-box .stTextInput>label{ font-family:'Signika',sans-serif; color:#2f7d56; }
      .stForm { padding:6px 4px; }
      .st-bd.st-cb{ display:block; margin:.15rem 0 .2rem 2px; color:#2f7d56; font-weight:600; }

      /* CTA button */
      .stForm button[kind="primary"]{
        width:100%; height:46px; border-radius:14px; border:none;
        font-weight:800; letter-spacing:.2px; color:#06361d;
        background:linear-gradient(90deg,#10c85b 0%, #7ad04d 100%);
        box-shadow:0 10px 24px rgba(67,169,94,.25);
      }
      .stForm button[kind="primary"]:hover{ filter:brightness(.98); transform:translateY(-1px); }

      /* --- Leaves spawned from the title --- */
      .title-leaves{ position:absolute; left:50%; transform:translateX(-50%); top:76px; width:220px; height:0; pointer-events:none; }
      .title-leaves .tleaf{
        position:absolute; top:-10px; left:50%; transform:translateX(-50%);
        color: #00b894; /* Lighter Green/Teal */
        opacity:.9; font-size:18px; filter:drop-shadow(0 2px 2px rgba(0,0,0,.12));
        animation: fallMini 4.5s linear infinite;
      }
    .title-leaves .tleaf:nth-child(1){ left:30%; animation-delay:.2s; font-size:18px;}
      .title-leaves .tleaf:nth-child(2){ left:55%; animation-delay:1.1s; font-size:20px; color: #ffc107;}
      .title-leaves .tleaf:nth-child(3){ left:72%; animation-delay:2.0s; font-size:19px;}
      .title-leaves .tleaf:nth-child(4){ left:44%; animation-delay:2.7s; font-size:21px; color: #ff69b4;}
      @keyframes fallMini{
    0% { transform:translate(-50%,-6px) rotate(0); opacity:.95; }
        100% { transform:translate(-50%,48px) rotate(260deg); opacity:.0; }
      }
    </style>
    """, unsafe_allow_html=True)


def _login_view():
    _inject_login_theme()

    # Background falling leaves
    st.markdown("""
    <div class='leaf-container'>
      <span class='leaf'>🍃</span><span class='leaf'>🍂</span><span class='leaf'>🍁</span>
      <span class='leaf'>🍃</span><span class='leaf'>🍂</span><span class='leaf'>🍁</span>
      <span class='leaf'>🍃</span><span class='leaf'>🍂</span><span class='leaf'>🍁</span><span class='leaf'>🍃</span>
    </div>
    """, unsafe_allow_html=True)

    if "login_pending" not in st.session_state:
        st.session_state["login_pending"] = False
    success_class = " success" if st.session_state["login_pending"] else ""

    # Card header (TITLE + title-leaves)
    st.markdown(f"""
      <div class="login-box">
        <div style="display:flex;justify-content:center;align-items:center;margin-bottom:10px;">
          <div style="width:52px;height:52px;border-radius:50%;display:grid;place-items:center;
                      background:radial-gradient(circle at 30% 30%,#e9fff2,#d9ffe9);
                      border:1px solid #c9f1da; box-shadow:0 6px 18px rgba(61,160,95,.18);">
            <span style="font-size:24px;">🌿</span>
          </div>
        </div>

        <div class="title-leaves">
          <span class="tleaf">🍃</span><span class="tleaf">🍂</span>
          <span class="tleaf">🍁</span><span class="tleaf">🍃</span>
        </div>

        <h2 class="login-title{success_class}">SustainifyAI</h2>
        <p class="login-sub">AI Model Based Climate Tracker</p>
    """, unsafe_allow_html=True)

    # Form
    with st.form("login_form", clear_on_submit=False):
        st.markdown('<label class="st-bd st-cb" style="font-family:Signika,sans-serif;">User Name</label>', unsafe_allow_html=True)
        u = st.text_input("User Name", key="login_user", label_visibility="collapsed", placeholder="demo@011")

        st.markdown('<label class="st-bd st-cb" style="font-family:Signika,sans-serif;">Password</label>', unsafe_allow_html=True)
        p = st.text_input("Password", key="login_pass", type="password", label_visibility="collapsed", placeholder="••••••••")

        c1, c2 = st.columns([1,1])
        with c1: st.checkbox("Keep me centered", value=True)
        with c2: st.link_button("Restore access", "#", type="secondary")

        submit = st.form_submit_button("Enter Sanctuary")

    st.markdown("</div>", unsafe_allow_html=True)  # close card

    # Auth handling (your existing creds)
    if submit:
        if u == VALID_USER and p == VALID_PASS:
            st.session_state["login_pending"] = True
            st.success("Login successful. Redirecting...")
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = u
            time.sleep(1.2)
            st.rerun()
        else:
            st.error("Invalid username or password.")
            st.session_state["login_pending"] = False

    
# --- LOGIN GATE ---
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None

if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()  # Halt the app here until logged in

# ------------------------------ Utility: Caching & API ------------------------------
@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float, float, str, str]]:
    """Use Open-Meteo geocoding (no API key) to resolve a place to (lat, lon, name, country)."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=20)
    if r.ok:
        js = r.json()
        if js.get("results"):
            res = js["results"][0]
            return float(res["latitude"]), float(res["longitude"]), res.get("name",""), res.get("country","")
    return None

@st.cache_data(show_spinner=False)
def fetch_openmeteo_daily(lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch daily climate variables from Open-Meteo ERA5 reanalysis (no key)."""
    today_date = dt.date.today()
    if end >= today_date:
        api_end_date = today_date - dt.timedelta(days=1)
    else:
        api_end_date = end

    if api_end_date < start:
        return pd.DataFrame({'time': [], 'temperature_2m_max': [], 'temperature_2m_mean': [], 'temperature_2m_min': [], 'precipitation_sum': [], 'windspeed_10m_max': [], 'shortwave_radiation_sum': []})

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": api_end_date.isoformat(), # Use the adjusted end date
        "daily": [
            "temperature_2m_mean","temperature_2m_max","temperature_2m_min",
            "precipitation_sum","windspeed_10m_max","shortwave_radiation_sum",
        ],
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status() 
    js = r.json()
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df

@st.cache_data(show_spinner=False)
def fetch_air_quality_current(lat: float, lon: float) -> pd.DataFrame:
    """Fetch latest air quality using Open-Meteo's Air Quality API (No key required)."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hourly_vars = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "domains": "auto",
        "timezone": "auto",
        "current": ",".join(hourly_vars)
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
    except requests.exceptions.RequestException as e:
        # st.error(f"Air Quality API (Open-Meteo) fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    if "current" in js and "hourly_units" in js:
        current_data = js["current"]
        units = js["hourly_units"]
        last_updated = current_data.get("time")
        
        for param in hourly_vars:
            value = current_data.get(param)
            unit = units.get(param, "ug/m3").replace("µg/m³", "ug/m3") 
            
            if value is not None:
                rows.append({
                    "location": f"{js.get('latitude', lat):.3f}, {js.get('longitude', lon):.3f}",
                    "parameter": param,
                    "value": float(value),
                    "unit": unit,
                    "date": last_updated,
                    "lat": js.get('latitude', lat),
                    "lon": js.get('longitude', lon),
                })

    return pd.DataFrame(rows)
# ------------------------------ NEW WASTE MANAGEMENT FUNCTIONS (Final Stabilization) ------------------------------
# ------------------------------ NEW WASTE MANAGEMENT FUNCTIONS (Final Stabilization) ------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_solid_waste_data(city_name: str = None) -> pd.DataFrame:
    """Simulated Solid Waste Generation data for UP cities and national metros (TPD)."""
    data = {
        'City': ['Prayagraj', 'Lucknow', 'Varanasi', 'Kanpur', 'Noida', 'Ayodhya', 'Mirzapur', 'Agra', 'Meerut', 'Ghaziabad', 'Bareilly', 'Moradabad', # UP Cities
                 'Mumbai', 'Kolkata', 'Delhi', 'Bengaluru'], # National Metros
        'Latitude': [25.4358, 26.8467, 25.3176, 26.4499, 28.5355, 26.8041, 25.1306, 27.1767, 28.9845, 28.6692, 28.3670, 28.8385,
                      19.0760, 22.5726, 28.7041, 12.9716],
        'Longitude': [81.8463, 80.9462, 82.9739, 80.3319, 77.3910, 82.1970, 82.5693, 78.0081, 77.7064, 77.4533, 79.4304, 78.7735,
                       72.8777, 88.3639, 77.1025, 77.5946],
        # Total Solid Waste (TPD - Tons Per Day)
        'Total_MSW_TPD': [1000, 1500, 750, 1600, 1800, 250, 150, 800, 500, 1700, 450, 400,
                          6500, 4500, 11000, 4000],
        # Plastic Waste (TPD - estimated ~8-15% of MSW)
        'Plastic_TPD': [110, 180, 85, 200, 220, 30, 18, 90, 60, 210, 55, 45,
                        800, 550, 1300, 500],
        # Future Trend Prediction (+ve = Increase, -ve = Decrease)
        'Predicted_TPD_Change_%': [12, 8, 15, 5, 10, 18, 20, 7, 9, 11, 14, 16,
                                     -5, 2, 4, -8] 
    }
    df = pd.DataFrame(data)
    
    # --- CRITICAL FIX: Ensure 'City' column is explicitly treated as string (object/bytes) immediately.
    df['City'] = df['City'].astype(str)
    # ------------------------------------------------------------------------------------
    
    if city_name:
        city_lower = city_name.lower().strip()
        
        # We perform the check and concatenation using string comparison
        if city_lower not in [c.lower() for c in df['City']]:
            default_row = {
                'City': city_name, 
                'Latitude': st.session_state.get('lat', 25.4),
                'Longitude': st.session_state.get('lon', 81.8),
                'Total_MSW_TPD': 600,
                'Plastic_TPD': 70,
                'Predicted_TPD_Change_%': 10
            }
            # Concat is safe because the dtypes are consistent
            df = pd.concat([df, pd.DataFrame([default_row])], ignore_index=True)
            
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_plastic_waste_trend(city_name: str) -> pd.DataFrame:
    """Simulated historical and forecasted Plastic Waste (TPD) for trend analysis."""
    df_msw = get_solid_waste_data(city_name)
    
    # Ensure the city match comparison is done safely by converting both sides to string
    # NOTE: The df_msw['City'] is already str from get_solid_waste_data, but casting here ensures safety.
    city_data_match = df_msw[df_msw['City'].astype(str).str.lower() == city_name.lower().strip()]
    
    if city_data_match.empty:
        base_plastic = 70.0
        predicted_change_percent = 10.0
    else:
        city_data = city_data_match.iloc[0]
        base_plastic = float(city_data['Plastic_TPD'])
        predicted_change_percent = float(city_data['Predicted_TPD_Change_%'])
    
    today_year = dt.date.today().year
    years_to_simulate = 5
    years = [today_year - i for i in range(years_to_simulate - 1, -1, -1)] 
    historical_growth_factor = 1.05 
    value_5_years_ago = base_plastic / (historical_growth_factor ** years_to_simulate)
    historical_data = []
    anomaly_year = np.random.choice(years[:-1]) 
    
    for i in range(years_to_simulate):
        year = years[i]
        value = value_5_years_ago + (base_plastic - value_5_years_ago) * (i / (years_to_simulate - 1))
        
        if year == anomaly_year:
              value *= 1.15
              
        ds_date = dt.date(year, 12, 31)
        
        historical_data.append({
            'ds': ds_date,
            'y': value,
            'Year': year,
            'Anomaly': True if year == anomaly_year else False
        })

    df_trend = pd.DataFrame(historical_data).sort_values('ds')
    df_trend['ds'] = pd.to_datetime(df_trend['ds'])
    
    next_year_value = base_plastic * (1 + predicted_change_percent/100)
    year_after_next_value = next_year_value * (1 + predicted_change_percent/100)
    
    df_future = pd.DataFrame({
        'ds': [dt.datetime(today_year + 1, 12, 31), dt.datetime(today_year + 2, 12, 31)],
        'y': [next_year_value, year_after_next_value],
        'Year': [today_year + 1, today_year + 2],
        'Anomaly': [False, False]
    })
    
    return pd.concat([df_trend, df_future], ignore_index=True)

# --- NEW AI FUNCTION TO FETCH LATEST SWACHH WINNER (Final Verified Result) ---
@st.cache_data(ttl=3600) # Cache for 1 hour to reduce API calls
def fetch_latest_swachh_winner_ai():
    """Fetches the latest declared winners, dynamically updating the year when announced."""
    # NOTE: GEMINI_API_KEY is globally accessible.
    if not GEMINI_API_KEY:
        return "Indore & Surat (2023)" # Safest Hardcoded Fallback
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Prompt Gemini to extract the specific information
    # We guide Gemini towards the 2024-25 result based on the search findings.
    prompt = (
        "What are the official primary winners in the Million Plus population category for the "
        "Swachh Survekshan 2024-25 Awards? List the winner(s) and the survey year/edition."
        "Respond ONLY with the name of the winner(s) and the survey year/edition in the format: 'City1, City2 & City3 (Year)'"
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        
        # Clean up the response (removes asterisks/markdown)
        clean_text = response.text.strip().replace('*', '')
        
        # We check the response against expected current results
        if "Indore" in clean_text and ("Surat" in clean_text or "Navi Mumbai" in clean_text):
            # If the AI successfully retrieved the 2024-25 winners, use the dynamic text.
            return clean_text 
        
        # Otherwise, fall back to the safe, last-known result (2023).
        return "Indore & Surat (2023)" 
        
    except Exception:
        # If API fails, fall back to the safe, verified result.
        return "Indore & Surat (2023)"


# --- NEW HELPER FUNCTION FOR THE BADGE (Renderer - This remains correct) ---
def render_swachh_star_badge():
    """Renders the dynamic Swachh Star City badge, displaying the latest verifiable result."""
    
    star_city_text = fetch_latest_swachh_winner_ai()
    
    st.markdown(f"""
    <div class="fade-in-element" style="--delay: 0.2s; text-align: center; background: #f8f9fa; border-radius: 8px; padding: 5px; border: 1px solid var(--brand2);">
    	<p style='color: #212529; font-size: 0.8rem; margin: 0;'>
    		<span class='flicker-crown-icon'>👑</span> Swachh Star City:
    	</p>
    	<p style='color: gold; font-weight: 700; font-size: 1.0rem; margin: 0; text-shadow: 0 0 5px orange;'>
    		{star_city_text}
    	</p>
    </div>
    """, unsafe_allow_html=True)
# ------------------------------ NEW Swachh Survekshan Data Placeholder ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_swachh_ranking_data(city_names: List[str], current_city: str) -> pd.DataFrame:
    """Simulated Swachh Survekshan ranking data."""
    rank_data = {
        'City': [
            # National Top (3)
            'Indore', 'Surat', 'Navi Mumbai', 
            
            # UP Nagar Nigams (17 Cities)
            'Lucknow', 'Varanasi', 'Prayagraj', 'Gorakhpur', 'Agra', 'Ghaziabad', 'Meerut', 
            'Moradabad', 'Aligarh', 'Bareilly', 'Kanpur', 'Firozabad', 'Ayodhya', 
            'Jhansi', 'Saharanpur', 'Mathura-Vrindavan', 'Shahjahanpur',
            
            # Key Nagar Palika Parishad (NPP) Proxies (3)
            'Bijnor', 'Shamshabad', 'Basti',
        ],
        
        # Rankings (23 items in each list - CORRECTED to match City list length)
        '2021': [
            1, 2, 4, 		# National (3)
            40, 35, 60, 90, 100, 150, 160, # Tier 1 UP (7)
            180, 200, 220, 250, 280, 290, # Tier 2 UP (6)
            310, 330, 350, 370, 		# Tier 3 UP (4)
            400, 420, 440 				# NPP Proxies/Lower Ranks (3)
        ],	
        
        '2022': [
            1, 2, 3,
            30, 28, 50, 75, 85, 130, 140,
            160, 180, 200, 230, 260, 270,
            290, 310, 330, 350, 		# Tier 3 UP (4)
            380, 400, 420
        ],
        
        '2023': [
            1, 1, 3,
            20, 22, 40, 60, 70, 110, 120,
            140, 160, 180, 210, 240, 250,
            270, 290, 310, 330, 		# Tier 3 UP (4)
            360, 380, 400
        ],
        
        # Simulated/Target (Based on recent performance improvements)
        '2024': [
            1, 1, 3,
            15, 18, 35, 50, 60, 90, 100,
            120, 140, 160, 190, 220, 230,
            250, 270, 290, 310, 		# Tier 3 UP (4)
            340, 360, 380
        ],
        
        '2025': [
            1, 1, 3,
            10, 15, 30, 40, 50, 80, 90,
            110, 130, 150, 180, 210, 220,
            240, 260, 280, 300, 		# Tier 3 UP (4)
            320, 340, 360
        ],
    }
    df = pd.DataFrame(rank_data)
    
    city_map = {c.lower(): c for c in df['City'].unique()}
    filtered_cities = []
    
    for name in city_names:
        matched_name = city_map.get(name.strip().lower())
        if matched_name and matched_name not in filtered_cities:
            filtered_cities.append(matched_name)

    current_city_lower = current_city.lower()
    city_is_known = current_city_lower in [c.lower() for c in filtered_cities]
    
    if not city_is_known:
        # If the user's current city is not a major corporation, 
        # assign it a dynamic, high, but improving rank (e.g., starting at rank 500)
        default_rank_series = {
            'City': current_city, 
            '2021': 500, '2022': 480, '2023': 450, '2024': 420, '2025': 390
        }
        df = pd.concat([df, pd.DataFrame([default_rank_series])], ignore_index=True)
        filtered_cities.append(current_city)
    
    return df[df['City'].isin(filtered_cities)]

# ------------------------------ NEW Pollution Ranking Function (IMPLEMENTATION) ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_pollution_rank_proxy(current_city_name: str, current_pm25: float) -> Tuple[int, str]:
    """
    Simulated rank based on PM2.5 level compared to 17 major UP Municipal Corporations.
    Lower PM2.5 means a better (lower number) rank.
    """
    # Simulated latest PM2.5 data (ug/m3) for the 17 UP Municipal Corporations
    simulated_city_data = {
        'City': ['Shahjahanpur', 'Mathura-Vrindavan', 'Jhansi', 'Ayodhya', 'Gorakhpur', 
                 'Firozabad', 'Aligarh', 'Saharanpur', 'Moradabad', 'Bareilly', 
                 'Meerut', 'Agra', 'Ghaziabad', 'Lucknow', 'Kanpur', 'Varanasi', 'Prayagraj'],
        'PM25_Value': [48.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 65.0, 68.0, 70.0, 
                        72.0, 75.0, 78.0, 80.0, 85.0, 90.0, 100.0]
    }
    df_rank_base = pd.DataFrame(simulated_city_data)

    # 1. Standardize and add the current selected city's live data
    city_data = {'City': current_city_name, 'PM25_Value': current_pm25}
    
    # Remove the current city if it's already in the base data to use the live reading
    df_rank_base = df_rank_base[df_rank_base['City'].str.lower() != current_city_name.lower()]
    
    # Add the current city (live data)
    df_rank_base = pd.concat([df_rank_base, pd.DataFrame([city_data])], ignore_index=True)

    # 2. Sort by PM2.5 value (ascending, lower is better)
    df_rank_base = df_rank_base.sort_values(by='PM25_Value', ascending=True).reset_index(drop=True)
    
    # 3. Assign Rank (index + 1)
    df_rank_base['Rank'] = df_rank_base.index + 1

    # 4. Find the rank of the current city
    current_rank_row = df_rank_base[df_rank_base['City'].str.lower() == current_city_name.lower()]
    
    if current_rank_row.empty:
        # Fallback if the city is somehow not in the final list
        final_rank = len(df_rank_base)
    else:
        final_rank = int(current_rank_row.iloc[0]['Rank'])
    
    # 5. Determine the status message (Total 17 cities + current live city)
    total_cities = len(df_rank_base) 
    
    # Determine Status based on your criteria (Poor for lower rank on pollution list)
    if final_rank <= 5:
        status = "Air Quality: BEST (Low Pollution)"
    elif final_rank <= 12:
        status = "Air Quality: MODERATE"
    else:
        status = "Air Quality: WORST (High Pollution)"

    return final_rank, f"{final_rank} / {total_cities} Corporations"

# ------------------------------ NEW Impact Simulation Functions (Retained from previous code) ------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_river_health_data(city_name: str):
    """Synthesizes data for the major river near the selected city."""
    
    city_lower = city_name.lower()
    
    if "kanpur" in city_lower:
        river = "Ganga (Kanpur)"
        do, bod, coliform, status = 5.8, 4.5, 2500, "Critical Stress"
    elif "varanasi" in city_lower:
        river = "Ganga (Varanasi)"
        do, bod, coliform, status = 6.8, 3.2, 1200, "High Stress"
    elif "lucknow" in city_lower or "jaunpur" in city_lower:
        river = "Gomti (UP)"
        do, bod, coliform, status = 5.0, 4.0, 3000, "Critical Stress"
    elif "prayagraj" in city_lower or "allahabad" in city_lower:
        river = "Ganga (Sangam/Prayagraj)"
        do, bod, coliform, status = 7.0, 3.5, 11000, "High Stress"
    else:
        # Default/General River Logic 
        river = f"{city_name} River (General)"
        do, bod, coliform, status = 7.5, 2.5, 800, "Moderate Stress"
        
    data = {
        "River": [river],
        "Dissolved Oxygen (DO mg/L)": [do], # Healthy > 6.0
        "BOD (mg/L)": [bod], # Good < 3.0
        "Coliform (MPN/100ml)": [coliform], # Safe < 500
        "Status": [status],
    }
    df = pd.DataFrame(data)
    df['Color'] = df['Status'].apply(lambda x: '#e63946' if x == 'Critical Stress' or x == x == 'Extreme Stress' else ('#ffc107' if x == 'High Stress' else '#38a3a5'))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_tree_inventory(city_name: str):
    """Synthesizes tree data and requirements for the selected city (Maximized UP Granularity)."""
    
    city_lower = city_name.lower()
    population_proxies = {
        "delhi": 19000000, "mumbai": 20000000, "bengaluru": 13000000, "chennai": 8000000,
        "kanpur": 2700000, "lucknow": 2700000, "ghaziabad": 2500000, "agra": 1800000,
        "varanasi": 1500000, "meerut": 1500000, "bareilly": 1200000, "aligarh": 1000000,
        "moradabad": 1000000, "firozabad": 1000000, "jhansi": 800000, "gorakhpur": 800000,
        "prayagraj": 1600000, "allahabad": 1600000, # Prayagraj specific population
    }

    current_trees_proxies = {
        "delhi": 3000000, "mumbai": 1500000, "bengaluru": 1200000, "chennai": 900000,
        "kanpur": 850000, "lucknow": 950000, "ghaziabad": 650000, "agra": 400000,
        "varanasi": 500000, "meerut": 350000, "bareilly": 300000, "aligarh": 250000,
        "moradabad": 250000, "firozabad": 200000, "jhansi": 180000, "gorakhpur": 190000,
        "prayagraj": 550000, "allahabad": 550000, # Prayagraj specific tree count
    }
    
    # Get base values, defaulting to a smaller urban size if city is not listed
    population = population_proxies.get(city_lower, 400000)
    current_trees = current_trees_proxies.get(city_lower, 100000)
        
    target_ratio = 10 # Trees per person (national standard recommendation)
    trees_needed = (population * target_ratio) - current_trees
    
    return {
        "city": city_name,
        "current": current_trees,
        "population": population,
        "target_ratio": target_ratio,
        "needed": max(0, trees_needed),
        "needed_per_capita": round(trees_needed / population, 2)
    }

def get_future_impact_prediction(pm25_level: float):
    """Predicts generalized health impact based on current PM2.5."""
    if pm25_level < 50:
        return {"health_risk": "Low", "advice": "Continue outdoor activities.", "color": "#38a3a5"} # Teal/Blue-Green for Good
    elif 50 <= pm25_level < 100:
        return {"health_risk": "Moderate", "advice": "Sensitive groups should limit prolonged outdoor exertion.", "color": "#ffc107"} # Yellow/Amber for Warning
    else:
        return {"health_risk": "High", "advice": "All groups should avoid prolonged or heavy exertion outdoors. Wear N95 masks.", "color": "#e63946"} # Red for Critical

def get_crop_loss_simulation(mean_temp: float) -> Tuple[float, float, str]:
    """Calculates simulated crop yield based on temperature rise."""
    base_yield = 100
    critical_temp = 25.0 # Critical threshold for Indian dry season crops (simple proxy)
    
    if mean_temp > critical_temp:
        loss_factor = min(0.35, (mean_temp - critical_temp) * 0.05) # Max loss 35%
        stressed_yield = base_yield * (1 - loss_factor)
        loss_percent = round(loss_factor * 100, 1)
        status = f"Severe stress (Avg Temp > {critical_temp} deg C)"
    else:
        stressed_yield = base_yield
        loss_percent = 0.0
        status = "Optimal/Moderate Stress"
        
    return stressed_yield, loss_percent, status

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_india_city_emissions() -> pd.DataFrame:
    """Mock CO2 data for major Indian cities for the Choropleth map."""
    data = {
        'City': ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', 'Varanasi', 'Prayagraj', 'Kanpur', 'Lucknow', 'Ahmedabad'],
        'Latitude': [19.0760, 28.7041, 12.9716, 13.0827, 22.5726, 25.3176, 25.4358, 26.4499, 26.8467, 23.0225],
        'Longitude': [72.8777, 77.1025, 77.5946, 80.2707, 88.3639, 82.9739, 81.8463, 80.3319, 80.9462, 72.5714],
        'CO2_Emissions_Annual_kT': [45000, 52000, 31000, 19000, 25000, 3500, 4200, 6000, 6500, 15000], # Kilotons per year proxy
    }
    return pd.DataFrame(data)

# --- NEW: Green Infrastructure & EV Placeholder Data Functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_pollution_free_energy_data(city_names: List[str]) -> pd.DataFrame:
    """Simulated pollution-free energy resource data (Capacity in MW)."""
    city_base_values = {
        "Prayagraj": {"Solar": 150, "Wind": 5, "Hydro": 20},
        "Lucknow": {"Solar": 200, "Wind": 10, "Hydro": 15},
        "Varanasi": {"Solar": 120, "Wind": 3, "Hydro": 10},
        "Kanpur": {"Solar": 180, "Wind": 8, "Hydro": 25},
        "Mumbai": {"Solar": 400, "Wind": 50, "Hydro": 30},
        "Delhi": {"Solar": 350, "Wind": 20, "Hydro": 5},
        "Bengaluru": {"Solar": 300, "Wind": 15, "Hydro": 20},
        "Agra": {"Solar": 80, "Wind": 5, "Hydro": 15},
    }
    rows = []
    for city in city_names:
        clean_city = city.strip()
        data = city_base_values.get(clean_city, {"Solar": 75, "Wind": 2, "Hydro": 10})
        rows.append({"City": clean_city, "Solar (MW)": data["Solar"], "Wind (MW)": data["Wind"], "Hydro (MW)": data["Hydro"]})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def get_registered_solar_connections(city_names: List[str]) -> pd.DataFrame:
    """Simulated registered solar connections (Units)."""
    city_base_values = {
        "Prayagraj": 25000, "Lucknow": 35000, "Varanasi": 20000, "Kanpur": 30000,
        "Mumbai": 80000, "Delhi": 70000, "Bengaluru": 60000, "Agra": 18000,
    }
    rows = []
    for city in city_names:
        clean_city = city.strip()
        connections = city_base_values.get(clean_city, np.random.randint(8000, 15000))
        rows.append({'City': clean_city, 'Solar Connections': connections})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def get_registered_ev_vehicles(city_names: List[str]) -> pd.DataFrame:
    """Simulated registered EV vehicles (Units)."""
    city_base_values = {
        "Prayagraj": 18000, "Lucknow": 25000, "Varanasi": 15000, "Kanpur": 22000,
        "Mumbai": 70000, "Delhi": 100000, "Bengaluru": 80000, "Agra": 12000,
    }
    rows = []
    for city in city_names:
        clean_city = city.strip()
        evs = city_base_values.get(clean_city, np.random.randint(5000, 12000))
        rows.append({'City': clean_city, 'Registered EVs': evs})
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def get_pollution_news_ticker(city_name: str) -> str:
    """Combines suggested text into a single, moving line, dynamically localized by city."""
    
    # --- City-Specific Dynamic Content ---
    river_data = get_river_health_data(city_name)
    tree_data = get_tree_inventory(city_name)
    
    # 1. City-Specific Tree Goal Headline
    trees_needed_str = f"{tree_data['needed']:,}"
    tree_headline = f"💡 {city_name} local bodies initiate massive tree plantation drive to bridge the {trees_needed_str} gap."
    
    # 2. City-Specific River Health Headline
    river_status = river_data['Status'].iloc[0]
    river_name = river_data['River'].iloc[0].split(' (')[0]
    river_headline = f"💧 {river_name} health at '{river_status}' status; BOD reduction initiatives intensify."
    
    # 3. City-Specific Swachh Rank Headline (Check against top 50)
    rank_df = get_swachh_ranking_data([city_name], city_name)
    current_rank = rank_df[rank_df['City'] == city_name].get('2025', pd.Series([np.nan])).iloc[0]
    
    swachh_headline = ""
    if not np.isnan(current_rank) and current_rank < 50:
        swachh_headline = f"🏆 {city_name} aims for Top 50 Global Swachh Rank; new waste collection methods deployed."
    else:
        swachh_headline = f"♻ New mandate targets 50% waste recycling to improve Swachh ranking for {city_name}."
        
    # --- Assemble final list (repeat for smooth scrolling) ---
    news_items = [
        tree_headline,
        swachh_headline,
        river_headline,
        "⚡ Focus shifts to solar roof-tops to boost urban renewable energy capacity.",
    ]
    
    return " | ".join(news_items) * 3

# ------------------------------ Sustainability Score ------------------------------

@dataclass
class SustainabilityInputs:
    pm25: float
    co2_per_capita: float              # optional proxy if available
    renewable_share: float # 0..100
    water_quality_index: float # 0..100
    waste_recycling_rate: float # 0..100


def compute_sustainability_score(inp: SustainabilityInputs) -> Tuple[float, Dict[str, float]]:
    """Composite score 0-100 with interpretable sub-scores and weights."""
    # Lower value is better for PM2.5 and CO2, using a reference bad threshold to scale to 0-1.
    pm25_scaled = np.clip(1 - (inp.pm25 / 75.0), 0, 1)  # 75 ug/m3 ~ very poor -> 0 score
    co2_scaled = np.clip(1 - (inp.co2_per_capita / 20.0), 0, 1)  # 20 t/cap ~ bad -> 0 score
    
    # Higher value is better for the rest
    ren_scaled = np.clip(inp.renewable_share / 100.0, 0, 1)
    water_scaled = np.clip(inp.water_quality_index / 100.0, 0, 1)
    waste_scaled = np.clip(inp.waste_recycling_rate / 100.0, 0, 1)

    weights = {
        "Air Quality (PM2.5)": 0.28,
        "CO2 / Capita": 0.18, 
        "Renewables Share": 0.24,
        "Water Quality": 0.15,
        "Recycling Rate": 0.15,
    }
    sub_scores_raw = {
        "Air Quality (PM2.5)": pm25_scaled,
        "CO2 / Capita": co2_scaled,
        "Renewables Share": ren_scaled,
        "Water Quality": water_scaled,
        "Recycling Rate": waste_scaled,
    }
    score = sum(sub_scores_raw[k]*w for k, w in weights.items()) * 100
    return float(score), {k: round(v*100, 1) for k, v in sub_scores_raw.items()}

# ------------------------------ Forecasting Helpers ------------------------------

def backtest_train_forecast(df: pd.DataFrame, target_col: str, horizon: int = 30, model_choice: str = "auto"):
    """Time-series train/validation split, fit model, forecast horizon days. Returns forecast and metrics."""
    ts = df[["time", target_col]].dropna().copy()
    ts = ts.sort_values("time")
    ts.rename(columns={"time":"ds", target_col:"y"}, inplace=True)

    # Use last 20% as validation
    n = len(ts)
    if n < 100:
        horizon = max(7, min(horizon, n//5))
    split_idx = max(5, int(n*0.8))
    train, valid = ts.iloc[:split_idx], ts.iloc[split_idx:]

    y_pred = None

    def prophet_fit_forecast():
        m = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(train)
        future = m.make_future_dataframe(periods=horizon)
        fcst = m.predict(future)
        return m, fcst, m.predict(valid[["ds"]])["yhat"].values, valid["y"].values

    def arima_fit_forecast():
        # Suppress internal warnings 
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = auto_arima(train["y"], seasonal=True, m=365, suppress_warnings=True, error_action='ignore', stepwise=True)
        
        yhat_valid = model.predict(n_periods=len(valid))
        
        steps = horizon
        future_preds = model.predict(n_periods=steps)
        fcst = pd.DataFrame({
            "ds": pd.date_range(valid["ds"].iloc[-1] + pd.Timedelta(days=1), periods=steps, freq='D'),
            "yhat": future_preds
        })
        return model, fcst, yhat_valid, valid["y"].values

    def ml_fit_forecast():
        # Simple lag features RF
        full = pd.concat([train, valid], axis=0).reset_index(drop=True)
        for lag in [1,2,7,14,30]:
            full[f"lag_{lag}"] = full["y"].shift(lag)
        full.dropna(inplace=True)
        
        # Prepare for ML model
        X = full.drop(columns=["ds","y"]).values
        y = full["y"].values
        split = int(len(full)*0.8)
        Xtr, Xva = X[:split], X[split:]
        ytr, yva = y[:split], y[split:]
        rf = RandomForestRegressor(n_estimators=400, random_state=42)
        rf.fit(Xtr, ytr)
        
        # Backtest prediction
        y_pred_bt = rf.predict(Xva)
        
        # Iterative future forecast (crucial for time series with lag features)
        last = full.iloc[-1:].copy()
        future_rows = []
        for i in range(horizon):
            feats = last.drop(columns=["ds","y"]).values
            yhat = rf.predict(feats)[0]
            new_date = last["ds"].iloc[0] + pd.Timedelta(days=1)
            row = {
                "ds": new_date,
                "y": np.nan,
            }
            # shift lags (Lag 1 becomes the prediction, Lag N becomes Lag N-1)
            for lag in [1,2,7,14,30]:
                if lag == 1:
                    row[f"lag_{lag}"] = yhat
                else:
                    # Look up the previous day's lag value (which corresponds to the lag-1 step)
                    # Use a complex lookup for correctness in iterative prediction
                    lag_col_name = f"lag_{lag-1}"
                    # If the lag column existed in the original input to the model, use the previous day's value
                    if lag_col_name in last.columns:
                        row[f"lag_{lag}"] = last[lag_col_name].iloc[0]
                    else:
                        # Fallback for short histories where lags don't exist
                        row[f"lag_{lag}"] = yhat 

            future_rows.append(row)
            last = pd.DataFrame([row]) # Update 'last' row for the next iteration
            
        fcst = pd.DataFrame(future_rows)
        fcst.rename(columns={"y": "yhat"}, inplace=True)
        return rf, fcst, y_pred_bt, yva


    model_used = None
    metrics = {"MAE": None, "MAPE": None}

    try:
        if (model_choice == "Prophet" and _HAS_PROPHET) or (model_choice == "auto" and _HAS_PROPHET):
            model_used = "Prophet"
            m, fcst, yhat_valid, y_valid = prophet_fit_forecast()
            metrics["MAE"] = float(mean_absolute_error(y_valid, yhat_valid))
            metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, yhat_valid))
        elif (model_choice == "ARIMA" and _HAS_ARIMA) or (model_choice == "auto" and _HAS_ARIMA):
            model_used = "ARIMA"
            m, fcst, yhat_valid, y_valid = arima_fit_forecast()
            metrics["MAE"] = float(mean_absolute_error(y_valid, yhat_valid))
            metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, yhat_valid))
        else:
            model_used = "ML Ensemble"
            m, fcst, y_pred_bt, y_valid = ml_fit_forecast()
            metrics["MAE"] = float(mean_absolute_error(y_valid, y_pred_bt))
            metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, y_pred_bt))
    except Exception as e:
        # Fallback to ML if the selected model fails unexpectedly
        # st.error(f"Selected model '{model_choice}' failed. Falling back to ML Ensemble. Error: {e}")
        model_used = "ML Ensemble"
        m, fcst, y_pred_bt, y_valid = ml_fit_forecast()
        metrics["MAE"] = float(mean_absolute_error(y_valid, y_pred_bt))
        metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, y_pred_bt))

    return model_used, ts, train, valid, fcst, metrics

# ------------------------------ Alerts (Telegram Optional) ------------------------------

def send_telegram(msg: str) -> bool:
    # Telegram code remains the same
    return False # Disabled for public code submission

# ------------------------------ Gauge Color Utility ------------------------------

def get_gauge_color(value, good_threshold, bad_threshold, reverse=False):
    """Returns color based on value and thresholds. Good is green, Bad is red. Using Bright Palette."""
    COLOR_GOOD = "#38a3a5" # Teal/Blue-Green
    COLOR_WARNING = "#ffc107" # Yellow/Amber
    COLOR_CRITICAL = "#e63946" # Red
    
    if reverse: # Lower value is better (e.g., CO2, PM2.5)
        if value <= good_threshold:
            return COLOR_GOOD 
        elif value < bad_threshold:
            return COLOR_WARNING 
        else:
            return COLOR_CRITICAL 
    else: # Higher value is better (e.g., Renewable Share, Water Quality)
        if value >= good_threshold:
            return COLOR_GOOD 
        elif value > bad_threshold:
            return COLOR_WARNING 
        else:
            return COLOR_CRITICAL 
# --- NEW HELPER: GENERIC DONUT CHART RENDERER ---
def render_donut_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str, custom_colors: List[str]):
    """Renders a standard donut chart with custom colors and size."""
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=f'*{title}*',
        hole=0.5,
        color_discrete_sequence=custom_colors
    )
    fig.update_layout(
        height=350, # Set a consistent, medium height
        margin=dict(l=5, r=5, t=30, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_traces(textposition='inside', textinfo='percent')
    return fig
            
# ------------------------------ CHATBOT LOGIC (FINAL STABLE VERSION) ------------------------------
# --- NEW UTILITY FUNCTION FOR CONTEXTUAL AI (FOR USE OUTSIDE chat_interface_embed) ---

# --- NEW AI INSIGHT GENERATION FUNCTION (Helper) ---
# --- GOVERNANCE ADVICE LOCAL FALLBACK (CRITICAL UTILITY) ---

def get_priority_governance_advice(city_name: str, msw_tpd: float, pm25: float, water_idx: float) -> str:
    """Provides structured priority advice based on local simulated metrics (local fallback)."""
    
    priorities = {}
    
    # Priority scoring logic (Higher score means worse performance/higher risk)
    if pm25 > 80: priorities['Air Quality (PM2.5)'] = 50 
    elif pm25 > 60: priorities['Air Quality (PM2.5)'] = 20
        
    if msw_tpd > 1500: priorities['Waste Management (TPD)'] = 60
    elif msw_tpd > 800: priorities['Waste Management (TPD)'] = 30
        
    if water_idx < 50: priorities['Water Quality (Index)'] = 70
    elif water_idx < 70: priorities['Water Quality (Index)'] = 40
        
    if not priorities:
        return f"Current environmental status for **{city_name}** is generally Moderate/Good across key metrics. Focus on maintaining **Green Infrastructure investment** to sustain momentum."
        
    top_priority = max(priorities, key=priorities.get)
    
    if 'Waste' in top_priority:
        return (
            "**Priority 1: Solid Waste Management 🗑️**\n"
            f"Local data indicates high waste stress (**{msw_tpd:.0f} TPD**). Immediate actions must include:\n"
            "1. **Mandatory 3-way segregation** enforcement at source.\n"
            "2. **Formalize and integrate** the informal waste-picker community for recycling efficiency.\n"
            "3. **Decentralized composting** initiatives for organic waste."
        )
    elif 'Air Quality' in top_priority:
        return (
            "**Priority 1: Air Quality Control 💨**\n"
            f"The high PM25 level (**{pm25:.1f} ug/m3**) requires urgent intervention:\n"
            "1. **Construction Dust Control:** Strict enforcement of covers and sprinkles at all construction sites.\n"
            "2. **Transit Shift:** Fast-track expansion of public transport and create 'No-Vehicle' zones.\n"
            "3. **Enforce Ban:** Aggressive penalization for open garbage and biomass burning."
        )
    elif 'Water Quality' in top_priority:
        return (
            "**Priority 1: River/Water Health 🌊**\n"
            f"The Water Quality Index (**{water_idx:.1f}%)** signals critical pollution stress. Focus on:\n"
            "1. **STP Mandate:** Ensure **100% treatment capacity** for all wastewater entering local rivers.\n"
            "2. **Septage Management:** Improve desludging and treatment of septic tank waste.\n"
            "3. **Industrial Effluent Monitoring** with zero liquid discharge mandates."
        )
    else:
        return "Local environmental data shows balanced pressure. Focus on **Energy Efficiency** and **Renewable Energy Expansion** to secure the score."


# --- DEDICATED FALLBACK RENDERER (To ensure correct visualization of local data) ---
def render_local_governance_fallback(_name, co2_pc, current_msw_tpd, pm25_now, water_idx):
    """Guarantees rendering of the local fallback advisory with guaranteed data."""
    # Ensure all inputs are robust, even if upstream data fetch failed
    pm = pm25_now if not math.isnan(pm25_now) else 80
    msw = current_msw_tpd if not math.isnan(current_msw_tpd) else 1000
    wtr = water_idx if not math.isnan(water_idx) else 65

    st.markdown("""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🎯</span>
                Virtual Analyst: Priority Governance Roadmap
            </h2>
        </div>
    """, unsafe_allow_html=True)

    advice_content = get_priority_governance_advice(_name, msw, pm, wtr)
    
    st.markdown(f"""
        <div class='metric-card fade-in-element' style='--delay: 0.5s; padding: 15px; border-left: 5px solid #ffc107; background: #fff8e1;'>
            <p style='color: #6a0000; font-weight: 700; margin-bottom: 5px;'>Local Advisory Mode ⚠️ (AI Connection Offline)</p>
            <p style='color: #212529; font-size: 0.95rem; line-height: 1.5;'>
                {advice_content}
            </p>
        </div>
    """, unsafe_allow_html=True)

def generate_contextual_ai_insight(system_prompt_addition: str, user_prompt: str) -> str:
    """
    Sends a specialized, one-off request to Gemini for high-level analytical insight.
    This function is used by the analysis cards in the Overview, Forecast, and Score tabs.
    """
    # Check for client existence, which is the most reliable check against NameError.
    if GEMINI_CLIENT is None:
        return "🤖 Analysis currently unavailable. The Virtual Data Scientist is offline."

    try:
        # Use the globally cached client
        client = GEMINI_CLIENT 
        
        # Combine base system prompt with the specific task instruction
        full_system_instruction = """
        You are an expert AI Data Auditor and Policy Analyst embedded in the SustainifyAI dashboard. 
        Focus strictly on the data and context provided. Be professional, concise, and highly actionable.
        """ + system_prompt_addition

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_prompt],
            config=types.GenerateContentConfig(system_instruction=full_system_instruction)
        )
        return response.text
        
    except Exception:
        return "🤖 Analysis currently unavailable. The Virtual Data Scientist is offline."

# ------------------------------ CHATBOT LOGIC (SMART CONTEXTUAL VERSION - Chat Window) ------------------------------

def generate_chatbot_response(prompt: str) -> str:
    """
    Handles the main chat window dialogue, prioritizing hardcoded/quick answers.
    """
    prompt_lower = prompt.lower().strip()
    norm = " ".join(prompt_lower.split())

    if any(keyword in prompt_lower for keyword in ["hello", "hi", "hey", "good morning", "good evening"]):
        return "Hello! I am SustainifyAI's Virtual Data Scientist. I'm here to analyze your data and guide you through the dashboard features. What can I analyze for you today? 💡"
    
    # --- Hardcoded Team/Project Details for fast response (as requested) ---
    if any(x in norm for x in ["who built", "who made", "creator", "team lead", "contact", "mentor", "akash pandey"]):
        return (
            "### About SustainifyAI - The Team\n"
            "**Created by:** *Nxt Zen Developers* (Technosavvys)\n"
            "**Guidance:** *Akash Pandey Sir*\n"
            "**Team Lead:** *Aditya Kumar Singh* (Backend • Data Science/AI/ML)\n"
            "**Core Member:** *Gaurang Verma* (Frontend • UI/UX)\n"
            "**Contact:** iitianadityakumarsingh@gmail.com"
        )
        
    # --- Fallback for Generic Metric definitions ---
    # Check against GEMINI_CHAT availability, which holds the session.
    if not GEMINI_CHAT:
        if "pm2.5" in norm or "air quality" in norm:
            return "💨 **PM2.5** is microscopic particulate matter; lower values mean better air quality for health. (AI is offline, using quick reference)."
        return "I can answer specific questions about the **dashboard's features, technical metrics, or any of the graphs** you see. Try asking about the **'Sustainability Score'** or the **'Anomaly Tracker'**! (Using fallback logic as Gemini client is unavailable.)"

    # --- Call Gemini for contextual/dynamic questions ---
    try:
        # GEMINI_CHAT holds the ongoing session object
        response = GEMINI_CHAT.send_message(prompt)
        return response.text
        
    except APIError as e:
        return f"🤖 Sorry, the Gemini API is temporarily unavailable. (Error: {e})"
    except Exception as e:
        return f"🤖 An unexpected error occurred with the AI model. Try asking about a specific metric like 'PM2.5'. (Debug: {type(e).__name__})"

# Initialize chat history (Keep this outside the functions, but before they are called)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! 👋 I am SustainifyAI's virtual assistant. Ask me anything about the dashboard's data or features."})


def chat_interface_embed():
    """
    Renders the embedded chatbot interface, now using the animated header.
    """
    st.markdown("<div class='metric-card fade-in-element' style='--delay: 1.5s; padding: 10px 15px;'>", unsafe_allow_html=True)
    
    # --- CRITICAL UPDATE: Call the proper animated header function ---
    # This replaces the boilerplate H3 and st.caption you previously had here.
    try:
        render_animated_chatbot_header()
    except NameError:
        st.markdown(
            f"""
            <h3 class="animated-sub-header" style="font-size: 1.4rem !important; margin-top: 5px;">
                 <span class="animated-emoji" style="--delay: 1.6s;">🤖</span> SustainifyAI Chatbot
            </h3>
            <p style='color: var(--muted); font-size: 0.9rem;'>Ask me about the project's features, scores, or graphs!</p>
            """, unsafe_allow_html=True
        )

    chat_container = st.container(height=420)
    
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the dashboard..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chatbot_response(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown("</div>", unsafe_allow_html=True)

# --- NEW UTILITY FUNCTION: AI Insight Renderer ---
def render_ai_analysis_card(title: str, prompt_text: str, delay: float):
    """Generates and renders an AI-driven text analysis card."""
    # NOTE: generate_contextual_ai_insight must be defined elsewhere in the script.
    # Placeholder implementation:
    ai_insight = f"AI processing analysis for: {prompt_text[:60]}..."
    
    st.markdown(f"""
    <div class='metric-card fade-in-element' style='--delay: {delay}s; padding: 15px; border-left: 5px solid var(--brand2);'>
        <p style='color: #008080; font-weight: 700; margin-bottom: 5px;'>{title} 🧠</p>
        <p style='color: #343a40; font-size: 0.95rem; line-height: 1.3;'>{ai_insight}</p>
    </div>
    """, unsafe_allow_html=True)


# --- ANIMATED CHATBOT HEADER HELPER (Must be defined above call) ---
def render_animated_chatbot_header():
    """Renders the animated, interactive chatbot header."""
    # Note: Using the new design structure with animations
    st.markdown(f"""
    <div class="chatbot-header-wrap">
        <span class="chatbot-icon">🤖</span>
        <div>
            <p class="chatbot-title">SustainifyAI Chatbot</p>
            <p class="chatbot-prompt">
                Ask me anything about the dashboard's data or features
                <span class="flicker-hand">👇</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
# ------------------------------------------------------------------


# --- 1. CO2 Emissions / Capita (t) ---
    with col1:
        # 1A. Insert AI Analysis Card
        prompt_co2 = f"Analyze the current CO2 per capita reading of {co2_pc_val:.1f} tons for {_name} against the global 2.0t target. Is this score high, moderate, or low risk? Propose one immediate action area."
        render_ai_analysis_card("CO2 Accountability Insight", prompt_co2, 0.4)

        # 1B. Insert Pill Heading and Plot Container
        st.markdown(f"""
            <span class="pill-heading">CO2 Emissions / Capita (t)</span>
            <div class="plot-wrap fade-in-element" style="--delay: 0.7s;">
            """, unsafe_allow_html=True)
            
        # Placeholder for actual CO2 Gauge rendering
        co2_goal = 2.0
        co2_gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=co2_pc_val,
            
            # CRITICAL FIX 1: Use the domain y-range to push the gauge arc lower
            domain={'x': [0, 1], 'y': [0.25, 1.0]}, # Gauge arc starts lower on the vertical axis (y=0.25)
            
            # CRITICAL FIX 2: Remove Plotly's default automatic title/number positioning
            title={'text': '', 'font': {'size': 0}}, 
            number={'font': {'color': 'var(--muted)', 'size': 50}}, 
            
            gauge = {
                'axis': {'range': [0, 4.0]}, 
                'bar': {'color': co2_color}, 
                'steps': [{'range': [0, 1.5], 'color': '#38a3a5'}, {'range': [2.5, 4.0], 'color': '#e63946'}],
                'threshold': {'value': co2_goal}
            }
        ))
        
        co2_gauge_fig.update_layout(
            height=450, 
            # CRITICAL FIX 3: Minimize all padding
            margin=dict(l=10, r=10, t=5, b=5), 
            paper_bgcolor='white', 
            plot_bgcolor='white'
        ) 
        st.plotly_chart(co2_gauge_fig, use_container_width=True)
        st.caption(f"Current value: {co2_pc_val:.1f} t. Goal: < {co2_goal} t.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        
    # --- 2. Monthly Temperature Norms ---
    with col2:
        # 2A. Insert AI Analysis Card (This works and adds intelligence)
        # ... (code for AI card remains)
        
        # 2B. Insert Pill Heading and Plot Container (This works)
        st.markdown(f"""
        <span class="pill-heading" style="margin-top: 15px;">Monthly Temperature Norms</span>
        <div class="plot-wrap fade-in-element" style="--delay: 0.7s;">
        """, unsafe_allow_html=True)
        
        # 2C. Chart Plotting Logic (The actual Plotly output)
        if not df_clim_in.empty:
            df_temp_monthly = df_clim_in.copy()
            df_temp_monthly['month'] = df_temp_monthly['time'].dt.month
            df_temp_monthly['Month_Name'] = df_temp_monthly['time'].dt.strftime('%b')
            
            df_monthly_agg = df_temp_monthly.groupby(['month', 'Month_Name']).agg(
                Avg_Max_Temp=('temperature_2m_max', 'mean'),
                Avg_Min_Temp=('temperature_2m_min', 'mean'),
            ).reset_index().sort_values('month')
            
            fig_temp_monthly = go.Figure(
                data=[
                    go.Bar(name='Avg_Max_Temp', x=df_monthly_agg['Month_Name'], y=df_monthly_agg['Avg_Max_Temp'], marker_color='#e63946'), 
                    go.Bar(name='Avg_Min_Temp', x=df_monthly_agg['Month_Name'], y=df_monthly_agg['Avg_Min_Temp'], marker_color='#007bff')
                ]
            )
            fig_temp_monthly.update_layout(
                barmode='group',
                title='*Monthly Temperature Norms (Avg Max | Avg Min)*',
                yaxis_title='Temp (°C)',
                height=450,
                margin=dict(l=10, r=10, t=50, b=10), # Uses t=50 to push down for alignment
                plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'),
                legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=1)
            )
            st.plotly_chart(fig_temp_monthly, use_container_width=True)
        else:
            st.info("Insufficient climate data to plot monthly temperature norms.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 3. Total Annual Precipitation (mm) ---
    with col3:
        # ... (Precipitation logic remains the same, using height=450 and t=50 for alignment) ...
        df_precip_annual_agg = df_clim_in.copy()
        if not df_precip_annual_agg.empty:
            df_precip_annual_agg['year'] = df_precip_annual_agg['time'].dt.year
            df_precip_annual_agg = df_precip_annual_agg.groupby('year')['precipitation_sum'].sum().reset_index()
            recent_trend = df_precip_annual_agg['precipitation_sum'].diff().iloc[-1] if len(df_precip_annual_agg) > 1 else 0.0
        else:
            recent_trend = 0.0
        
        # 3A. Insert AI Analysis Card
        prompt_precip = f"For {_name}, the annual rainfall shows a change of {recent_trend:.1f} mm in the last recorded year. Is this trend toward drought or flooding, and what infrastructure challenge does it pose?"
        render_ai_analysis_card("Precipitation Trend Context", prompt_precip, 0.6) # This is the top AI insight card
        
        # 3B. Insert Pill Heading and Plot Container
        st.markdown(f"""
            <span class="pill-heading">Annual Precipitation Distribution</span>
            <div class="plot-wrap fade-in-element" style="--delay: 0.7s;">
            """, unsafe_allow_html=True)
        
        # 3C. Total Annual Precipitation (PIE CHART)
        if not df_clim_in.empty:
            custom_colors = ['red', 'blue', 'green', 'yellow', 'pink', 'deepskyblue', 'orange']
            fig_precip_pie = px.pie(
                df_precip_annual_agg,
                names='year', # Use year as labels
                values='precipitation_sum', # Use precipitation as values
                title='*Percentage Share of Total Precipitation*',
                color_discrete_sequence=custom_colors, 
                hole=0.4 
            )
            
            fig_precip_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=1))) 
            fig_precip_pie.update_layout(
                height=450,
                margin=dict(l=10, r=10, t=50, b=10), # Uses t=50 to push down for alignment
                plot_bgcolor='white', 
                paper_bgcolor='white', 
                font=dict(color='#212529'),
                showlegend=True
            )
            st.plotly_chart(fig_precip_pie, use_container_width=True)
            
        else:
            st.info("Insufficient climate data to plot annual precipitation.")
            
        st.markdown("</div>", unsafe_allow_html=True) # Close plot-wrap container
        
# --- NEW AI FUNCTION TO FETCH LATEST SWACHH WINNER ---
@st.cache_data(ttl=3600) # Cache for 1 hour to reduce API calls
def fetch_latest_swachh_winner_ai():
    """Fetches the latest declared winners for the Swachh Survekshan (India's Cleanest City)."""
    # NOTE: GEMINI_API_KEY is defined at the top of the script.
    if not GEMINI_API_KEY:
        return "Indore & Surat (2025)" # Fallback to hardcoded
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Prompt Gemini to extract the specific information
    prompt = (
        "What are the official results for the latest Swachh Survekshan (Cleanest City in India) "
        "declared for cities with population > 1 Lakh? List the primary winner(s) and the relevant survey year/date."
        "Respond ONLY with the name of the winner(s) and the survey year/edition in the format: 'City1 & City2 (Year)'"
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        # Clean up the response to ensure only the city/year format is returned
        clean_text = response.text.strip().replace('*', '')
        if clean_text:
            return clean_text
        return "Indore & Surat (2023)" # Safe fallback if AI gives verbose response
        
    except Exception:
        return "Indore & Surat (2023)" # Safe fallback on API error


# --- NEW HELPER FUNCTION FOR THE BADGE (Needed to access the dynamic text) ---
def render_swachh_star_badge():
    """Renders the dynamic Swachh Star City badge."""
    # This calls the AI-enhanced function
    star_city_text = fetch_latest_swachh_winner_ai()
    
    st.markdown(f"""
    <div class="fade-in-element" style="--delay: 0.2s; text-align: center; background: #f8f9fa; border-radius: 8px; padding: 5px; border: 1px solid var(--brand2);">
    	<p style='color: #212529; font-size: 0.8rem; margin: 0;'>
    		<span class='flicker-crown-icon'>👑</span> Swachh Star City:
    	</p>
    	<p style='color: gold; font-weight: 700; font-size: 1.0rem; margin: 0; text-shadow: 0 0 5px orange;'>
    		{star_city_text}
    	</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------ Sidebar Controls (UPDATED) ------------------------------

# --- 1. Sidebar Header (Updated with new component styling) ---
with st.sidebar:
    st.markdown("""
    <div class='sidebar-icon-container'>
        <span class='big-earth-icon'>🌎</span>
        <span class='big-tree-icon'>🌳</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"## 🌿 SustainifyAI", 
        unsafe_allow_html=True
    )
    st.caption("Vibrant climate intelligence dashboard — real data, forecasts & insights")

    # Logout button in sidebar (placed after the login is successful)
    if st.button("🚪 Logout"):
        st.session_state["auth_ok"] = False
        st.session_state["auth_user"] = None
        st.rerun()

    st.markdown("### 🔍 Location & Date Range")
    place_default = st.text_input("Search a city / place", value="Prayagraj")
    
    # Retain the geocoding logic for getting lat/lon
    geo = geocode_place(place_default) if place_default.strip() else None

    if geo is None:
        st.error("Couldn't geocode the place. Try a larger city or correct spelling.")
        st.stop()

    lat, lon, _name, _country = geo
    st.session_state['lat'] = lat
    st.session_state['lon'] = lon
    st.success(f"📍 {_name}, {_country} | {lat:.3f}, {lon:.3f}")

    today = dt.date.today()
    start_date = st.date_input("📅 Start date", value=today - dt.timedelta(days=365*5))
    end_date = st.date_input("📅 End date", value=today)


    st.markdown("### 🤖 AI Forecasting Model")
    model_choice = st.selectbox(
        "Forecast model",
        ["auto", "Prophet", "ARIMA", "ML Ensemble"],
        index=0,
        help="Choose the AI model: Prophet is great for strong seasonality (e.g., yearly temps); ARIMA is a classic statistical model; ML Ensemble (Random Forest) is a non-linear fallback.",
        key="sidebar_forecast_model" # <-- CORRECTED: ADDED UNIQUE KEY
    )

    st.markdown("### 🚨 Alert Thresholds")
    alert_pm25 = st.slider("PM2.5 alert threshold (ug/m3)", 10, 200, 90, help="If the current PM2.5 (Air Quality) exceeds this threshold, a warning alert will be triggered on the dashboard.")
    alert_temp = st.slider("Max temp alert (deg C)", 30, 50, 44, help="If the latest recorded maximum temperature exceeds this threshold, a heat warning alert will be triggered.")

    st.markdown("### ♻ Sustainability Index Inputs")
    with st.expander("Customize Sustainability Scoring"):
        co2_pc = st.number_input("🟫 CO₂ per capita (tons)", min_value=0.0, value=1.9, step=0.1, key="co2_input") # Added key
        ren_share = st.slider("🔋 Renewable energy share (%)", 0, 100, 35, key="ren_share_slider") # Added key
        water_idx = st.slider("💧 Water quality index (%)", 0, 100, 65, key="water_idx_slider") # Added key
        recycle = st.slider("🗑 Waste recycling rate (%)", 0, 100, 30, key="recycle_slider") # Added key

    st.markdown("---")
    st.caption("Built with 💚 by Nxt Zen Developers— SustainifyAI 2025")
# ------------------------------ Sidebar Toggle (OPTIONAL JS - retained) ------------------------------
# Optional JS to manually toggle the sidebar for a more custom slide effect demo
toggle_html = """
<script>
(function() {
  const sidebar = window.parent.document.querySelector('.stSidebar'); 
  if (!sidebar) return;
  const stApp = window.parent.document.querySelector('.stApp'); 
  
  // Check the initial state managed by Streamlit
  let is_expanded = stApp.classList.contains('st-ef') || stApp.classList.contains('st-e8'); // Common classes for expanded state
  
  const btn = document.createElement('button');
  btn.innerHTML = '☰';
  btn.style.position = 'fixed';
  btn.style.top = '10px';
  btn.style.left = '10px';
  btn.style.zIndex = '1000';
  btn.style.background = '#38a3a5';
  btn.style.color = 'white';
  btn.style.border = 'none';
  btn.style.borderRadius = '5px';
  btn.style.padding = '10px 14px';
  btn.style.cursor = 'pointer';
  
  // Add JS-controlled styles for better interaction
  if(is_expanded) {
    // Initial position for the expanded state (Streamlit default)
  } else {
    // If initially collapsed, position the button appropriately
    btn.style.left = '10px';
  }
  
  document.body.appendChild(btn);

  btn.onclick = () => {
    // Streamlit's internal sidebar toggle is complex. 
    // We try to trigger Streamlit's native collapse action instead of overriding CSS transforms, 
    // but Streamlit components don't directly access the internal collapse button. 
    // We stick to CSS override for the "sliding" visual effect on hover.
    // The following code is for demonstration, relying on Streamlit's initial state for the visual effect.
    const sidebar_button = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (sidebar_button) {
        sidebar_button.click();
    } else {
        // Fallback or just let the CSS hover effect handle the "slide"
    }
  };
})();
</script>
"""
#components.html(toggle_html, height=0, width=0) # Commented out as it causes Streamlit runtime issues

# ------------------------------ Data Pulls ------------------------------
with st.spinner("Fetching climate history (Open-Meteo ERA5)…"):
    try:
        df_clim = fetch_openmeteo_daily(lat, lon, start_date, end_date)
    except Exception as e:
        st.error(f"Open-Meteo fetch failed: {e}")
        st.stop()

with st.spinner("Fetching latest air quality (Open-Meteo AQ)…"):
    df_aq = fetch_air_quality_current(lat=lat, lon=lon)
    
# --- NEW WASTE DATA PULL ---
with st.spinner("Fetching solid waste data (Simulated)…"):
    df_msw = get_solid_waste_data(_name)

# ------------------------------ Header (MODIFIED to use brand_banner) ------------------------------
colA, colB = st.columns([0.7,0.3])
with colA:
    # --- NEW: brand_banner Call ---
    brand_banner(
        title="SustainifyAI — Sustainability & Climate Tracker",
        subtitle="Real-time environmental data • Predictive AI • Governance tools",
        badges=["CO₂ Insights", "Air Quality", "Water Stress", "Net-Zero Roadmaps"],
        ctas=[("Launch App", "primary"), ("Learn More", "alt")],
        icon="🌎",
    )

with colB:
    st.markdown("<div class='glass-card-header'>" , unsafe_allow_html=True)
    st.metric("Location", f"{_name}, {_country}")
    st.metric("Period", f"{start_date:%d %b %Y} → {end_date:%d %b %Y}")
    
    # --- Swachh Survekshan Rank Mirror (Crown REMOVED) ---
    rank_df = get_swachh_ranking_data([_name], _name)
    current_rank = rank_df[rank_df['City'] == _name].get('2025', pd.Series([np.nan])).iloc[0] if not rank_df.empty else np.nan
    rank_display = f"Rank: {int(current_rank)}" if not np.isnan(current_rank) else "Rank: N/A"

    st.markdown(f"""
    <div class='rank-mirror-button'>
    	Swachh Rank: {rank_display}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ------------------------------ NEWS TICKER PIPE & SWACHH STAR (Updated Colors) ------------------------------
col_news, col_swachh_star = st.columns([0.8, 0.2])

with col_news:
    st.markdown(
        f"""
        <div class="news-pipe-container fade-in-element" style="--delay: 0.1s;">
            <div class="news-pipe-content">{get_pollution_news_ticker(_name)}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ... (near the NEWS TICKER section) ...

with col_swachh_star:
    # --- SWACCH STAR CITY BADGE (NOW USES GEMINI API) ---
    render_swachh_star_badge() # Call the new AI-enhanced renderer
    # ---------------------------------------------------


# ------------------------------ KPIs (Updated Symbols and Explanations) ------------------------------

# Extract values for cleaner use
pm25_now = float(df_aq.loc[df_aq["parameter"]=="pm2_5", "value"].head(1).fillna(np.nan).values[0]) if not df_aq.empty and (df_aq["parameter"]=="pm2_5").any() else np.nan
mean_temp = df_clim['temperature_2m_mean'].mean()
max_wind = df_clim['windspeed_10m_max'].max()
total_rain = df_clim['precipitation_sum'].sum()
total_solar = df_clim['shortwave_radiation_sum'].sum() # NEW KPI VALUE

# --- NEW WASTE KPI VALUES (FIXED with robust fallback) ---
city_msw_data = df_msw[df_msw['City'].str.lower() == _name.lower()].iloc[0] if not df_msw.empty and (df_msw['City'].str.lower() == _name.lower()).any() else None
current_msw_tpd = float(city_msw_data['Total_MSW_TPD']) if city_msw_data is not None else 600.0 # Robust Fallback
predicted_msw_change = float(city_msw_data['Predicted_TPD_Change_%']) if city_msw_data is not None else 10.0 # Robust Fallback


# --- KPI Definitions with Emojis and Simplified Explanation for a 12-year-old ---
kpi_data = [
    {
        "label": "PM2.5 (ug/m3) 💨", 
        "value": ("-" if math.isnan(pm25_now) else f"{pm25_now:.1f}"), 
        "id_key": "pm25_kpi",
        "help": "This is tiny dust and smoke particles in the air. The lower the number, the cleaner the air is for your lungs!"
    },
    {
        "label": "Mean Temp (deg C) 🌡️", 
        "value": f"{mean_temp:.1f}", 
        "id_key": "mean_temp_kpi",
        "help": "The average temperature across the time period. This helps us see if the weather is generally too hot or too cold."
    },
    {
        "label": "<span class='rotating-windmill'>⚙️</span> Max Wind (m/s) 🌬️", # Use gear/windmill for rotation effect
        "value": f"{max_wind:.1f}", 
        "id_key": "max_wind_kpi",
        "help": "The fastest wind speed recorded. Strong winds are important for air circulation and generating wind power."
    },
    # --- NEW KPI INSERTED HERE ---
    {
        "label": "Pollution Rank 🏆", 
        "value": np.nan,
        "id_key": "pollution_rank_kpi",
        "custom_render": "rank", # Flag for custom rendering
        "help": "Simulated rank based on current PM2.5 compared to other major cities. Lower number (closer to 1) is better. Target is Top 30."
    },
    # --- END NEW KPI ---
    {
        "label": "Total Rain (mm) 💧", 
        "value": f"{total_rain:.1f}", 
        "id_key": "total_rain_kpi",
        "help": "The total amount of water (rain) that fell during the selected time. We need rain for clean drinking water and for our crops to grow!"
    },
    {
        "label": "Total Solar Rad (MJ/m²) ☀️", 
        "value": f"{total_solar:.1f}", 
        "id_key": "total_solar_kpi",
        "help": "The total solar energy that reached the ground. High solar radiation is essential for solar power generation."
    },
    # --- Waste KPI MOVED to the end ---
    {
        "label": "Total Waste (TPD) 🗑️",
        "value": f"{current_msw_tpd:.0f}",
        "predicted_change": predicted_msw_change,
        "id_key": "total_waste_kpi",
        "custom_render": "waste", # Flag for custom rendering
        "help": "The total Tons Per Day (TPD) of solid waste generated in the city. Lower is better for a cleaner city!"
    },
]

# --- NEW FUNCTION: Render the Pollution Rank KPI ---
def render_rank_kpi(col, data, delay, current_city_name, current_pm25):
    """Renders the Pollution Rank KPI with dynamic colors based on the Top 30 goal."""
    
    # 1. Get the dynamic rank data
    if math.isnan(current_pm25):
        rank_text = "-"
        rank_subtitle = "Data Unavailable"
        color = "#6c757d" # Muted Gray
    else:
        rank, subtitle = get_pollution_rank_proxy(current_city_name, current_pm25)
        rank_text = str(rank)
        rank_subtitle = f"PM2.5 Rank: {subtitle}"
        
        # 2. Apply Dynamic Styling (Top 30 is the goal)
        if rank <= 30:
            color = "#38a3a5" # Teal/Green for Good
            icon = "✅"
            css_class = "metric-card animated-card-good"
        elif rank <= 70:
            color = "#ffc107" # Amber for Moderate
            icon = "🟡"
            css_class = "metric-card animated-card-moderate"
        else:
            color = "#e63946" # Red for Critical
            icon = "❌"
            css_class = "metric-card animated-card-critical" # Enhanced animation for critical state
    
    with col:
        st.markdown(
            f"""
            <div class='{css_class} fade-in-element gradient-border' style='--delay: {delay}s; --border-color: {color};'>
                <div class='kpi-label'><span class='animated-icon' style='color: {color}; font-size: 1.2em;'>{icon}</span> {data['label']}</div>
                <div class='kpi-value glowing-text' style='color: {color} !important; font-size: 2.5rem;'>
                    #{rank_text}
                </div>
                <p class='sliding-text' style='color: var(--muted); font-size: 0.8rem; margin-top: 5px; margin-bottom: 0;'>
                    {rank_subtitle}
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Manually create the tooltip text without generating the visible metric element
        st.markdown(f"<p style='visibility: hidden; height: 1px;' data-testid='stMetric' title='{data['help']}'>{data['label']}</p>", unsafe_allow_html=True)

# --- NEW FUNCTION: Render the Waste Alert KPI with custom styling ---
def render_waste_alert_kpi(col, data, delay):
    # Ensure current_tpd is a string for f-string formatting inside HTML
    current_tpd = str(data['value']) 
    change = data.get('predicted_change', 0.0)
    
    # Enhanced color logic and animation classes
    if change > 0:
        icon_html = f"<span class='animated-increase'>▲</span>" # Red UP arrow with animation
        trend_msg = f"Predicted Increase: {change:.1f}%"
        card_class = "metric-card animated-card-critical pulsing-border"
        value_class = "kpi-waste-value glowing-text"
        alert_color = "#e63946" # Red for warning
    elif change < 0:
        icon_html = f"<span class='animated-decrease'>▼</span>" # Green DOWN arrow with animation
        trend_msg = f"Predicted Decrease: {abs(change):.1f}%"
        card_class = "metric-card animated-card-good shimmer"
        value_class = "kpi-value"
        alert_color = "#38a3a5" # Teal for good
    else:
        icon_html = ""
        trend_msg = "Stable Prediction"
        card_class = "metric-card animated-card-moderate"
        value_class = "kpi-value"
        alert_color = "#ffc107" # Amber for neutral

    with col:
        # Enhanced card with animations and dynamic border effects
        st.markdown(
            f"""
            <div class='{card_class} fade-in-element gradient-border' style='--delay: {delay}s; --border-color: {alert_color};'>
                <div class='kpi-label'>
                    <span class='waste-icon animated-icon rotating'>🗑️</span> 
                    {data['label'].split(" ")[0]} TPD
                </div>
                <div class='{value_class}' style='color: {alert_color} !important;'>
                    {current_tpd} {icon_html}
                </div>
                <p class='sliding-text' style='color: var(--muted); font-size: 0.8rem; margin-top: 5px; margin-bottom: 0;'>
                    {trend_msg}
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Manually create the tooltip text without generating the visible metric element
        st.markdown(f"<p style='visibility: hidden; height: 1px;' data-testid='stMetric' title='{data['help']}'>{data['label']}</p>", unsafe_allow_html=True)

# Render KPIs
kpi_cols = st.columns(len(kpi_data)) # <--- MODIFICATION: Dynamically set to 7 columns

for i, data in enumerate(kpi_data):
    delay = 0.3 + i * 0.1
    
    if data.get("custom_render") == "rank": # <--- MODIFICATION: New Rank check
        # Custom Render the Pollution Rank KPI
        render_rank_kpi(kpi_cols[i], data, delay, _name, pm25_now)
        
    elif data.get("custom_render") == "waste": # <--- MODIFICATION: Waste is the 7th item
        # Custom Render the Waste KPI
        render_waste_alert_kpi(kpi_cols[i], data, delay)
        
    else:
        with kpi_cols[i]:
            # Enhanced Standard KPI Rendering with Animations
            st.markdown(
                f"""
                <div class='metric-card animated-card-moderate fade-in-element gradient-border' 
                    style='--delay: {delay}s; padding-top:10px; --border-color: var(--brand);'>
                    <div class='kpi-label animated-icon'>{data['label']}</div>
                    <div class='kpi-value glowing-text'>{data['value']}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Standard st.metric call for Tooltip/Layout purposes
            display_label = data['label'].split('>')[-1].replace('</span>', '').strip() if '<span' in data['label'] else data['label']
            st.metric(display_label, "", help=data['help'])

# Alerts
alerts = []
if not math.isnan(pm25_now) and pm25_now >= alert_pm25:
    alerts.append(f"⚠ High PM2.5 detected: {pm25_now:.1f} ug/m3 >= threshold {alert_pm25}")

if not df_clim.empty:
    latest_max_temp = float(df_clim["temperature_2m_max"].iloc[-1])
    if latest_max_temp >= alert_temp:
        alerts.append(f"🔥 *CURRENT HEAT ALERT:* Latest max temperature of {latest_max_temp:.1f} deg C exceeded threshold {alert_temp} deg C.")
else:
    alerts.append("ℹ Climate data not loaded, temperature alert is inactive.")

if alerts:
    # Enhanced Alert Display
    st.markdown(
        f"""
        <div class='metric-card animated-card-critical fade-in-element gradient-border' 
            style='--delay: 0.5s; padding: 15px; --border-color: #e63946;'>
            <div class='animated-icon' style='font-size: 1.5rem; color: #e63946; margin-bottom: 10px;'>⚠️</div>
            <div class='sliding-text' style='color: #e63946; font-weight: 600;'>
                {" <br> ".join(alerts)}
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    # send_telegram is disabled for public code submission

# ------------------------------ Tabs (COMPLETED) ------------------------------

# Function Definition (UPDATE SIGNATURE TO INCLUDE _name)
@st.cache_data(show_spinner=False)
def generate_prototype_graphs(df_clim_in: pd.DataFrame, co2_pc_val: float, _name: str):
    """Generates the three graphs and their AI-driven analysis cards."""
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate colors (assuming get_gauge_color is defined)
    co2_color = get_gauge_color(co2_pc_val, 1.5, 2.5, reverse=True) 
    
    # --- 1. CO2 Emissions / Capita (t) ---
    with col1:
        # 1A. Insert AI Analysis Card
        prompt_co2 = f"Analyze the current CO2 per capita reading of {co2_pc_val:.1f} tons for {_name} against the global 2.0t target. Is this score high, moderate, or low risk? Propose one immediate action area."
        render_ai_analysis_card("CO2 Accountability Insight", prompt_co2, 0.4)

        st.markdown("""
            <div class="fade-in-element plot-wrap" style="--delay: 0.1s; height: 350px;">
                <h3 class="animated-sub-header" style="font-size: 1.1rem !important; margin-top: 5px; margin-bottom: 5px;">CO2 Emissions / Capita (t)</h3>
            </div>
            """, unsafe_allow_html=True)
        # Placeholder for actual CO2 Gauge rendering (must be added back by you if missing)
        co2_goal = 2.0
        co2_gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=co2_pc_val,
            gauge = {'axis': {'range': [0, 4.0]}, 'bar': {'color': co2_color}, 
                     'steps': [{'range': [0, 1.5], 'color': '#38a3a5'}, {'range': [2.5, 4.0], 'color': '#e63946'}],
                     'threshold': {'value': co2_goal}}
        ))
        co2_gauge_fig.update_layout(height=336, margin=dict(l=10, r=10, t=40, b=10)) 
        st.plotly_chart(co2_gauge_fig, use_container_width=True)
        st.caption(f"Current value: {co2_pc_val:.1f} t. Goal: < {co2_goal} t.", unsafe_allow_html=True)


    # --- 2. Monthly Temperature Norms ---
    with col2:
        # 2A. Insert AI Analysis Card (This works and adds intelligence)
        df_monthly_agg = df_clim_in.copy()
        max_temp_val = df_monthly_agg['temperature_2m_max'].max() if not df_monthly_agg.empty else 'N/A'
        
        prompt_temp = f"The recorded temperature for {_name} shows a maximum mean of {max_temp_val}°C. What are the key health and agricultural risks associated with this temperature peak during the hot season in this region?"
        render_ai_analysis_card("Seasonal Risk Interpretation", prompt_temp, 0.5)
        
        # 2B. Insert Chart Title and Container
        # Note: We use a simple markdown for the title and a general container for the plot to avoid premature closing of tags.
        st.markdown(f"""
    <span class="pill-heading" style="margin-top: 15px;">Monthly Temperature Norms</span>
""", unsafe_allow_html=True)
            
        st.markdown("""
            <div class="plot-wrap fade-in-element" style="--delay: 0.7s;">
            """, unsafe_allow_html=True)
        
        # 2C. Chart Plotting Logic (The actual Plotly output)
        if not df_clim_in.empty:
            df_temp_monthly = df_clim_in.copy()
            df_temp_monthly['month'] = df_temp_monthly['time'].dt.month
            df_temp_monthly['Month_Name'] = df_temp_monthly['time'].dt.strftime('%b')
            
            df_monthly_agg = df_temp_monthly.groupby(['month', 'Month_Name']).agg(
                Avg_Max_Temp=('temperature_2m_max', 'mean'),
                Avg_Min_Temp=('temperature_2m_min', 'mean'),
            ).reset_index().sort_values('month')
            
            fig_temp_monthly = go.Figure(
                data=[
                    go.Bar(name='Avg_Max_Temp', x=df_monthly_agg['Month_Name'], y=df_monthly_agg['Avg_Max_Temp'], marker_color='#e63946'), 
                    go.Bar(name='Avg_Min_Temp', x=df_monthly_agg['Month_Name'], y=df_monthly_agg['Avg_Min_Temp'], marker_color='#007bff')
                ]
            )
            fig_temp_monthly.update_layout(
                barmode='group',
                title='*Monthly Temperature Norms (Avg Max | Avg Min)*',
                yaxis_title='Temp (°C)',
                height=336, 
                plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'),
                legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=1)
            )
            st.plotly_chart(fig_temp_monthly, use_container_width=True)
        else:
            st.info("Insufficient climate data to plot monthly temperature norms.")
            
        st.markdown("</div>", unsafe_allow_html=True)

   # --- 3. Total Annual Precipitation (mm) ---
    with col3:
        # --- Data Aggregation (Consolidated) ---
        df_precip_annual_agg = df_clim_in.copy()
        if not df_precip_annual_agg.empty:
            df_precip_annual_agg['year'] = df_precip_annual_agg['time'].dt.year
            df_precip_annual_agg = df_precip_annual_agg.groupby('year')['precipitation_sum'].sum().reset_index()
            recent_trend = df_precip_annual_agg['precipitation_sum'].diff().iloc[-1] if len(df_precip_annual_agg) > 1 else 0.0
        else:
            recent_trend = 0.0
        
        # 3A. Insert AI Analysis Card
        prompt_precip = f"For {_name}, the annual rainfall shows a change of {recent_trend:.1f} mm in the last recorded year. Is this trend toward drought or flooding, and what infrastructure challenge does it pose?"
        render_ai_analysis_card("Precipitation Trend Context", prompt_precip, 0.6) # This is the top AI insight card
        
        # 3B. Insert Pill Heading and Plot Container
        st.markdown(f"""
            <span class="pill-heading">Annual Precipitation Distribution</span>
            <div class="plot-wrap fade-in-element" style="--delay: 0.7s;">
            """, unsafe_allow_html=True)
        
        # 3C. Total Annual Precipitation (PIE CHART WITH CUSTOM BRIGHT COLORS)
        if not df_clim_in.empty:
            
            # --- Defining the Custom Color Palette ---
            custom_colors = ['red', 'blue', 'green', 'yellow', 'pink', 'deepskyblue', 'orange']
            
            # Generating the Pie Chart (Donut Style)
            fig_precip_pie = px.pie(
                df_precip_annual_agg,
                names='year', # Use year as labels
                values='precipitation_sum', # Use precipitation as values
                title='*Percentage Share of Total Precipitation*',
                # --- CRITICAL CHANGE: Using the custom discrete color sequence ---
                color_discrete_sequence=custom_colors, 
                hole=0.4 # Makes it a donut chart
            )
            
            fig_precip_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=1))) # Added white line for better slice separation
            fig_precip_pie.update_layout(
                height=336, 
                plot_bgcolor='white', 
                paper_bgcolor='white', 
                font=dict(color='#212529'),
                showlegend=True
            )
            st.plotly_chart(fig_precip_pie, use_container_width=True)
            
        else:
            st.info("Insufficient climate data to plot annual precipitation.")
            
        st.markdown("</div>", unsafe_allow_html=True) # Close plot-wrap container

TAB_OVERVIEW, TAB_AIR, TAB_TRENDS, TAB_FORECAST, TAB_STORY, TAB_SCORE, TAB_CARBON, TAB_GREEN_INFRA, TAB_SWACHH, TAB_WASTE, TAB_ABOUT = st.tabs([
    "🌍 Overview", "💨 Air Quality", "📈 Climate Trends", "📊 Forecasts", "💥 Impact Story", "💯 Sustainability Score", "🧍 Personal Carbon", "🔋 Green Infra & EVs", "🧼 Cleanliness Rank", "🗑 Waste Management", "🚀 About"
])

# --- OVERVIEW TAB CONTENT (FIXED: Ensuring full content rendering) ---
with TAB_OVERVIEW:
    
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🌍</span>
                Overview: Core Environmental Metrics
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # --- NEW: PROTOTYPE GRAPHS INSERTED HERE ---
    generate_prototype_graphs(df_clim, co2_pc, _name)
    
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    
    col_waste_trend, col_chat = st.columns([0.6, 0.4])

    with col_waste_trend:
        
        # --- NEW: PLASTIC WASTE TREND LINE CHART ---
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.3s;">
                <h3 class="animated-sub-header">
                    <span class="animated-emoji" style="--delay: 0.4s;">🧴</span> Plastic Waste Trend (TPD) & Forecast
                </h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        df_plastic = get_plastic_waste_trend(_name)
        
        fig_plastic = go.Figure()
        
        # Historical Data (Blue)
        df_hist = df_plastic[df_plastic['Year'] <= today.year]
        fig_plastic.add_trace(go.Scatter(
            x=df_hist['ds'], 
            y=df_hist['y'], 
            mode='lines+markers', 
            name='Historical Data (TPD)',
            line=dict(color='#007bff', width=3), # Primary Blue
            marker=dict(size=8, symbol='circle')
        ))

        # Forecast Data (Yellow/Amber Dashed)
        df_fcst = df_plastic[df_plastic['Year'] > today.year]
        fig_plastic.add_trace(go.Scatter(
            x=df_fcst['ds'], 
            y=df_fcst['y'], 
            mode='lines+markers', 
            name='Forecasted TPD',
            line=dict(color='#ffc107', width=3, dash='dash'), # Amber/Yellow
            marker=dict(size=8, symbol='diamond')
        ))
        
        # ANNOTATE SPIKES/ANOMALIES
        annotations = []
        for _, row in df_plastic[df_plastic['Anomaly'] == True].iterrows():
            annotations.append(dict(
                x=row['ds'], y=row['y'],
                xref="x", yref="y",
                text=f"⚠️ Anomaly: {row['y']:.0f} TPD",
                showarrow=True, arrowhead=7, ax=0, ay=-40,
                font=dict(color='#e63946', size=12),
                arrowcolor='#e63946', arrowwidth=2
            ))
            
            fig_plastic.update_layout(
            title='*Plastic Waste Generation in {} (TPD)*'.format(_name),
            xaxis_title="Year",
            yaxis_title="Plastic Waste (TPD - Tons Per Day)",
            height=480, # Increased height by 20%
            hovermode="x unified",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            annotations=annotations,
            # ADDED: Gridlines
            xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )
        
        st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
        st.plotly_chart(fig_plastic, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption("Forecast shows expected plastic waste based on historical trend and city's current growth rate. Spikes indicate unusual generation events.")


    with col_chat:
        # --- CHATBOT (MOVED UP HERE) ---
        chat_interface_embed()
        
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    # ------------------------------ SECONDARY VISUALIZATIONS (Bottom Row) ------------------------------
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 2.0s;">
            <h3 class="animated-sub-header">
                <span class="animated-emoji" style="--delay: 2.1s;">📈</span> Seasonality Anomaly Tracker (YoY Change)
            </h3>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Prepare Annual Data for YoY comparison (last 5 years)
    df_annual = df_clim.copy()
    df_annual['year'] = df_annual['time'].dt.year
    df_annual = df_annual.groupby('year').agg(
        Annual_Mean_Temp=('temperature_2m_mean', 'mean'),
        Annual_Precipitation=('precipitation_sum', 'sum')
    ).reset_index()

    # Calculate Year-over-Year change (excluding the first year)
    df_annual['YoY_Temp_Change'] = df_annual['Annual_Mean_Temp'].diff()
    df_annual['YoY_Precip_Change'] = df_annual['Annual_Precipitation'].diff()

    # Filter for the last 5 relevant years (Need at least 2 points for YoY change, so filter last 6 if possible)
    start_year = df_annual['year'].max() - 4 # Show last 5 years
    df_yoy = df_annual[df_annual['year'] >= start_year].copy()
    
    # Create the beautiful line plot
    fig_yoy = go.Figure()

    # Trace 1: Temperature Change (Red/Orange - Heat)
    temp_data = df_yoy[df_yoy['YoY_Temp_Change'].notna()]
    fig_yoy.add_trace(go.Scatter(
        x=temp_data['year'], 
        y=temp_data['YoY_Temp_Change'], 
        mode='lines+markers', 
        name='Annual Mean Temp Change (deg C)',
        line=dict(color='#e63946', width=4), # Bold Red
        marker=dict(size=10, symbol='circle', line=dict(width=2, color='White')),
        yaxis='y1'
    ))

    # Trace 2: Precipitation Change (Blue - Water)
    precip_data = df_yoy[df_yoy['YoY_Precip_Change'].notna()]
    fig_yoy.add_trace(go.Scatter(
        x=precip_data['year'], 
        y=precip_data['YoY_Precip_Change'], 
        mode='lines+markers', 
        name='Annual Precipitation Change (mm)',
        line=dict(color='#007bff', width=4, dash='dash'), # Primary Blue, Dashed
        marker=dict(size=10, symbol='diamond', line=dict(width=2, color='White')),
        yaxis='y2' # Use secondary Y-axis
    ))

    # Add a zero line for reference
    fig_yoy.add_hline(y=0, line_dash="dot", line_color="#343a40", opacity=0.5, annotation_text="Normal Baseline (No Change)")

    # --- ANOMALY DETECTION AND ANNOTATION LOGIC (NEW) ---
    annotations = []
    TEMP_THRESHOLD = 1.0 # 1 degree C shift is considered a significant anomaly
    RAIN_THRESHOLD = 250.0 # 250 mm shift is considered a significant anomaly

    for _, row in df_yoy.dropna(subset=['YoY_Temp_Change', 'YoY_Precip_Change']).iterrows():
        year = row['year']
        
        # 1. Temperature Anomaly Check
        temp_change = row['YoY_Temp_Change']
        if abs(temp_change) >= TEMP_THRESHOLD:
            reason = "Extreme Warming Spike" if temp_change > 0 else "Unusual Cooling"
            color = '#e63946'
            annotations.append(dict(
                x=year, y=temp_change,
                xref="x", yref="y1",
                text=f"🔥 {reason}: +{temp_change:.1f}°C YoY",
                showarrow=True, arrowhead=7, ax=0, ay=-40 if temp_change > 0 else 40,
                font=dict(color=color, size=12, family="Arial, sans-serif"),
                arrowcolor=color, arrowwidth=2
            ))

        # 2. Precipitation Anomaly Check
        precip_change = row['YoY_Precip_Change']
        if abs(precip_change) >= RAIN_THRESHOLD:
            reason = "Heavy Monsoon Spike" if precip_change > 0 else "Severe Drought/Rainfall Deficit"
            color = '#007bff'
            annotations.append(dict(
                x=year, y=precip_change,
                xref="x", yref="y2",
                text=f"💧 {reason}: +{precip_change:.0f}mm YoY",
                showarrow=True, arrowhead=7, ax=0, ay=-40 if precip_change > 0 else 40,
                font=dict(color=color, size=12, family="Arial, sans-serif"),
                arrowcolor=color, arrowwidth=2
            ))
            
    fig_yoy.update_layout(
        title='*Year-over-Year Change in Temperature and Rainfall*',
        xaxis_title="Year (Year-over-Year Comparison Starting from Year+1)",
        height=600, # Increased height by 20% to accommodate annotations
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=annotations,
        
        # Primary Y-axis (Temperature)
        yaxis=dict(
            title=dict(
                text='Temperature Change (deg C)',
                font=dict(color='#e63946') 
            ), 
            tickfont=dict(color='#e63946'),
            showgrid=False # Hide grid for temp
        ),
        # Secondary Y-axis (Precipitation)
        yaxis2=dict(
            title=dict( 
                text='Precipitation Change (mm)',
                font=dict(color='#007bff')
            ),
            tickfont=dict(color='#007bff'),
            overlaying='y',
            side='right',
            showgrid=True, # Show grid for precip
            gridcolor='#e9ecef'
        )
    )

    st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 2.2s;">', unsafe_allow_html=True)
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <p class='fade-in-element' style='--delay: 2.3s; color: #495057;'>
        <span style='color:#e63946;'>**Red Line:**</span> Shows if the **Mean Temperature** was hotter (+) or colder (-) than the previous year.
        <span style='color:#007bff;'>**Blue Line:**</span> Shows if the **Total Rainfall** was more (+) or less (-) than the previous year. 
        **Arrows mark significant anomalies** with the predicted environmental cause (e.g., El Niño, Drought, etc.).
    </p>
    """, unsafe_allow_html=True)

# --- AIR QUALITY TAB CONTENT (Completed) ---
with TAB_AIR:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">💨</span>
                Air Quality Monitoring: Latest Pollutant Readings
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    
    if df_aq.empty:
        st.info("No AQ data found for this location window.")
    else:
        # Prepare data for Pie Chart
        df_pie = df_aq[['parameter', 'value']].copy()
        df_pie['parameter'] = df_pie['parameter'].str.replace('_', ' ').str.title()
        df_pie.rename(columns={'parameter': 'Pollutant', 'value': 'Value'}, inplace=True)

        col_chart, col_table = st.columns([0.4, 0.6])

        with col_chart:
            # --- 1. Composition Donut Chart (Vibrant Palette) ---
            st.markdown(
                """
                <div class="fade-in-element" style="--delay: 0.3s;">
                    <h3 class="animated-sub-header" style="font-size: 1.4rem !important;">Pollutant Composition</h3>
                </div>
                """, unsafe_allow_html=True
            )
            fig_pie = px.pie(
                df_pie,
                values='Value', 
                names='Pollutant',
                title='*Composition of Current Air Pollutants*',
                hole=0.4, # <--- MODIFIED: This line makes it a Donut Chart
                color_discrete_sequence=['#ffc107', '#ff69b4', '#007bff', '#e63946', '#17a2b8', '#fd7e14'], 
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='white', width=1)))
            fig_pie.update_layout(
                height=540, # Increased height by 20%
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#212529'),
                legend_title_text="Pollutant"
            )
            st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_table:
            # --- 2. Data Table ---
            st.markdown(
                """
                <div class="fade-in-element" style="--delay: 0.4s;">
                    <h3 class="animated-sub-header" style="font-size: 1.4rem !important;">Detailed Readings</h3>
                </div>
                """, unsafe_allow_html=True
            )
            df_aq_display = df_aq[["date", "parameter", "value", "unit"]].rename(
                columns={"date": "Last Updated", "parameter": "Pollutant", "value": "Value", "unit": "Unit"}
            )
            st.markdown('<div class="fade-in-element" style="--delay: 0.6s;">', unsafe_allow_html=True)
            st.dataframe(df_aq_display, use_container_width=True)
            st.caption("Data source is Open-Meteo Air Quality API. Values represent instantaneous readings.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TRENDS TAB CONTENT (Completed) ---
with TAB_TRENDS:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">📊</span>
                Climate Trends & Variable Correlation
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    
    # Prepare data for Global Warming Line (Smoothed Trend)
    df_temp = df_clim.copy()
    if not df_temp.empty:
        df_temp['warming_trend'] = df_temp['temperature_2m_mean'].rolling(window=365, center=True).mean()
    
    colA, colB = st.columns(2)
    
    with colA:
        # Sub-Header
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.3s;">
                <h3 class="animated-sub-header" style="font-size: 1.4rem !important;">Max Wind Speed Trend (m/s)</h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        # --- Max Wind Speed (Blue Highlight with Red Anomaly) ---
        
        if not df_clim.empty:
            max_wind_value = df_clim['windspeed_10m_max'].max()
            abnormal_point = df_clim[df_clim['windspeed_10m_max'] == max_wind_value].iloc[0]
            abnormal_date = abnormal_point['time']
            
            alert_msg = f"🌪 Extreme Wind Alert! Recorded {max_wind_value:.1f} m/s on {abnormal_date.strftime('%Y-%m-%d')}."
            st.warning(alert_msg)

        fig = go.Figure()
        
        # Base Wind Speed Line (Primary Blue)
        fig.add_trace(go.Scatter(x=df_clim["time"], y=df_clim["windspeed_10m_max"], name="Max Wind Speed", line=dict(color='#007bff', width=2))) # Blue for Train
        
        # Add annotation (Arrow and text) pointing to the abnormal peak
        if not df_clim.empty:
            fig.add_annotation(
                x=abnormal_date, 
                y=abnormal_point['windspeed_10m_max'], 
                text="ABNORMAL SPIKE", 
                showarrow=True, 
                font=dict(color="#e63946", size=12), # Red 
                arrowhead=2, 
                arrowsize=1.5, 
                arrowwidth=2, 
                arrowcolor="#e63946", 
                ax=0, ay=-50
            )
            
        fig.update_layout(
            title="*Max Wind Speed Trend with Anomaly Detection*",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            xaxis_title="Date",
            yaxis_title="Wind Speed (m/s)"
        )
        st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with colB:
        # Sub-Header
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.4s;">
                <h3 class="animated-sub-header" style="font-size: 1.4rem !important;">Mean Temperature Trend (deg C)</h3>
            </div>
            """, unsafe_allow_html=True
        )
        # --- Mean Temperature Trend (Yellow/Orange Warming Line) ---
        fig = go.Figure()
        
        # Base Mean Temperature Line (Muted Blue)
        fig.add_trace(go.Scatter(
            x=df_clim["time"], 
            y=df_clim["temperature_2m_mean"], 
            name="Daily Mean Temp", 
            line=dict(color='#457b9d', width=2), # Muted Blue
            opacity=0.8
        ))
        
        # Global Warming Trend Line (Smoothed and Luminous)
        if not df_temp.empty:
            fig.add_trace(go.Scatter(
                x=df_temp["time"], 
                y=df_temp["warming_trend"], 
                name="Global Warming Trend (365-day Avg)", 
                line=dict(color='#fd7e14', width=4, dash='dashdot'), # Orange for contrast
                opacity=0.9
            ))
            
        fig.update_layout(
            title="*Mean Temperature Trend with Global Warming Line*",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            xaxis_title="Date",
            yaxis_title="Temperature (deg C)",
            legend=dict(y=0.99, x=0.01)
        )
        st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.6s;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # ADDED: Full-screen button for visualization demo
        # if st.button("Click to View Temperature Trend Fullscreen/Maximized"):
        # 	st.toast("Plotly charts support built-in fullscreen/zoom on double-click or use the top-right toolbar!", icon="📈")
        
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    
    # --- Correlation Snapshot (Light Scale) ---
    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 0.8s;">
            <h3 class="animated-sub-header">
                <span class="animated-emoji" style="--delay: 0.9s;">🔗</span> Correlation Snapshot
            </h3>
        </div>
        """, unsafe_allow_html=True
    )
    corr_df = df_clim.drop(columns=["time"]).corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_df, 
        text_auto=".2f",
        aspect="auto", 
        title="*Climate Variables Correlation Matrix*",
        color_continuous_scale=px.colors.sequential.Plotly3_r, # Using a light-friendly scale
    )
    fig_corr.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 1.0s;">', unsafe_allow_html=True)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Updated Explanation Section Colors
    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 1.2s;">
            <h3 class="animated-sub-header" style="font-size: 1.4rem !important;">🔍 Technical Explanation: Understanding Correlation</h3>
        </div>
        """, unsafe_allow_html=True
    )
    st.caption("""
    Correlation measures how closely two variables move together, ranging from **-1.0 to +1.0**.

    * **+1.0 (Bright Yellow/Orange):** Perfect *Positive Correlation*. When one variable (e.g., Max Temperature) increases, the other variable (e.g., Mean Temperature) increases at the same time.
    * **-1.0 (Dark Color):** Perfect *Negative Correlation*. When one variable (e.g., Solar Radiation) increases, the other variable (e.g., Cloud Cover/Precipitation) decreases.
    * **0.0 (Mid-color):** *No Correlation*. The variables have no discernible linear relationship.
    """)
    
    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 1.4s;">
            <h3 class="animated-sub-header" style="font-size: 1.4rem !important;">🗣 सहसंबंध स्नैपशॉट (Correlation Snapshot) का मतलब समझें</h3>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("""
    <div class='card fade-in-element' style='--delay: 1.6s; background: #f1f7fd; border: 1px solid #007bff; color: #212529; padding: 15px; border-radius: 8px;'>
    <p style='color:#495057;'>यह ग्रिड (grid) हमें बताती है कि मौसम के अलग-अलग फैक्टर (factors) एक-दूसरे से कैसे जुड़े हैं।</p>
    
    <ul style='color:#495057;'>
    	<li>*हाई वैल्यू (High Value) - ज़्यादातर चमकीला पीला/नारंगी:* इसका मतलब है *Strong Positive Relation*। मतलब, अगर **एक चीज़ बढ़ती है, तो दूसरी भी लगभग उसी Rate से बढ़ती है*। (Example: Max Temp और Mean Temp हमेशा साथ बढ़ते हैं: 0.96)</li>
    	<li>*लो वैल्यू (Low Value) - ज़्यादातर गहरा रंग:* इसका मतलब है *Negative Relation* या *Weak Relation*। मतलब, अगर एक चीज़ बढ़ती है, तो **दूसरी घट जाती है**, या उनका कोई खास कनेक्शन नहीं है। (Example: Rainfall ज़्यादा होने पर Solar Radiation (धूप) कम हो जाती है: लगभग -0.31)</li>
    	<li>*जीरो के पास (Near Zero):* मतलब उन दोनों के बीच *कोई ख़ास Connection नहीं* है।</li>
    </ul>

    <p style='color:#212529; font-weight:600;'>इस डेटा से मुख्य बातें (Key Insights):</p>
    <ul style='color:#495057;'>
    	<li>*गर्मी और धूप:* तापमान (Temperature) और धूप (Shortwave Radiation) का Strong Connection है (0.76)।</li>
    	<li>*Precipitation (बारिश):* बारिश का किसी भी Temperature से बहुत कम Connection है (Max Temp से 0.00)।</li>
    	<li>*Global Warming:* *'Year'* का *Temperature* से छोटा लेकिन Positive Connection (0.15) दिख रहा है, जो बताता है कि समय के साथ *Overall Temperature* थोड़ा बढ़ रहा है।</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Starting at approximately line 1111 (inside the original with TAB_FORECAST: block)
with TAB_FORECAST:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🔮</span>
                AI Forecasts with Backtest Metrics
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    target = st.selectbox(
        "Target to forecast", 
        ["temperature_2m_mean","temperature_2m_max","temperature_2m_min","precipitation_sum"], 
        index=0,
        help="The variable you want the AI model to predict into the future (e.g., Mean Temperature)."
    )
    horizon = st.slider("Forecast horizon (days)", 7, 365, 90)

    model_used, ts, train, valid, fcst, metrics = backtest_train_forecast(df_clim[["time", target]].dropna(), target, horizon=horizon, model_choice=model_choice)

    st.info(f"Model used: *{model_used}* | MAE: *{metrics['MAE']:.3f}* | MAPE: *{metrics['MAPE']:.3f}*")

    # --- NEW AI ANALYSIS SECTION ---
    # Inside the TAB_FORECAST block:
# ... (Previous Forecast Chart Generation Code) ...

# --- NEW AI MODEL CRITIQUE SECTION ---
# --- NEW AI ANALYSIS SECTION ---
with TAB_FORECAST:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🔮</span>
                AI Forecasts with Backtest Metrics
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    # 1. TARGET VARIABLE SELECTION (REQUIRED for the forecast function)
    target = st.selectbox(
        "Target to forecast",
        ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        index=0,
        help="The variable you want the AI model to predict into the future (e.g., Mean Temperature).",
        key="forecast_target_variable" # Added unique key to prevent conflict
    )
    
    # 2. MODEL CHOICE AND HORIZON (Assuming these are correctly defined in the main scope/sidebar)
    # NOTE: The st.markdown("### 🤖 AI Forecasting Model")... st.selectbox(key="sidebar_forecast_model")... 
    # block MUST remain OUTSIDE of this 'with TAB_FORECAST' section if it's in the sidebar.
    
    # We retrieve the model_choice and target variable from the global scope/sidebar definition.
    # We also define a unique key for the target selectbox.

    horizon = st.slider("Forecast horizon (days)", 7, 365, 90,key="forecast_horizon_slider")

    # 3. FORECAST EXECUTION (Requires target and model_choice, which must be defined)
    model_used, ts, train, valid, fcst, metrics = backtest_train_forecast(
        df_clim[["time", target]].dropna(), 
        target, 
        horizon=horizon, 
        model_choice=model_choice
    )

    st.info(f"Model used: *{model_used}* | MAE: *{metrics['MAE']:.3f}* | MAPE: *{metrics['MAPE']:.3f}*")

    # Keep Metric explanations
    st.caption("""
    <span style='color:#495057;'>*MAE (Mean Absolute Error):* The average magnitude of errors in the predictions, measured in the same units as the target (e.g., deg C). Lower is better.</span>
    <br>
    <span style='color:#495057;'>*MAPE (Mean Absolute Percentage Error):* The average percentage error of the prediction. 10% MAPE means the forecast is off by 10% on average. Lower is better.</span>
    """, unsafe_allow_html=True)
    
    # ... (AI Critique and Plotly chart rendering blocks follow here) ...
    
    # --- FIXED AI ANALYSIS SECTION ---
    if metrics["MAE"] is not None:
        
        # 1. Prepare specialized prompt for Forecast Critique (The Prompt is correctly structured here)
        prompt_ai_user = (
            f"The city {_name} used the model {model_used}. The forecast for '{target}' produced MAE: {metrics['MAE']:.3f} and MAPE: {metrics['MAPE']:.3f}. "
            f"Explain this reliability to a policy maker. What does the MAE imply for planning and resource risk?"
        )
        
        # 2. Generate Intelligent Insight (CORRECTED FUNCTION CALL)
        # Argument 1: system_prompt_addition (Instructions for the AI persona)
        # Argument 2: user_prompt (The actual data/question)
        ai_reliability_insight = generate_contextual_ai_insight(
            "You are an expert AI Model Auditor focused on policy implications. Your response must be concise (max 4 sentences) and highly authoritative.",
            prompt_ai_user 
        )

        st.markdown("<div class='metric-card fade-in-element' style='--delay: 0.6s; border: 2px solid var(--brand2);'>", unsafe_allow_html=True)
        st.markdown(f"**🧠 Virtual Data Scientist: Forecast Reliability Critique**")
        st.markdown(f"<p style='margin-top: 10px; color:#343a40; font-size: 0.95rem;'>{ai_reliability_insight}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # --- PLOTLY CHART RENDERING ---
    fig = go.Figure()
    # Premium Line Styling (Updated to Bright Palette)
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train Data", line=dict(color='#007bff', width=2))) # Blue for Train
    fig.add_trace(go.Scatter(x=valid["ds"], y=valid["y"], name="Validation Data", line=dict(color='#e63946', width=2))) # Red for Validation

    # unify forecast frame
    if "ds" not in fcst.columns:
        if "time" in fcst.columns:
            fcst.rename(columns={"time":"ds"}, inplace=True)
        else:
            # create series from last date
            last = ts["ds"].iloc[-1]
            fcst["ds"] = pd.date_range(last + pd.Timedelta(days=1), periods=len(fcst), freq='D')
    
    yhat = fcst["yhat"] if "yhat" in fcst.columns else fcst.iloc[:,1]
    
    # Prediction line (Yellow/Amber and Dashed)
    fig.add_trace(go.Scatter(x=fcst["ds"], y=yhat, name="Forecast", line=dict(color='#ffc107', width=3, dash="dash")))
    
    fig.update_layout(
        title=f"*AI Forecast: {target}*", 
        xaxis_title="Date", 
        yaxis_title=target,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.4s;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button("⬇ Download Forecast CSV", data=fcst.to_csv(index=False), file_name=f"forecast_{target}.csv", mime="text/csv")

# --- IMPACT STORY TAB CONTENT (Completed) ---
with TAB_STORY: # RENDERED IMPACT STORY CONTENT
    # --- Inject the new animated background container (Outer Wrapper) ---
    st.markdown("<div class='impact-story-container'>", unsafe_allow_html=True)
    
    # --- Redesigned Main Header (Glow, Animation) ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">💥</span>
                Climate Change Impact Story: Real-World Consequences for {_name}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    # --- Row 1: Crop Loss Simulation (Updated Colors) ---
    # Sub-Header (Clean, Bold, Hover Glow)
    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 0.3s;">
            <h3 class="animated-sub-header">
                <span class="animated-emoji" style="--delay: 0.4s;">🌾</span> Local Crop Yield Loss Simulation
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_loss, col_map = st.columns([0.45, 0.55])
    
    # Use mean_temp calculated in the main block
    stressed_yield, loss_percent, status = get_crop_loss_simulation(mean_temp) 

    with col_loss:
        # Re-styling the temperature display (Bolder, cleaner font)
        st.markdown(
            f"""
            <div class="fade-in-element" style="--delay: 0.5s;">
                <p style="font-family: 'Signika', sans-serif; font-size: 1.3rem; font-weight: 600; color: var(--muted);">
                    Average Mean Temperature: 
                    <span style="color: #e63946; font-size: 1.5rem; font-weight: 800;">{mean_temp:.1f} deg C</span>
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        st.info(f"Predicted Agricultural Status: *{status}*")
        
        # Plotly Gauge to simulate yield loss visually (Updated Colors)
        fig_loss = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = stressed_yield,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Simulated Crop Yield (Relative %)", 'font': {'size': 20, 'color': '#212529'}},
            gauge = {
                'shape': "angular",
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#495057"},
                'bar': {'color': "#007bff"}, # Blue fill
                'bgcolor': "#e9ecef",
                'steps': [
                    {'range': [0, 80], 'color': '#e63946'}, # Red Danger Zone
                    {'range': [80, 95], 'color': '#ffc107'}, # Yellow Warning Zone
                    {'range': [95, 100], 'color': '#38a3a5'} # Teal Optimal Zone
                ],
                'threshold': {
                    'line': {'color': "#e63946", 'width': 5},
                    'thickness': 0.8,
                    'value': stressed_yield
                }
            }
        ))
        fig_loss.update_layout(
            height=420, margin=dict(l=10, r=10, t=50, b=10), 
            plot_bgcolor='white', paper_bgcolor='white', 
            font=dict(color='#212529')
        )
        st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.6s;">', unsafe_allow_html=True)
        st.plotly_chart(fig_loss, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Re-styling the loss percentage for maximum impact
        st.markdown(
            f"""
            <div class="fade-in-element" style="--delay: 0.7s;">
                <p style='color: #e63946; font-weight:700; font-size: 1.2rem; margin-top: 15px;'>
                Estimated Loss: 
                <span style='font-size:2.0rem; margin-left: 10px; text-shadow: 0 0 5px rgba(230, 57, 70, 0.8);'>-{loss_percent:.1f}%</span>
                </p>
                <p style='color: #495057;'>High average temperatures (above 25 deg C proxy) severely damage grain filling and flower production, leading to direct economic loss for farmers and instability in food supply chains.</p>
            </div>
            """, unsafe_allow_html=True
        )

    # --- Row 2: City CO2 Accountability Map (Updated Colors) ---
    with col_map:
        # Sub-Header (Clean, Bold, Hover Glow)
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.8s;">
                <h3 class="animated-sub-header" style="font-size: 1.4rem !important; margin-top: 5px;">
                    <span class="animated-emoji" style="--delay: 0.9s;">💨</span> City CO2 Accountability Map (kT/Yr)
                </h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        df_emissions = get_all_india_city_emissions()
        
        # Dynamic Select Box for City Comparison
        st.multiselect(
            "Select cities for CO2 contribution comparison",
            options=df_emissions['City'].tolist(),
            default=[_name] if _name in df_emissions['City'].tolist() else [df_emissions['City'].iloc[0]],
            key="story_city_multiselect_redesign", 
            help="Compare your selected city's estimated annual carbon emissions against other major Indian cities."
        )

        df_filtered = df_emissions[df_emissions['City'].isin(st.session_state.story_city_multiselect_redesign)]
        
        # Choropleth Map showing CO2 contribution (Updated Colors)
        fig_map = px.scatter_geo(
            df_emissions,
            lat='Latitude',
            lon='Longitude',
            text=df_emissions.apply(lambda row: f"{row['City']}<br>CO2: {row['CO2_Emissions_Annual_kT']:,} kT", axis=1),
            size="CO2_Emissions_Annual_kT",
            color="CO2_Emissions_Annual_kT",
            projection="natural earth",
            title="Relative Annual CO2 Contribution of Major Cities",
            color_continuous_scale=px.colors.sequential.YlOrRd, # Yellow/Orange/Red scale for emissions
            scope='asia'
        )
        
        # Highlight selected cities with an intense Pink ring 
        selected_lats = df_filtered['Latitude'].tolist()
        selected_lons = df_filtered['Longitude'].tolist()
        
        fig_map.add_trace(go.Scattergeo(
            lat=selected_lats,
            lon=selected_lons,
            mode='markers',
            marker=dict(
                size=df_filtered['CO2_Emissions_Annual_kT'].apply(lambda x: x/400).tolist(), 
                color='#ff69b4', # Pink Highlight
                line_width=3,
                opacity=0.8,
                line_color='#ffffff'
            ),
            hoverinfo='none',
            name='Selected City (Highlight)'
        ))
        
        fig_map.update_geos(
            lataxis_range=[5, 35], 
            lonaxis_range=[65, 90], 
            showcountries=True, 
            countrycolor="#495057", # Darker country lines
            subunitcolor="#495057",
            showland=True,
            landcolor="#e9ecef" # Light land color
        )
        
        fig_map.update_layout(
            height=540, # Increased by 20%
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            margin={"r":0,"t":40,"l":0,"b":0},
            coloraxis_colorbar=dict(title="CO2 (kT/yr)")
        )
        st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 1.0s;">', unsafe_allow_html=True)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True) # Pulsing Separator

    # --- Row 3: Glacier Melt and Sea Level Risk (Global/Regional Context) ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 1.2s;">
            <h3 class="animated-sub-header">
                <span class="animated-emoji" style="--delay: 1.3s;">🧊</span> Regional Climate Threats
            </h3>
        </div>
        """, unsafe_allow_html=True
    )
    col_glacier, col_sea = st.columns(2)
    
    with col_glacier:
        st.markdown(
            f"""
            <div class="fade-in-element" style="--delay: 1.4s;">
                <h4 style="font-family: 'Signika', sans-serif; font-weight: 700; color: #008080;">
                    Himalayan Glacier Melt & Water Security
                </h4>
                <p style='color: #495057;'>**Global Temperature Rise** directly leads to accelerated melting of Himalayan glaciers, which are the primary source of water for major Indian rivers like the Ganga.
                <ul style='color: #495057;'>
                	<li>**Glacier Mass Loss (Est. since 2000):** <span style='color: #e63946; font-weight:700;'>-25% to -35%</span></li>
                	<li>**Direct Impact for {_name}:** Increased risk of **flash floods** initially, followed by long-term **severe water scarcity** as the ice mass depletes.</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        
    with col_sea:
        risk_level = "Indirect"
        risk_color = '#ffc107'
        if _name in ['Mumbai', 'Chennai', 'Kolkata', 'Kochi']:
            risk_level = "EXTREME"
            risk_color = '#e63946'

        st.markdown(
            f"""
            <div class="fade-in-element" style="--delay: 1.5s;">
                <h4 style="font-family: 'Signika', sans-serif; font-weight: 700; color: #008080;">
                    Sea Level Rise Risk (Coastal Threat)
                </h4>
                <p style='color: #495057;'>Melting glaciers and thermal expansion of seawater drive **sea level rise**.
                <ul style='color: #495057;'>
                	<li>**Local Risk for {_name}:** <span style='color: {risk_color}; font-weight:700;'>{risk_level}</span> (Source of indirect risk: Mass climate migration & supply chain disruption).</li>
                	<li>**Coastal Metros Risk (Mumbai/Kolkata):** Major infrastructure, historical sites, and large population centers are at high risk of **permanent inundation** by 2100.</li>
                	<li>**Impact:** Loss of wetlands, increased saline intrusion into groundwater, and massive infrastructure costs.</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True
        )

    # --- Original Future Impact Section (Health & Afforestation) ---
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 1.7s;">
            <h3 class="animated-sub-header">
                <span class="animated-emoji" style="--delay: 1.8s;">🏃</span> Air Pollution & Public Health Advisory
            </h3>
        </div>
        """, unsafe_allow_html=True
    )
    pm_level = pm25_now if not math.isnan(pm25_now) else 80.0
    prediction = get_future_impact_prediction(pm_level)
    
    c_pred, c_adv = st.columns([0.4, 0.6])

    with c_pred:
        st.markdown("<div class='metric-card fade-in-element' style='--delay: 1.9s;'>", unsafe_allow_html=True)
        st.metric(f"Predicted Health Risk at PM2.5 of {pm_level:.1f} ug/m3", prediction['health_risk'])
        st.warning(f"*Running Advisory:* {prediction['advice']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- Afforestation Goals ---
        st.markdown(f"""
            <div class="fade-in-element" style="--delay: 2.0s;">
            <h3 class='animated-sub-header' style='font-size: 1.4rem !important; margin-top: 15px;'>
                <span class='planting-tree animated-emoji' style='--delay: 2.1s;'>🌳</span> Afforestation Goals for {_name}
            </h3>
            </div>
        """, unsafe_allow_html=True)

        tree_data = get_tree_inventory(_name)

        c_tree1, c_tree2 = st.columns(2)

        # Current Trees (Est.)
        with c_tree1:
            st.markdown(f"""
            	<div class='goal-card-metric fade-in-element' style='--delay: 2.2s;'>
            		<p style='color:#6c757d; font-size:1.0rem; margin-bottom: 5px;'>Current Trees (Est.)</p>
            		<div class='current-number'>{tree_data['current']:,}</div>
            	</div>
            """, unsafe_allow_html=True)

        # Trees Needed
        with c_tree2:
            st.markdown(f"""
            	<div class='goal-card-metric fade-in-element' style='--delay: 2.3s;'>
            		<p style='color:#6c757d; font-size:1.0rem; margin-bottom: 5px;'>Trees Needed (Goal)</p>
            		<div class='goal-number'>{tree_data['needed']:,}</div>
            	</div>
            """, unsafe_allow_html=True)

        st.caption(f"Goal: {tree_data['target_ratio']} trees per person (Population: {tree_data['population']:,})")

    with c_adv:
        # River Health is still displayed for comprehensive impact
        river_data = get_river_health_data(_name)
        river_name = river_data['River'].iloc[0]
        river_base_name = river_name.split(' (')[0] if ' (' in river_name else river_name # e.g., "Ganga"

        st.markdown(
            f"""
            <div class="fade-in-element" style="--delay: 2.4s;">
                <h3 class='animated-sub-header' style='font-size: 1.4rem !important; margin-top: 15px;'>
                    <span class="animated-emoji" style="--delay: 2.5s;">🌊</span> River Health Status ({river_name})
                </h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.dataframe(river_data[['Dissolved Oxygen (DO mg/L)', 'BOD (mg/L)', 'Coliform (MPN/100ml)', 'Status']], hide_index=True, use_container_width=True)

        st.markdown(
            f"""
            <div class="fade-in-element" style="--delay: 2.6s;">
                <h3 class='animated-sub-header' style='font-size: 1.4rem !important;'>
                    <span class="animated-emoji" style="--delay: 2.7s;">📢</span> Actionable Advisories
                </h3>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown(f"""
        <div class="card fade-in-element" style='--delay: 2.8s; background: #f1f7fd; border: 1px solid #007bff; padding: 10px; border-radius: 8px;'>
        *Public Advice:* Focus on waste segregation, reducing personal vehicle use, and avoiding littering near {river_base_name} River.
        *Government Advice:* Prioritize industrial effluent treatment, expand public transit, and invest in large-scale urban greening projects in {_name}.
        </div>
        """, unsafe_allow_html=True)

    # --- Close the main animated container ---
    st.markdown("</div>", unsafe_allow_html=True)

# Starting at approximately line 1358 (inside the original with TAB_SCORE: block)
with TAB_SCORE:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">💯</span>
                City Sustainability Score
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    pm_for_score = pm25_now if not math.isnan(pm25_now) else 60.0
    score, sub = compute_sustainability_score(SustainabilityInputs(
        pm25=pm_for_score,
        co2_per_capita=co2_pc,
        renewable_share=float(ren_share),
        water_quality_index=float(water_idx),
        waste_recycling_rate=float(recycle),
    ))
    colL, colR = st.columns([0.45,0.55])
    with colL:
        st.markdown("<div class='metric-card fade-in-element' style='--delay: 0.4s;'>", unsafe_allow_html=True)
        st.metric("Sustainability Score", f"{score:.1f} / 100")
        st.caption("A composite score (0-100) based on five dimensions: Air Quality, CO2 Emissions, Renewable Energy usage, Water Quality, and Waste Recycling Rate. Higher is better.")
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.3s;">
                <h3 class="animated-sub-header" style="font-size: 1.4rem !important; margin-top: 5px;">Score Breakdown</h3>
            </div>
            """, unsafe_allow_html=True
        )
        sub_df = pd.DataFrame({"Dimension": list(sub.keys()), "Score": list(sub.values())})
        fig = px.bar(
            sub_df, 
            x="Dimension", 
            y="Score", 
            title="Sub-Scores (0-100)",
            color="Score",
            color_continuous_scale=px.colors.sequential.Plotly3,
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'))
        st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
   # Inside the TAB_SCORE block (after score/sub calculation):
# ... (Code that renders score and sub_df chart) ...

# --- NEW AI POLICY PRIORITIZATION ---
# Only show policy advice if the score is below the "Good" threshold (e.g., 75/100)
# --- VIRTUAL ANALYST: PRIORITY GOVERNANCE ROADMAP (FIXED LOGIC) ---

# --- Gather robust inputs for the local fallback logic ---
# Ensure variables are defined and robustly accessed outside the condition
pm = pm25_now if not math.isnan(pm25_now) else 80
msw = current_msw_tpd if not math.isnan(current_msw_tpd) else 1000
wtr = water_idx if not math.isnan(water_idx) else 65

# Assuming this entire section is placed INSIDE your existing 'with TAB_SCORE:' block

# --- CHECK FOR LOW SCORE (The conditional start) ---
if score < 75: 
    # 1. Determine lowest scores (This part is always safe to run)
    sorted_sub = sorted(sub.items(), key=lambda item: item[1])
    lowest_dimension = sorted_sub[0][0]
    lowest_score = sorted_sub[0][1]

    # --- INPUTS FOR BOTH AI AND LOCAL FALLBACK ---
    pm = pm25_now if not math.isnan(pm25_now) else 80
    msw = current_msw_tpd if not math.isnan(current_msw_tpd) else 1000
    wtr = water_idx if not math.isnan(water_idx) else 65

    
    # 2. DECISION POINT: AI ON or AI OFF?
    if GEMINI_CLIENT:
        # --- AI ONLINE: Attempt Live Analysis ---
        
        # Prepare the prompt using the calculated scores
        prompt_advice = (
            f"The city {_name} has a current sustainability score of {score:.1f}/100. The lowest scoring area is '{lowest_dimension}' at {lowest_score}/100. "
            f"Generate a **prioritized, highly actionable list of 3 governance actions** the municipal body should take immediately to specifically address this weakest dimension and raise the score. Use bullet points."
        )
        
        # Call the AI model
        governance_advice_output = generate_contextual_ai_insight(
            "You are a Municipal Policy Advisor, focused on immediate, specific, and cost-effective policy interventions.",
            prompt_advice
        )

        # Display the result (will show AI output if successful)
        st.markdown("### 🎯 Virtual Analyst: Priority Governance Roadmap")
        st.success(governance_advice_output)

    else:
        # --- AI OFFLINE: RUN GUARANTEED LOCAL FALLBACK ---
        
        # Execute the guaranteed local renderer function (defined in your utilities)
        # This function formats the header and calls get_priority_governance_advice internally.
        render_local_governance_fallback(_name, co2_pc, msw, pm, wtr)

        
with TAB_CARBON:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">👣</span>
                Personal Carbon Footprint & City Emissions
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    # 1. India-Wide City CO2 Comparison Map (Choropleth/Scattergeo)
    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 0.3s;">
            <h3 class="animated-sub-header">🗺️ Major City Annual CO2 Emissions Comparison (Proxy Data)</h3>
        </div>
        """, unsafe_allow_html=True
    )
    
    df_map = get_all_india_city_emissions()
    
    # Identify the current selected city's location for highlighting
    current_city_name = _name
    
    # Create the Scattergeo map
    fig_map = go.Figure(data=go.Scattergeo(
        lon = df_map['Longitude'],
        lat = df_map['Latitude'],
        text = df_map.apply(lambda row: f"{row['City']}<br>CO2: {row['CO2_Emissions_Annual_kT']:,} kT", axis=1),
        mode = 'markers',
        marker = dict(
            size = df_map['CO2_Emissions_Annual_kT'] / 1000, # Scale marker size by kT
            sizemode = 'area',
            sizemin = 5,
            color = df_map['CO2_Emissions_Annual_kT'],
            colorscale = px.colors.sequential.Sunset_r, # Use a warm color scale for emissions
            cmin = df_map['CO2_Emissions_Annual_kT'].min(),
            cmax = df_map['CO2_Emissions_Annual_kT'].max(),
            colorbar_title = "CO2 (kT/yr)",
            line_color='rgba(0,0,0,0.5)'
        ),
    ))
    
    # Add a separate trace for the current city highlight
    current_city_data = df_map[df_map['City'].str.lower() == current_city_name.lower()]
    if not current_city_data.empty:
        fig_map.add_trace(go.Scattergeo(
            lon = current_city_data['Longitude'],
            lat = current_city_data['Latitude'],
            text = current_city_data.apply(lambda row: f"**SELECTED: {row['City']}**<br>CO2: {row['CO2_Emissions_Annual_kT']:,} kT", axis=1),
            mode = 'markers',
            marker = dict(
                size = current_city_data['CO2_Emissions_Annual_kT'].iloc[0] / 1000 + 10, # Make the selected city marker bigger
                color = '#ff69b4', # Pink for selected highlight
                line_width = 3,
                line_color = '#FFFFFF'
            ),
            hoverinfo='text',
            name='Selected City'
        ))

    fig_map.update_layout(
        title_text = f'CO2 Emissions Across Indian Cities (Current City: **{current_city_name}**)',
        showlegend = False,
        geo = dict(
            scope = 'asia',
            lonaxis_range= [68, 98],
            lataxis_range= [5, 38],
            subunitcolor = "#007bff", # Blue border for states/subunits
            subunitwidth = 1,
            countrycolor = "#495057", # Darker border for country
            countrywidth = 1.5,
            bgcolor = '#f8f9fa', # Map background color
            lakecolor = '#e9ecef', # Light lake/water body color
            landcolor = '#ffffff', # White land color
            coastlinecolor = '#adb5bd',
            projection_type = 'mercator'
        ),
        height=660, # Increased by 20%
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529')
    )
    st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    
    # 2. Personal Carbon Footprint Calculator 
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.7s;">
            <h3 class="animated-sub-header">Personal Carbon Footprint for {_name} (Quick Estimate)</h3>
        </div>
        """, unsafe_allow_html=True
    )

    # 1. Auto-Estimation Checkbox
    auto_estimate = st.checkbox(
        "⚡ *Auto-Fetch Household Estimate* based on current location data", 
        value=False,
        key="auto_estimate_carbon_tab", 
        help="Calculates a baseline household footprint using India-specific averages, adjusted by your city's air quality, climate, and sustainability metrics."
    )

    # Define India-Specific Baseline Averages for Carbon Footprint Estimation
    BASE_KWH = 180           # National average monthly consumption
    BASE_KM_CAR = 350 # Average urban monthly car travel (higher side)
    BASE_LPG = 6      # Average monthly LPG use (kg)
    BASE_FLIGHTS = 1   # Average flights per year
    
    # Emission Factors (unchanged)
    EF_CAR = 0.18 # kg CO2e / km
    EF_KWH = 0.7  # kg CO2e / kWh (India grid average)
    EF_FLIGHT = 180 # kg CO2e / 2-hour flight
    EF_LPG = 3.0    # kg CO2e / kg LPG
    DIET_MAP = {"Heavy meat": 300, "Mixed": 200, "Vegetarian": 150, "Vegan": 120}

    colA, colB = st.columns(2)
    
    # 2. Determine Inputs (Manual or Auto-Fetched)
    if auto_estimate:
        st.info(f"Using *{_name}* environmental data to localize the estimate.")
        
        # PROXY LOGIC: Intelligent data-driven estimation 
        
        # Proxy 1: Diet Type (Proxy based on city Water Quality Index)
        diet_proxy = "Mixed"
        if water_idx < 50:
            diet_proxy = "Heavy meat"
        elif water_idx > 80:
            diet_proxy = "Vegetarian"
        
        # Proxy 2: Car Travel (Proxy based on population proxy and max wind speed)
        tree_data = get_tree_inventory(_name)
        pop_adj = tree_data['population'] / 1000000 # millions
        # Reduce travel slightly if pop is high (congestion) or wind is extreme
        km_car_proxy = max(50, BASE_KM_CAR - int(pop_adj * 50) - int(df_clim['windspeed_10m_max'].max() * 5))
        
        # Proxy 3: Electricity consumption (Proxy based on high/low temperatures for AC use)
        mean_temp_swing = df_clim['temperature_2m_max'].max() - df_clim['temperature_2m_min'].min()
        kwh_proxy = int(BASE_KWH + mean_temp_swing * 2) 

        # Use city's recycling rate from sidebar input
        recycle_rate_proxy = recycle
        
        # Other simple averages
        flights_proxy = BASE_FLIGHTS
        lpg_proxy = BASE_LPG

        with colA:
            km_car = st.number_input("Monthly car travel (km)", value=km_car_proxy, disabled=True, key="auto_km_car")
            kwh = st.number_input("Monthly electricity use (kWh)", value=kwh_proxy, disabled=True, key="auto_kwh")
            flights = st.number_input("Flights per year (2-hr avg)", value=flights_proxy, disabled=True, key="auto_flights")
        with colB:
            diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=list(DIET_MAP.keys()).index(diet_proxy), disabled=True, key="auto_diet")
            lpg = st.number_input("Monthly LPG use (kg)", value=lpg_proxy, disabled=True, key="auto_lpg")
            recycle_rate = st.slider("Household recycling (%)", 0, 100, recycle_rate_proxy, disabled=True, key="auto_recycle")

    else:
        # Manual Input Mode (Original code structure)
        with colA:
            km_car = st.number_input("Monthly car travel (km)", 0, 10000, 300, key="manual_km_car")
            kwh = st.number_input("Monthly electricity use (kWh)", 0, 2000, 180, key="manual_kwh")
            flights = st.number_input("Flights per year (2-hr avg)", 0, 50, 1, key="manual_flights")
        with colB:
            diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=1, key="manual_diet")
            lpg = st.number_input("Monthly LPG use (kg)", 0, 100, 6, key="manual_lpg")
            recycle_rate = st.slider("Household recycling (%)", 0, 100, 20, key="manual_recycle")

    # 3. Calculation and Display
    
    # Calculate effective EF_KWH (Dynamically adjusted by Renewables Share)
    effective_ef_kwh = EF_KWH * (1 - ren_share/100)
    st.caption(f"Calculated effective CO2 factor for electricity in this city: {effective_ef_kwh:.3f} kg/kWh (based on {ren_share}% renewables)")

    # Calculate monthly carbon emissions
    if km_car is None: km_car = 0
    if kwh is None: kwh = 0
    if flights is None: flights = 0
    if lpg is None: lpg = 0
    
    monthly = (
        km_car * EF_CAR + 
        kwh * effective_ef_kwh + # Use adjusted EF
        (flights * EF_FLIGHT) / 12 + 
        lpg * EF_LPG + 
        DIET_MAP[diet]
    ) * (1 - recycle_rate / 400) # Simple reduction for waste

    st.markdown("<div class='metric-card fade-in-element' style='--delay: 0.9s;'>", unsafe_allow_html=True)
    # Enhanced Carbon Footprint Display
    st.markdown(
        f"""
        <div class='metric-card animated-card-moderate fade-in-element gradient-border' 
            style='--delay: 0.3s; padding: 15px; --border-color: #38a3a5;'>
            <h3 style='font-size: 1.2rem; color: var(--muted);'>Estimated Monthly CO2 Footprint</h3>
            <div class='glowing-text' style='font-size: 2.5rem; color: #38a3a5; margin: 10px 0;'>
                {monthly/1000:.2f} <span style='font-size: 1.5rem;'>t CO2e</span>
            </div>
            <div class='sliding-text' style='color: var(--muted); font-size: 0.9rem;'>
                CO2 equivalent tons per month
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ENHANCEMENT: Add expander to explain the calculation factors
    with st.expander("❓ How is this calculated? (Emission Factors Explained)", expanded=False):
        st.caption("""
        This tool provides a *quick, simplified estimate* of monthly **CO2e (Carbon Dioxide equivalent)** emissions. 
        The calculation relies on global and regional *Emission Factors (EFs)*:
        
        * *Car Travel (EF_CAR = 0.18):* Assumes **0.18 kg CO2e per km** for an average gasoline car.
        * *Electricity (EF_KWH):* The factor is **dynamically adjusted** based on the Renewable Energy Share input from the sidebar. Lower renewables mean higher grid emissions.
        * *Flights (EF_FLIGHT = 180):* Assumes **180 kg CO2e per 2-hour flight**.
        * *LPG Use (EF_LPG = 3.0):* Assumes **3.0 kg CO2e per kg of LPG** consumed.
        * *Diet:* Based on estimated monthly emissions for different diet types.
        * *Recycling:* A reduction is applied based on the input rate to account for waste diversion.
        """)

    fig = px.pie(names=["Travel","Electricity","Flights","LPG","Diet"],
              values=[
                      km_car * EF_CAR, 
                      kwh * effective_ef_kwh, # Use adjusted EF in pie chart too
                      (flights * EF_FLIGHT) / 12, 
                      lpg * EF_LPG, 
                      DIET_MAP[diet]
                      ],
              title="Breakdown (kg CO2e per month)")
    # Bright/Environmental colors for the pie chart
    fig.update_traces(marker=dict(colors=px.colors.sequential.Plotly3))
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'))
    st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 1.1s;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- GREEN INFRASTRUCTURE TAB CONTENT (Completed) ---
# --- GREEN INFRASTRUCTURE TAB CONTENT (Completed & Enhanced) ---
with TAB_GREEN_INFRA:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🔋</span>
                Green Infrastructure & Electric Vehicle Adoption
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 0.3s;">
            <p>Monitor and compare the proportional distribution of green capacity and sustainable transport adoption across selected cities.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Get a list of all cities for the multiselect (Keep this unchanged)
    known_cities = ["Prayagraj", "Lucknow", "Varanasi", "Kanpur", "Mumbai", "Delhi", "Bengaluru", "Agra"]
    all_cities = sorted(list(set(known_cities + [_name]))) 

    selected_cities_green = st.multiselect(
        "Select cities to compare (including current city):",
        options=all_cities,
        default=[_name, "Lucknow", "Mumbai"], 
        key="green_infra_city_select"
    )

    if not selected_cities_green:
        st.info("Please select at least one city to view the data.")
    else:
        st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

        # --- DEFINE CUSTOM COLOR PALETTE (For the Donut Charts) ---
        CUSTOM_VIBRANT_PALETTE = ["#F0E90C", "#00ff1a", "#FF2E3C", "#00B7FF", '#FF69B4', '#FF9900', "#ea2626"] 
        
        # --- 1. SETUP 3-COLUMN LAYOUT FOR DONUT CHARTS ---
        col_energy, col_solar, col_ev = st.columns(3)

        # --- Chart 1: TOTAL CLEAN ENERGY MIX DISTRIBUTION ---
        with col_energy:
            df_energy = get_pollution_free_energy_data(selected_cities_green)
            
            # 1. Aggregate total capacity across selected cities
            df_total_capacity = df_energy[['Solar (MW)', 'Wind (MW)', 'Hydro (MW)']].sum().reset_index()
            df_total_capacity.columns = ['Source', 'Capacity (MW)']
            
            # 2. Render Donut Chart
            # NOTE: We use the existing function calls to define the aesthetic.
            fig_energy_mix = px.pie(
                df_total_capacity,
                names='Source',
                values='Capacity (MW)',
                title='*Total Clean Energy Mix (MW)*',
                hole=0.5,
                color_discrete_sequence=CUSTOM_VIBRANT_PALETTE
            )
            fig_energy_mix.update_layout(height=350, margin=dict(l=5, r=5, t=30, b=10), showlegend=True)
            st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
            st.plotly_chart(fig_energy_mix, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Chart 2: SOLAR ADOPTION DISTRIBUTION (By City) ---
        with col_solar:
            df_solar_conn = get_registered_solar_connections(selected_cities_green)
            
            # Render Donut Chart (Distribution by City)
            fig_solar_dist = px.pie(
                df_solar_conn, 
                names='City', 
                values='Solar Connections', 
                title='*Solar Connections Distribution by City*', 
                hole=0.5,
                color_discrete_sequence=CUSTOM_VIBRANT_PALETTE
            )
            fig_solar_dist.update_layout(height=350, margin=dict(l=5, r=5, t=30, b=10), showlegend=True)
            st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.6s;">', unsafe_allow_html=True)
            st.plotly_chart(fig_solar_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


        # --- Chart 3: EV ADOPTION DISTRIBUTION (By City) ---
        with col_ev:
            df_ev = get_registered_ev_vehicles(selected_cities_green)
            
            # Render Donut Chart (Distribution by City)
            fig_ev_dist = px.pie(
                df_ev, 
                names='City', 
                values='Registered EVs', 
                title='*EV Registrations Distribution by City*', 
                hole=0.5,
                color_discrete_sequence=CUSTOM_VIBRANT_PALETTE
            )
            fig_ev_dist.update_layout(height=350, margin=dict(l=5, r=5, t=30, b=10), showlegend=True)
            st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.7s;">', unsafe_allow_html=True)
            st.plotly_chart(fig_ev_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.caption("This graph illustrates the adoption of electric vehicles, a key indicator for sustainable transportation.")

# --- CLEANLINESS RANK TAB CONTENT (Completed) ---
with TAB_SWACHH:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🧼</span>
                Swachh Survekshan (Cleanliness) Ranking Trend
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 0.3s;">
            <p>Track the performance of major cities in Uttar Pradesh and top national performers based on the annual cleanliness survey.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Get the complete list of UP Nagar Nigams and National toppers for the multiselect
    all_known_swachh_cities = get_swachh_ranking_data([], _name)['City'].unique().tolist()
    default_swachh_cities = sorted(list(set(all_known_swachh_cities + [_name, "Lucknow", "Indore", "Surat"])))

    selected_swachh_cities = st.multiselect(
        "Select cities for historical cleanliness ranking comparison:",
        options=default_swachh_cities,
        default=[_name, "Lucknow", "Indore", "Surat"],
        key="swachh_city_select",
        help="Select any major city in UP or the top national performers."
    )

    if not selected_swachh_cities:
        st.info("Please select at least one city to view the cleanliness ranking trend.")
    else:
        # UPDATED CALL: Passing current city name for proper default handling
        df_swachh = get_swachh_ranking_data(selected_swachh_cities, _name)
        
        if df_swachh.empty:
            st.warning("No data found for the selected cities.")
        else:
            st.markdown(
                """
                <div class="fade-in-element" style="--delay: 0.5s;">
                    <h3 class="animated-sub-header">
                        <span class="animated-emoji" style="--delay: 0.6s;">🏆</span> City Global Rank Trend
                    </h3>
                </div>
                """, unsafe_allow_html=True
            )
            # Melt the DataFrame for plotting (Years as categories)
            df_swachh_melted = df_swachh.melt(id_vars='City', var_name='Year', value_name='Global Rank')
            
            # Line chart showing Rank over 5 years
            fig_rank = px.line(
                df_swachh_melted,
                x='Year',
                y='Global Rank',
                color='City',
                title='*City Global Rank Trend (Lower Rank is Better)*',
                markers=True,
                labels={'Global Rank': 'Global Rank (Lower = Better)'},
                color_discrete_sequence=px.colors.qualitative.Bold 
            )
            
            fig_rank.update_yaxes(
                autorange="reversed", # Invert axis
                tickmode='linear',
                dtick=50, # Set major ticks at intervals
                showgrid=True,
                gridcolor='#e9ecef'
            )
            fig_rank.update_layout(
                plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529')
            )

            st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.8s;">', unsafe_allow_html=True)
            st.plotly_chart(fig_rank, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("This chart tracks the simulated global rank of cities over the past 5 years. A downward trend (lower rank number) indicates improvement in cleanliness.")
            
            st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="fade-in-element" style="--delay: 1.0s;">
                    <h3 class="animated-sub-header">Detailed Swachh Ranking Table</h3>
                </div>
                """, unsafe_allow_html=True
            )
            st.dataframe(df_swachh, use_container_width=True)

# --- WASTE MANAGEMENT TAB CONTENT (Completed) ---
with TAB_WASTE:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🗑️</span>
                Waste Management & City Accountability (TPD)
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    # 1. WASTE ACCOUNTABILITY MAP (UP and Metros)
    st.markdown(
        """
        <div class="fade-in-element" style="--delay: 0.3s;">
            <h3 class="animated-sub-header">🗺️ Solid Waste Generation (TPD) for Key Cities</h3>
        </div>
        """, unsafe_allow_html=True
    )

    df_map_waste = get_solid_waste_data()
    
    # Filter only UP cities for a focused comparison
    up_cities = ['Prayagraj', 'Lucknow', 'Varanasi', 'Kanpur', 'Noida', 'Ayodhya', 'Mirzapur', 'Agra', 'Meerut', 'Ghaziabad', 'Bareilly', 'Moradabad']
    df_up_and_selected = df_map_waste[df_map_waste['City'].isin(up_cities)].copy()
    
    # Add selected city if it's a metro not in UP list
    if _name not in df_up_and_selected['City'].tolist():
        selected_row = df_map_waste[df_map_waste['City'] == _name]
        df_up_and_selected = pd.concat([df_up_and_selected, selected_row], ignore_index=True)
        
    # Map Visualization (Scattergeo for bubble size based on TPD)
    fig_waste_map = go.Figure(data=go.Scattergeo(
        lon = df_up_and_selected['Longitude'],
        lat = df_up_and_selected['Latitude'],
        text = df_up_and_selected.apply(lambda row: f"{row['City']}<br>MSW: {row['Total_MSW_TPD']:,} TPD<br>Plastic: {row['Plastic_TPD']:,} TPD", axis=1),
        mode = 'markers',
        marker = dict(
            size = df_up_and_selected['Total_MSW_TPD'] / 30, # Scale marker size by TPD
            sizemode = 'area',
            sizemin = 5,
            color = df_up_and_selected['Total_MSW_TPD'],
            colorscale = px.colors.sequential.YlOrRd, # Warm color scale for emissions
            cmin = df_up_and_selected['Total_MSW_TPD'].min(),
            cmax = df_up_and_selected['Total_MSW_TPD'].max(),
            colorbar_title = "Total MSW (TPD)",
            line_color='#212529',
            line_width=1
        ),
    ))
    
    # Highlight current city with a prominent pink border
    current_city_data = df_up_and_selected[df_up_and_selected['City'].str.lower() == _name.lower()]
    if not current_city_data.empty:
        fig_waste_map.add_trace(go.Scattergeo(
            lon = current_city_data['Longitude'],
            lat = current_city_data['Latitude'],
            mode = 'markers',
            marker = dict(
                size = current_city_data['Total_MSW_TPD'].iloc[0] / 30 + 10, # Bigger size
                color = '#ff69b4', # Pink highlight
                line_width = 4,
                line_color = '#FFFFFF',
                opacity=0.8
            ),
            hoverinfo='none',
            name='Selected City'
        ))

    fig_waste_map.update_layout(
        title_text = f'Solid Waste Generation (TPD) in UP Cities & {_name}',
        showlegend = False,
        geo = dict(
            scope = 'asia',
            lonaxis_range= [70, 90], # Focus on UP/North India
            lataxis_range= [20, 32], 
            subunitcolor = "#007bff", 
            countrycolor = "#495057",
            bgcolor = '#f8f9fa',
            landcolor = '#ffffff',
            projection_type = 'mercator'
        ),
        height=660, # Increased by 20%
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529')
    )
    st.markdown('<div class="plot-wrap fade-in-element" style="--delay: 0.5s;">', unsafe_allow_html=True)
    st.plotly_chart(fig_waste_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Bubble size and color indicate the Total Municipal Solid Waste (TPD). Highlighted bubble is the current selected city.")
    
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)
    
    # 2. WASTE REDUCTION MODEL
    col_waste_details, col_waste_table = st.columns([0.6, 0.4])
    
    with col_waste_details:
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.7s;">
                <h3 class="animated-sub-header">🌱 Model-Driven Waste Reduction Strategy: The 4R Approach</h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        # Determine current waste status
        waste_status = "High Stress" if current_msw_tpd > 1000 else "Moderate Stress"
        
        st.markdown(f"""
        <div class="metric-card fade-in-element" style="--delay: 0.8s; border: 1px solid #38a3a5;">
            <p style='color: #212529; font-weight: 700; font-size: 1.1rem; margin-bottom: 5px;'>
                Current Waste Stress Level: <span style='color: #e63946;'>{waste_status}</span>
            </p>
            <p style='color: #495057;'>
                Based on the current MSW generation of **{current_msw_tpd:.0f} TPD**, the primary strategy must focus on achieving the **4R Principle (Reduce, Reuse, Recycle, Recover)**:
            </p>
            <ul style='color: #495057;'>
                <li>**REDUCE:** Implement steep fines for single-use plastic violations and promote bulk buying.</li>
                <li>**REUSE/RECYCLE:** Formalize the **informal waste-picker sector** to boost segregation efficiency (India's best recycling asset).</li>
                <li>**RECOVER:** Invest in **Decentralized Composting** for the high organic waste fraction and **Waste-to-Energy (WTE)** plants for non-recyclable Refuse-Derived Fuel (RDF).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_waste_table:
        st.markdown(
            """
            <div class="fade-in-element" style="--delay: 0.9s;">
                <h3 class="animated-sub-header">City Waste Detail</h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        df_display = df_msw[df_msw['City'].str.lower() == _name.lower()][['City', 'Total_MSW_TPD', 'Plastic_TPD', 'Predicted_TPD_Change_%']].rename(
            columns={'Total_MSW_TPD': 'Total MSW (TPD)', 'Plastic_TPD': 'Plastic (TPD)', 'Predicted_TPD_Change_%': 'Predicted Change (%)'}
        )
        st.markdown('<div class="fade-in-element" style="--delay: 1.0s;">', unsafe_allow_html=True)
        st.dataframe(df_display.transpose(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Predicted change is a simplified annual projection based on current consumption trends and waste management performance.")
    
# --- ABOUT PROJECT TAB CONTENT (Completed) ---
with TAB_ABOUT:
    # --- ANIMATED HEADER ---
    st.markdown(
        f"""
        <div class="fade-in-element" style="--delay: 0.1s;">
            <h2 class="animated-main-header">
                <span class="animated-emoji" style="--delay: 0.2s;">🚀</span>
                About SustainifyAI Project
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr class='gradient-separator'>", unsafe_allow_html=True)

    st.header("Hackathon Submission")
    st.markdown("<div class='team-title'>Nxt Gen Developers</div>", unsafe_allow_html=True)
    st.subheader(" Members")
    st.markdown("""
        * **Aditya Kumar Singh**
        * **Gaurang Verma**
        * **Vandana Yadav**
        * **Gaurav Shakya**
        * **Saurabh Shukla**
    """)
    
    st.markdown("---")
    
    st.subheader("Project: SustainifyAI — Bright Sustainability Tracker")
    
    st.markdown("### Why SustainifyAI was Created")
    st.markdown("""
        SustainifyAI was developed to bridge the **gap between raw climate data and actionable governance/citizen insight**. In regions facing rapid environmental changes, a tool that consolidates real-time air quality, historical climate trends, river health, and personalized impact analysis is crucial. Our goal is to make **complex environmental data accessible, visual, and predictive** for better planning.
    """)
    
    st.markdown("### How SustainifyAI Helps Government & Policymakers")
    st.markdown("""
        1. **Alerts & Anomaly Detection:** Provides immediate, location-specific warnings for PM2.5 or extreme heat, enabling **proactive health and safety responses** (e.g., advising school closures or issuing heatwave warnings).
        2. **Goal Tracking (e.g., Afforestation):** Quantifies city-specific targets (e.g., trees needed per capita), offering **measurable progress indicators** for green initiatives.
        3. **Resource Allocation:** The Sustainability Score and detailed sub-scores (Water Quality, Air Quality, etc.) highlight the **most critical areas requiring immediate investment and policy intervention**.
        4. **Long-term Planning:** AI forecasts provide a future outlook on temperatures and precipitation, essential for **infrastructure planning** (e.g., water management, drought preparation).
    """)
    
    st.markdown("### The Role of AI, ML, and Data Science")
    st.markdown("""
        * **Data Science & Engineering:** Used to **ingest, clean, and structure** vast amounts of historical climate data (ERA5 reanalysis) and real-time air quality data (Open-Meteo). We derive key metrics like **Monthly Norms** and **Warming Trends** using time-series analysis and moving averages.
        * **Machine Learning (ML) & AI Forecasting:**
        	* **Forecasting Models:** We use established ML/AI models—**Prophet** (for strong seasonality), **ARIMA** (for classical time-series analysis), and a **Random Forest ML Ensemble** (for non-linear trend capture)—to predict future climate variables like temperature and precipitation up to a year ahead.
        	* **Metric Validation:** Models are rigorously backtested using metrics like **MAE** and **MAPE** to ensure forecast accuracy before deployment.
        	* **Intelligent Proxies:** AI/ML is used in the Carbon Footprint tab to create **intelligent, localized auto-estimates** based on the city's overall sustainability metrics.
    """)

    st.markdown("### Technology Stack & Demonstration Platform")
    st.markdown("""
        | Component | Technology | Why We Used It |
        | :--- | :--- | :--- |
        | **Frontend/Demonstration** | **Streamlit & Plotly** | Streamlit enabled **rapid prototyping** and creating a complex, interactive web application entirely in Python. Plotly provides **futuristic, interactive, and mobile-friendly visualizations**. |
        | **Data/Backend (APIs)** | **Open-Meteo (ERA5 & AQ APIs)** | Provides free, high-quality, geographically granular historical climate and real-time air quality data, eliminating the need for complex API keys for demonstration. |
        | **Development Environment** | **VS Code** | Used for its robust Python support, rapid debugging, and seamless integration with this code. |
        | **ML/Statistical Models** | **Prophet, pmdarima, scikit-learn** | A robust collection of libraries for professional-grade time-series forecasting and ML ensemble creation. |
    """)

    st.markdown("---")
    
    st.subheader("Conclusion")
    st.markdown("""
        SustainifyAI is a proof-of-concept demonstrating a **unified, intelligent dashboard** capable of serving both citizens and government bodies with critical environmental intelligence. Our focus on **user experience, actionable metrics, and predictive AI** makes this a powerful tool for driving sustainability initiatives forward.
    """)

    st.markdown("---")
    
    # Final Button
    st.markdown("""
    <div style='text-align: center;'>
    	<div class="stButton btn-primary">
    		<button style='width: 100%; border-radius: 8px; padding: 10px 20px; font-size: 1.2rem;'>
    			Thank you for reviewing our project! 🙏
    		</button>
    	</div>
    </div>
    """, unsafe_allow_html=True)
