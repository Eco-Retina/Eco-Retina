import sys
import subprocess
import os
import traceback
import warnings
import codecs
import json
import ctypes  

if os.name == 'nt':
    try:
        # Create a unique identifier for your application
        myappid = 'ecoretina.workbench.' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

# --- FORCE UTF-8 ENCODING GLOBALLY TO PREVENT ASCII CRASHES ---
if sys.stdout and sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr and sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings("ignore")

# ==========================================
# 1. ROBUST AUTOMATIC DEPENDENCY INSTALLER
# ==========================================
def install_dependencies_and_restart():
    print("[INFO] Missing libraries detected. Starting background installation...")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo(
            "Installation Required", 
            "Essential libraries (AI, Machine Learning...) are missing on this computer.\n\n"
            "Automatic installation will begin. Please wait 1 to 2 minutes...\n"
            "The software will open automatically once it is finished!"
        )
        root.destroy()
    except Exception:
        pass

    # We DO NOT install 'mistralai' to avoid the import bug. We will use 'requests' instead.
    packages = [
        "scikit-learn", "pandas", "numpy", "scipy", "xgboost", 
        "statsmodels", "codecarbon", "customtkinter", "matplotlib", 
        "google-genai", "pillow", "openai", "anthropic", "groq", "requests"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages)
    
    print("[SUCCESS] Installation complete! Restarting application...")
    os.execv(sys.executable, ['python'] + sys.argv)

# Test critical imports. If even one fails, we launch the installer (only if non-compiled).
try:
    import customtkinter as ctk
    import pandas as pd
    import xgboost as xgb
    from google import genai
    import openai
    import anthropic
    from groq import Groq
    import requests
except ImportError as e:
    # getattr(sys, 'frozen', False) checks if the code is running via PyInstaller
    if getattr(sys, 'frozen', False):
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showerror(
            "Critical Import Error", 
            f"A required module was not included during PyInstaller compilation.\nTechnical error: {e}"
        )
        sys.exit(1)
    else:
        # Development mode (Classic Python)
        install_dependencies_and_restart()

# --- 2. FORCE PYTHON TO LOOK IN THE CURRENT SCRIPT DIRECTORY ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 3. STANDARD IMPORTS ---
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import time
import csv
import re
from datetime import datetime

import sklearn
import numpy as np
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- AI PROVIDERS IMPORTS ---
from google.genai import types

# --- CUSTOM ECO-RETINA IMPORT ---
from eco_retina import EcoRETINA
ECO_RETINA_AVAILABLE = True

def get_resource_path(relative_path):
    """ Returns the absolute path to the resource, necessary for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def create_tinted_icon(path, size, color):
    """ Loads a PNG with transparency, resizes it, and tints it with a solid color """
    try:
        img = Image.open(get_resource_path(path)).convert("RGBA")
        img = img.resize(size)
        solid = Image.new("RGBA", img.size, color)
        mask = img.split()[3]
        green_img = Image.composite(solid, img, mask)
        return ctk.CTkImage(green_img, size=size)
    except Exception:
        return None

# ==========================================
# AI AGENT ENGINE (MULTI-PROVIDER STREAMING)
# ==========================================
class EcoRetinaChatAgent:
    def __init__(self, api_key: str, provider: str):
        self.provider = provider
        self.api_key = api_key
        
        self.system_prompt = (
            "You are the Chief Econometrician and AI Support Guide for the EcoRETINA ML Workbench.\n\n"
            "ROLE 1 - WORKBENCH NAVIGATOR: Guide users if they are lost.\n"
            "- Tab 1 (Data): Import, handle outliers, encode dummies, drop columns, scale data.\n"
            "- Tab 2 (Algorithms): Select model, set hyperparameters, run training pipeline.\n"
            "- Tab 3 (Compare): Compare runs, analyze stats, view historical benchmarks.\n"
            "- Tab 5 (Predict): Load new datasets for Inference and export predictions.\n\n"
            "ROLE 2 - SENIOR ECONOMETRICIAN: When analyzing run metrics, provide a rigorous academic interpretation:\n"
            "1. Assess R-squared and Adjusted R-squared to explain the variance captured.\n"
            "2. Identify potential OVERFITTING by strictly comparing Train vs Test performance.\n"
            "3. Evaluate prediction accuracy using RMSE and MAPE.\n"
            "4. Analyze the Shapiro-Wilk p-value.\n"
            "5. Comment on environmental efficiency based on CodeCarbon emissions.\n\n"
            "ROLE 3 - ACTIVE DATA ENGINEER & ML OPERATOR (CRITICAL): If the user asks you to modify data, "
            "change an algorithm, tune hyperparameters, or run a model, you MUST execute it by adding exact command tags at the very end of your response.\n"
            "Available tags:\n"
            "[CMD_DROP_COL:column_name] -> Drops the specified column.\n"
            "[CMD_DROP_NA] -> Drops missing values.\n"
            "[CMD_ENCODE:column_name] -> One-Hot Encodes the column.\n"
            "[CMD_SET_ALGO:algo_name] -> Selects algorithm. Valid: EcoRETINA, OLS, Lasso, Ridge, ElasticNet, XGBoost, Random Forest, Neural Network.\n"
            "[CMD_SET_PARAM:widget_name=value] -> Modifies a parameter.\n"
            "[CMD_SET_TARGET:column_name] -> Sets the Target Variable (Y).\n"
            "[CMD_SELECT_PREDICTOR:column_name] -> Selects a specific predictor variable (Feature).\n"
            "[CMD_SELECT_ALL_PREDICTORS] -> Selects all available predictors automatically.\n"
            "[CMD_RUN_MODEL] -> Starts the training pipeline.\n"
            "Example: 'I have set XGBoost, selected the target, all predictors, and started training. [CMD_SET_ALGO:XGBoost][CMD_SET_TARGET:Price][CMD_SELECT_ALL_PREDICTORS][CMD_RUN_MODEL]'\n\n"
        )
        
        self.history = []
        
        if self.provider == "Google Gemini":
            self.client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.3 
            )
            self.chat = self.client.chats.create(model="gemini-2.5-flash", config=config)
            
        elif self.provider == "OpenAI (ChatGPT)":
            self.client = openai.OpenAI(api_key=api_key)
            self.history.append({"role": "system", "content": self.system_prompt})
            
        elif self.provider == "Anthropic (Claude)":
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif self.provider == "Mistral AI":
            # Direct API without external library
            self.history.append({"role": "system", "content": self.system_prompt})
            
        elif self.provider == "Groq":
            self.client = Groq(api_key=api_key)
            self.history.append({"role": "system", "content": self.system_prompt})

    def send_message_stream(self, text: str):
        max_retries = 7 # Allow up to 7 attempts to cover overloaded servers
        for attempt in range(max_retries):
            try:
                if self.provider == "Google Gemini":
                    response_stream = self.chat.send_message_stream(text)
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text 
                    return 
                    
                elif self.provider == "OpenAI (ChatGPT)":
                    self.history.append({"role": "user", "content": text})
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=self.history,
                        temperature=0.3,
                        stream=True
                    )
                    full_reply = ""
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_reply += content
                            yield content
                    self.history.append({"role": "assistant", "content": full_reply})
                    return

                elif self.provider == "Anthropic (Claude)":
                    self.history.append({"role": "user", "content": text})
                    full_reply = ""
                    with self.client.messages.stream(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0.3,
                        system=self.system_prompt,
                        messages=self.history
                    ) as stream:
                        for text_chunk in stream.text_stream:
                            full_reply += text_chunk
                            yield text_chunk
                    self.history.append({"role": "assistant", "content": full_reply})
                    return

                elif self.provider == "Mistral AI":
                    self.history.append({"role": "user", "content": text})
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    }
                    payload = {
                        "model": "mistral-large-latest",
                        "messages": self.history,
                        "temperature": 0.3,
                        "stream": True
                    }
                    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, stream=True)
                    response.raise_for_status() # Raises an error if HTTP code 4xx or 5xx
                    
                    full_reply = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str == "[DONE]": break
                                try:
                                    import json
                                    chunk_data = json.loads(data_str)
                                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                        content = chunk_data["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            full_reply += content
                                            yield content
                                except Exception: pass
                                    
                    self.history.append({"role": "assistant", "content": full_reply})
                    return

                elif self.provider == "Groq":
                    self.history.append({"role": "user", "content": text})
                    response = self.client.chat.completions.create(
                        model="llama-3.3-70b-versatile", # Updated model!
                        messages=self.history,
                        temperature=0.3,
                        stream=True
                    )
                    full_reply = ""
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_reply += content
                            yield content
                    self.history.append({"role": "assistant", "content": full_reply})
                    return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # --- ANTI-OVERLOAD PROTECTION (RATE LIMIT / TOO MANY REQUESTS / TOO LARGE) ---
                if any(err in error_str for err in ["429", "too many requests", "rate limit", "503", "quota", "overloaded", "too large", "capacity"]):
                    if attempt < max_retries - 1:
                        # 1. Remove the user message we just added to the history to avoid duplicates on the next attempt
                        if self.provider != "Google Gemini" and len(self.history) > 0 and self.history[-1]["role"] == "user":
                            self.history.pop()
                            
                        # 2. Exponential wait: 3s, 6s, 12s, 24s, 48s...
                        wait_time = 3 * (2 ** attempt) 
                        
                        # 3. Inform the user on the interface
                        yield f"\n[⏳ Server overloaded or too many requests. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})]\n"
                        time.sleep(wait_time)
                        
                        # 4. Restart the loop (next attempt)
                        continue 
                
                # If it's another error (invalid key) or we exhausted all our attempts
                yield f"\n[ERROR] AI API Error ({self.provider}): {str(e)}"
                return

# ==========================================
# WRAPPER FOR STATSMODELS OLS
# ==========================================
class OLSWrapper:
    def __init__(self, res):
        self.sm_model = res
    def predict(self, X):
        return self.sm_model.predict(X)

# ==========================================
# MODERN UI INITIALIZATION (CustomTkinter)
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class EcoRetinaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EcoRETINA ML Workbench PRO")
        #
        try:
            icon_path = get_resource_path("images/logoecoretinaV2.ico")
            self.root.iconbitmap(icon_path)
        except Exception:
            try:
                png_path = get_resource_path("images/logoecoretina.png")
                icon_img = tk.PhotoImage(file=png_path)
                self.root.iconphoto(False, icon_img)
            except Exception:
                pass
        self.root.geometry("1400x850") 
        self.root.minsize(1300, 800) 
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        try:
            self.root.state('zoomed')
        except tk.TclError:
            try:
                self.root.attributes('-zoomed', True)
            except:
                pass
        
        self.df = None
        self.df_predict = None  
        self.df_history = []  
        self.df_future = []   
        self.run_history = {}          
        self.latest_run_by_algo = {}   
        self.tree_tags_configured = False
        self.active_algo = "EcoRETINA"  
        self.ai_agent = None 
        self.hyperparams_visible = False
        
        self.target_var = tk.StringVar(value="")

        self.style = ttk.Style()
        self.style.theme_use("default")
        self.style.configure("Treeview", background="#2b2b2b", foreground="white", rowheight=35, fieldbackground="#2b2b2b", borderwidth=0, font=("Segoe UI", 14)) 
        self.style.map('Treeview', background=[('selected', '#1f6aa5')])
        self.style.configure("Treeview.Heading", background="#1f252b", foreground="white", borderwidth=1, font=("Segoe UI", 15, "bold")) 
        
        self.f_title = ctk.CTkFont(size=24, weight="bold")
        self.f_subtitle = ctk.CTkFont(size=20, weight="bold")
        self.f_text = ctk.CTkFont(size=16)
        
        self.build_header()
        
        self.body_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body_frame.pack(expand=True, fill="both")

        self.body_frame.columnconfigure(0, weight=1) 
        self.body_frame.columnconfigure(1, weight=0) 
        self.body_frame.rowconfigure(0, weight=1)

        self.main_view_frame = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        self.log_view_frame = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        self.tutorial_view_frame = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        
        self.main_view_frame.grid(row=0, column=0, padx=(20, 10), pady=(10, 20), sticky="nsew")

        self.tabview = ctk.CTkTabview(
            self.main_view_frame, 
            corner_radius=10,
            segmented_button_selected_color="#68B946",
            segmented_button_selected_hover_color="#539438",
            segmented_button_unselected_color="#1A1A1A"
        )
        self.tabview._segmented_button.configure(font=ctk.CTkFont(size=16, weight="bold"))
        self.tabview.pack(expand=True, fill="both")

        self.tab_data = self.tabview.add("1. Data & Pre-Processing")
        self.tab_algo = self.tabview.add("2. Algorithms & Params")
        self.tab_compare = self.tabview.add("3. Compare Results")
        self.tab_predict = self.tabview.add("4. Predict (New Data)") 
        
        self.build_data_tab()
        self.build_algo_tab()
        self.build_compare_tab()
        self.build_predict_tab()
        
        self.build_log_view()
        self.build_tutorial_view()
        self.build_ai_sidebar()
        
        self.log_event("Application initialized successfully. Ready to import dataset.")

    def on_closing(self):
        if messagebox.askyesno("Exit Confirmation", "Are you sure you want to exit EcoRETINA?\nAny unsaved data or results will be lost."):
            self.root.destroy()

    def bind_hover_scroll(self, widget):
        def _on_mousewheel(event):
            if sys.platform == "darwin":
                widget.yview_scroll(int(-1 * event.delta), "units")
            else:
                widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"
        def _linux_scroll_up(event):
            widget.yview_scroll(-1, "units")
            return "break"
        def _linux_scroll_down(event):
            widget.yview_scroll(1, "units")
            return "break"
        widget.bind("<MouseWheel>", _on_mousewheel)
        widget.bind("<Button-4>", _linux_scroll_up)
        widget.bind("<Button-5>", _linux_scroll_down)

    def build_header(self):
        header_frame = ctk.CTkFrame(self.root, height=70, corner_radius=0)
        header_frame.pack(fill='x', side='top')
        header_frame.pack_propagate(False)
        
        self.btn_menu = ctk.CTkButton(header_frame, text="≡", width=50, font=ctk.CTkFont(size=35, weight="bold"), command=self.show_nav_menu, fg_color="#3b3b3b", hover_color="#4b4b4b")
        self.btn_menu.pack(side='left', padx=(20, 10), pady=10)

        try:
            logo_path = get_resource_path("images/logoecoretina.png")
            logo_img = ctk.CTkImage(Image.open(logo_path), size=(50, 50)) 
            logo_label = ctk.CTkLabel(header_frame, image=logo_img, text="") 
            logo_label.pack(side='left', padx=(10, 10), pady=10)
        except Exception: pass

        self.btn_undo = ctk.CTkButton(header_frame, text="↩", width=50, command=self.undo, font=ctk.CTkFont(size=55, weight="bold"), state="disabled", fg_color="#3b3b3b", hover_color="#4b4b4b")
        self.btn_undo.pack(side='left', padx=(10, 5), pady=20)
        
        self.btn_redo = ctk.CTkButton(header_frame, text="↪", width=50, command=self.redo, font=ctk.CTkFont(size=55, weight="bold"), state="disabled", fg_color="#3b3b3b", hover_color="#4b4b4b")
        self.btn_redo.pack(side='left', padx=5, pady=20)

        self.btn_ai_toggle = ctk.CTkButton(header_frame, text="AI Assistant", font=ctk.CTkFont(size=16, weight="bold"), fg_color="#1f6aa5", hover_color="#144870", command=self.toggle_ai_sidebar)
        self.btn_ai_toggle.pack(side='right', padx=20, pady=20)

        ctk.CTkLabel(header_frame, text="EcoRetina Workbench", font=self.f_text, text_color="gray").pack(side='right', padx=10, pady=20)

    def show_nav_menu(self):
        nav_menu = tk.Menu(self.root, tearoff=0, bg="#1a1a1a", fg="white", font=("Segoe UI", 16, "bold"), activebackground="#1f6aa5", activeforeground="white")
        nav_menu.add_command(label="Main Workspace", command=lambda: self.switch_view("Main"))
        nav_menu.add_separator()
        nav_menu.add_command(label="Activity Log", command=lambda: self.switch_view("Activity Log"))
        nav_menu.add_command(label="Tutorial", command=lambda: self.switch_view("Tutorial"))
        
        x = self.btn_menu.winfo_rootx()
        y = self.btn_menu.winfo_rooty() + self.btn_menu.winfo_height()
        
        try:
            nav_menu.tk_popup(x, y)
        finally:
            nav_menu.grab_release()

    def switch_view(self, view_name):
        self.main_view_frame.grid_forget()
        self.log_view_frame.grid_forget()
        self.tutorial_view_frame.grid_forget()
        
        if view_name == "Main":
            self.main_view_frame.grid(row=0, column=0, padx=(20, 10), pady=(10, 20), sticky="nsew")
        elif view_name == "Activity Log":
            self.log_view_frame.grid(row=0, column=0, padx=(20, 10), pady=(10, 20), sticky="nsew")
        elif view_name == "Tutorial":
            self.tutorial_view_frame.grid(row=0, column=0, padx=(20, 10), pady=(10, 20), sticky="nsew")

    def build_ai_sidebar(self):
        self.ai_sidebar = ctk.CTkFrame(self.body_frame, width=450, corner_radius=10)
        self.ai_sidebar.pack_propagate(False)
        self.ai_sidebar.grid_propagate(False)

        top_frame = ctk.CTkFrame(self.ai_sidebar, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(top_frame, text="AI Copilot", font=self.f_title, text_color="#68B946").pack(side="left")
        ctk.CTkButton(top_frame, text="Close", width=50, fg_color="#3b3b3b", hover_color="#ef4444", command=self.toggle_ai_sidebar, font=self.f_text).pack(side="right")

        self.model_provider_var = tk.StringVar(value="Google Gemini")
        self.model_selector = ctk.CTkComboBox(
            self.ai_sidebar,
            values=["Google Gemini", "OpenAI (ChatGPT)", "Anthropic (Claude)", "Mistral AI", "Groq"],
            variable=self.model_provider_var,
            command=self.update_api_placeholder,
            font=self.f_text
        )
        self.model_selector.pack(fill="x", padx=10, pady=(10, 5))

        self.api_key_entry = ctk.CTkEntry(self.ai_sidebar, placeholder_text="Paste Gemini API Key (AIza...)", show="*", font=self.f_text)
        self.api_key_entry.pack(fill="x", padx=10, pady=5)
        
        # --- BUTTONS CONNECT / DISCONNECT ---
        btn_conn_frame = ctk.CTkFrame(self.ai_sidebar, fg_color="transparent")
        btn_conn_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_connect = ctk.CTkButton(btn_conn_frame, text="Connect", command=self.init_ai_agent, fg_color="#3b3b3b", hover_color="#4b4b4b", font=self.f_text)
        self.btn_connect.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        self.btn_disconnect = ctk.CTkButton(btn_conn_frame, text="Disconnect", command=self.disconnect_ai_agent, state="disabled", fg_color="#ef4444", hover_color="#dc2626", font=self.f_text)
        self.btn_disconnect.pack(side="right", expand=True, fill="x", padx=(5, 0))
        self.btn_analyze = ctk.CTkButton(self.ai_sidebar, text="Analyze Last Run", command=self.analyze_last_run, state="disabled", fg_color="#1f6aa5", hover_color="#144870", font=self.f_text)
        self.btn_analyze.pack(fill="x", padx=10, pady=(15, 5))
        
        self.btn_tune = ctk.CTkButton(self.ai_sidebar, text="Auto-Tune Hyperparameters", command=self.tune_last_model, state="disabled", fg_color="#164A31", hover_color="#22c55e", font=self.f_text)
        self.btn_tune.pack(fill="x", padx=10, pady=(0, 15))

        self.chat_scroll = ctk.CTkScrollableFrame(self.ai_sidebar, fg_color="#1e1e1e", corner_radius=10)
        self.chat_scroll.pack(expand=True, fill="both", padx=10, pady=5)
        
        self.current_ai_label = None
        self.current_ai_text = ""

        self.add_chat_bubble("System", "Select your LLM and provide the API Key to initialize...")

        input_frame = ctk.CTkFrame(self.ai_sidebar, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=10)
        
        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text="Ask a question...", height=40, font=self.f_text)
        self.chat_input.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.send_ai_message())

        ctk.CTkButton(input_frame, text="Send", width=60, height=40, fg_color="#68B946", hover_color="#539438", command=self.send_ai_message, font=self.f_text).pack(side="right")

    def update_api_placeholder(self, choice):
        placeholders = {
            "Google Gemini": "Paste Gemini API Key (AIza...)",
            "OpenAI (ChatGPT)": "Paste OpenAI API Key (sk-...)",
            "Anthropic (Claude)": "Paste Anthropic Key (sk-ant-...)",
            "Mistral AI": "Paste Mistral API Key",
            "Groq": "Paste Groq API Key (gsk_...)"
        }
        self.api_key_entry.configure(placeholder_text=placeholders.get(choice, "Paste API Key"))

    def toggle_ai_sidebar(self):
        if self.ai_sidebar.winfo_ismapped():
            self.ai_sidebar.grid_forget()
        else:
            self.ai_sidebar.grid(row=0, column=1, padx=(0, 20), pady=(10, 20), sticky="nsew")

    def add_chat_bubble(self, sender, text):
        container = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        container.pack(fill="x", pady=5)

        if sender == "User":
            bg_color = "#1f6aa5"
            anchor_pos = "e"
            txt_color = "white"
            justify = "right"
        elif sender == "AI":
            bg_color = "#2b2b2b" 
            anchor_pos = "w"     
            txt_color = "white"
            justify = "left"
        else:
            bg_color = "transparent"
            anchor_pos = "center"
            txt_color = "gray"
            justify = "center"

        bubble = ctk.CTkFrame(container, fg_color=bg_color, corner_radius=15)
        bubble.pack(anchor=anchor_pos, padx=10)

        label = ctk.CTkLabel(bubble, text=text, text_color=txt_color, justify=justify, wraplength=320, font=self.f_text)
        label.pack(padx=15, pady=10)

        self.root.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)
        
        return label 

    def init_ai_agent(self):
        key = self.api_key_entry.get().strip()
        provider = self.model_provider_var.get()
        if not key:
            messagebox.showwarning("Missing Key", f"Please paste your {provider} API Key.")
            return
        try:
            self.ai_agent = EcoRetinaChatAgent(api_key=key, provider=provider)
            self.add_chat_bubble("System", f"Agent connected successfully via {provider}! How can I help you?")
            self.btn_analyze.configure(state="normal")
            self.btn_tune.configure(state="normal")
            self.api_key_entry.configure(state="disabled")
            self.model_selector.configure(state="disabled")
            self.btn_connect.configure(state="disabled")
            self.btn_disconnect.configure(state="normal")
            
            # ---> ADDED THIS LINE TO ACTIVATE THE TEXT AREA <---
            self.chat_input.configure(state="normal") 
            
        except Exception as e:
            messagebox.showerror("Agent Error", f"Failed to connect:\n{str(e)}")
    

    def disconnect_ai_agent(self):
      
        self.ai_agent = None
        
        self.api_key_entry.configure(state="normal")
        self.model_selector.configure(state="normal")
        self.btn_connect.configure(state="normal")
        
        self.btn_disconnect.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        self.btn_tune.configure(state="disabled")
        self.chat_input.configure(state="disabled")
        
        self.add_chat_bubble("System", "🔴 Agent disconnected. You can now select a different model and enter a new API key.")

    def _append_ai_chunk(self, text_chunk):
        if self.current_ai_label:
            if "⏳" in self.current_ai_text:
                self.current_ai_text = ""
                
            self.current_ai_text += text_chunk
            self.current_ai_label.configure(text=self.current_ai_text)
            self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _finish_ai_stream(self):
        full_text = self.current_ai_text
        commands_executed = []
        
        # 1. COMMANDES DATA
        if self.df is not None:
            drop_cols = re.findall(r'\[CMD_DROP_COL:(.*?)\]', full_text)
            for col in drop_cols:
                col = col.strip()
                if col in self.df.columns:
                    self.save_state(f"AI dropped {col}")
                    self.df = self.df.drop(columns=[col])
                    commands_executed.append(f"Dropped column: '{col}'")
            
            if '[CMD_DROP_NA]' in full_text:
                self.save_state("AI dropped NA")
                initial_len = len(self.df)
                self.df = self.df.dropna()
                commands_executed.append(f"Dropped {initial_len - len(self.df)} rows with missing values (NA)")
            
            encode_cols = re.findall(r'\[CMD_ENCODE:(.*?)\]', full_text)
            for col in encode_cols:
                col = col.strip()
                if col in self.df.columns:
                    self.save_state(f"AI encoded {col}")
                    dummies = pd.get_dummies(self.df[col], prefix=col).astype(int)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                    commands_executed.append(f"One-Hot Encoded column: '{col}'")
        
        # 2. COMMANDE : CHANGER D'ALGORITHME
        set_algo = re.findall(r'\[CMD_SET_ALGO:(.*?)\]', full_text)
        for algo in set_algo:
            algo = algo.strip()
            valid_algos = ["EcoRETINA", "OLS", "Lasso", "Ridge", "ElasticNet", "XGBoost", "Random Forest", "Neural Network"]
            if algo in valid_algos:
                self.tabview.set("2. Algorithms & Params")
                self.show_hyperparams_page(algo)
                commands_executed.append(f"Selected algorithm: '{algo}'")
                self.root.update_idletasks() # Force l'affichage pour que les listes de variables se créent

        # 3. COMMANDE : CHANGER LES HYPERPARAMÈTRES
        set_params = re.findall(r'\[CMD_SET_PARAM:(.*?)\]', full_text)
        for param_str in set_params:
            if "=" in param_str:
                param_name, param_value = param_str.split("=", 1)
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                if hasattr(self, param_name):
                    widget = getattr(self, param_name)
                    try:
                        if isinstance(widget, (tk.StringVar, tk.IntVar, tk.DoubleVar)):
                            widget.set(param_value)
                        elif isinstance(widget, ctk.CTkEntry):
                            widget.delete(0, 'end')
                            widget.insert(0, param_value)
                        elif isinstance(widget, ctk.CTkComboBox):
                            widget.set(param_value)
                        commands_executed.append(f"Set '{param_name}' to {param_value}")
                    except Exception as e:
                        pass

        # 4. COMMANDE : CIBLE ET PRÉDICTEURS (NOUVEAU)
        set_target = re.findall(r'\[CMD_SET_TARGET:(.*?)\]', full_text)
        for target in set_target:
            target = target.strip()
            if self.df is not None and target in self.df.columns:
                self.target_var.set(target)
                if hasattr(self, 'target_combo') and self.target_combo.winfo_exists():
                    self.target_combo.set(target)
                commands_executed.append(f"Set Target Variable to: '{target}'")

        if '[CMD_SELECT_ALL_PREDICTORS]' in full_text:
            # Active les cases "Select All" selon le modèle choisi
            if hasattr(self, 'select_all_feat_var') and hasattr(self, 'list_features'):
                self.select_all_feat_var.set(1)
                self.toggle_features()
            if hasattr(self, 'select_all_cont_var') and hasattr(self, 'list_cont'):
                self.select_all_cont_var.set(1)
                self.toggle_cont()
            if hasattr(self, 'select_all_dum_var') and hasattr(self, 'list_dummy'):
                self.select_all_dum_var.set(1)
                self.toggle_dum()
            commands_executed.append("Selected all available predictors")
            
        select_preds = re.findall(r'\[CMD_SELECT_PREDICTOR:(.*?)\]', full_text)
        for pred in select_preds:
            pred = pred.strip()
            def select_in_listbox(lb_name):
                if hasattr(self, lb_name):
                    lbox = getattr(self, lb_name)
                    if lbox and lbox.winfo_exists():
                        for i in range(lbox.size()):
                            if lbox.get(i) == pred:
                                lbox.select_set(i)
                                return True
                return False
            
            found = False
            if select_in_listbox('list_features'): found = True
            if select_in_listbox('list_cont'): found = True
            if select_in_listbox('list_dummy'): found = True
            if found:
                commands_executed.append(f"Selected predictor: '{pred}'")

        # 5. Nettoyage du texte affiché
        clean_text = re.sub(r'\[CMD_.*?\]', '', full_text).strip()
        if self.current_ai_label:
            self.current_ai_label.configure(text=clean_text)
        
        # 6. Affichage des actions exécutées
        if commands_executed:
            if not set_algo: # On rafraichit seulement si on n'a pas déjà recréé la page d'algo
                self.refresh_workspace()
            action_summary = "\n".join(f"- {cmd}" for cmd in commands_executed)
            self.add_chat_bubble("System", f"Magic Actions applied by AI:\n{action_summary}")

        # 7. COMMANDE : LANCER L'ENTRAÎNEMENT
        if '[CMD_RUN_MODEL]' in full_text:
            self.add_chat_bubble("System", "🚀 AI triggered the training pipeline...")
            self.tabview.set("2. Algorithms & Params")
            
            # --- FILET DE SÉCURITÉ : Auto-sélection si l'IA a oublié ---
            # Pour les modèles classiques (XGBoost, OLS, etc.)
            if hasattr(self, 'list_features') and len(self.list_features.curselection()) == 0:
                if hasattr(self, 'select_all_feat_var'):
                    self.select_all_feat_var.set(1)
                    self.toggle_features()
                    
            # Pour le modèle EcoRETINA (qui sépare variables continues et dummy)
            if hasattr(self, 'list_cont') and len(self.list_cont.curselection()) == 0:
                if hasattr(self, 'select_all_cont_var'):
                    self.select_all_cont_var.set(1)
                    self.toggle_cont()
            # ------------------------------------------------------------

            # On attend 500ms pour laisser l'interface cocher les cases visuellement avant de lancer les calculs
            self.root.after(500, self.start_training)

        self.chat_input.configure(state="normal")
        if hasattr(self, 'btn_analyze'): self.btn_analyze.configure(state="normal")
        if hasattr(self, 'btn_tune'): self.btn_tune.configure(state="normal")
        self.chat_input.focus()

    def send_ai_message(self):
        if not self.ai_agent: return
        msg = self.chat_input.get().strip()
        if not msg: return
        
        self.add_chat_bubble("User", msg)
        self.current_ai_text = "⏳ Thinking..."
        self.current_ai_label = self.add_chat_bubble("AI", self.current_ai_text)
        
        self.chat_input.delete(0, 'end')
        self.chat_input.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        self.btn_tune.configure(state="disabled")
        
        def fetch_stream():
            try:
                for chunk in self.ai_agent.send_message_stream(msg):
                    self.root.after(0, self._append_ai_chunk, chunk) 
            except Exception as e:
                self.root.after(0, self._append_ai_chunk, f"\n[ERROR]: {str(e)}")
            finally:
                self.root.after(0, self._finish_ai_stream)
            
        threading.Thread(target=fetch_stream, daemon=True).start()

    def analyze_last_run(self):
        if not self.run_history:
            messagebox.showinfo("No Runs", "Please run a training pipeline first.")
            return
        
        last_run_id = list(self.run_history.keys())[-1]
        run_data = self.run_history[last_run_id]
        m = run_data['metrics']
        
        prompt = (
            f"Please analyze my latest ML pipeline execution:\n"
            f"- Model: {run_data['model_name']}\n"
            f"- Train R2: {m['R2_Train']:.4f} | Test R2: {m['R2_Test']:.4f}\n"
            f"- Test RMSE: {m['RMSE_Test']:.4f} | Test MAPE: {m['MAPE_Test']:.2f}%\n"
            f"- CodeCarbon Emissions: {m['Emissions']:.6f} kgCO2eq\n"
            f"Explain if there is overfitting, evaluate the performance, and comment on the environmental impact."
        )
        
        self.add_chat_bubble("User", f"[Action: Analyze Run '{last_run_id}']")
        self.current_ai_text = "⏳ Analyzing your metrics..."
        self.current_ai_label = self.add_chat_bubble("AI", self.current_ai_text)
        
        self.chat_input.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        self.btn_tune.configure(state="disabled")
        
        def fetch_stream():
            try:
                for chunk in self.ai_agent.send_message_stream(prompt):
                    self.root.after(0, self._append_ai_chunk, chunk)
            except Exception as e:
                self.root.after(0, self._append_ai_chunk, f"\n[ERROR]: {str(e)}")
            finally:
                self.root.after(0, self._finish_ai_stream)
                
        threading.Thread(target=fetch_stream, daemon=True).start()

    def profile_dataset(self):
        if self.df is None:
            messagebox.showwarning("Data Required", "Please load a dataset first.")
            return

        if not self.ai_agent:
            if not self.ai_sidebar.winfo_ismapped(): self.toggle_ai_sidebar()
            messagebox.showinfo("AI Connection", "Please connect the AI Assistant first.")
            return

        if not self.ai_sidebar.winfo_ismapped():
            self.toggle_ai_sidebar()

        buf = []
        buf.append(f"ROWS: {len(self.df)} | COLUMNS: {len(self.df.columns)}")
        buf.append("\nMISSING VALUES PER COLUMN:")
        buf.append(self.df.isnull().sum().to_string())
        buf.append("\nNUMERIC STATISTICS (Min/Max/Mean):")
        buf.append(self.df.describe().to_string())
        data_summary = "\n".join(buf)

        prompt = (
            "I have loaded a new dataset into the EcoRETINA Workbench. Please act as a Data Profiler.\n"
            f"Here is the raw statistical summary:\n{data_summary}\n\n"
            "Analyze this summary. Point out any potential anomalies (like extreme max values, missing data), "
            "and tell me exactly which variables I should drop, clip, or encode using the 'Data & Pre-Processing' tab."
        )

        self.add_chat_bubble("User", "[Action: Profile Dataset]")
        self.current_ai_text = "⏳ Scanning your dataset for anomalies..."
        self.current_ai_label = self.add_chat_bubble("AI", self.current_ai_text)

        self.chat_input.configure(state="disabled")
        if hasattr(self, 'btn_analyze'): self.btn_analyze.configure(state="disabled")
        if hasattr(self, 'btn_tune'): self.btn_tune.configure(state="disabled")

        def fetch_stream():
            try:
                for chunk in self.ai_agent.send_message_stream(prompt):
                    self.root.after(0, self._append_ai_chunk, chunk)
            except Exception as e:
                self.root.after(0, self._append_ai_chunk, f"\n[ERROR]: {str(e)}")
            finally:
                self.root.after(0, self._finish_ai_stream)

        threading.Thread(target=fetch_stream, daemon=True).start()

    def tune_last_model(self):
        if not self.run_history:
            messagebox.showinfo("No Runs", "Please run a training pipeline first.")
            return
        
        last_run_id = list(self.run_history.keys())[-1]
        run_data = self.run_history[last_run_id]
        m = run_data['metrics']
        config = run_data.get('config', {})

        params_str = "Default"
        if config:
            if config['algo'] == 'EcoRETINA':
                params_str = str(config.get('eco_kwargs', {}))
            else:
                params_str = str(config.get('other_kwargs', {}))

        prompt = (
            f"My latest model ({run_data['model_name']}) has the following metrics:\n"
            f"- Train R2: {m['R2_Train']:.4f} | Test R2: {m['R2_Test']:.4f}\n"
            f"- Train MAPE: {m['MAPE_Train']:.2f}% | Test MAPE: {m['MAPE_Test']:.2f}%\n"
            f"The hyperparameters used were:\n{params_str}\n\n"
            "As an AI Coach, please tell me if this model is overfitting or underfitting. "
            "Then, give me 3 specific, actionable recommendations on how to adjust these hyperparameters in the interface to improve the Test metrics."
        )

        self.add_chat_bubble("User", f"[Action: Auto-Tune '{last_run_id}']")
        self.current_ai_text = "⏳ Searching for parameter optimizations..."
        self.current_ai_label = self.add_chat_bubble("AI", self.current_ai_text)

        self.chat_input.configure(state="disabled")
        self.btn_analyze.configure(state="disabled")
        self.btn_tune.configure(state="disabled")

        def fetch_stream():
            try:
                for chunk in self.ai_agent.send_message_stream(prompt):
                    self.root.after(0, self._append_ai_chunk, chunk)
            except Exception as e:
                self.root.after(0, self._append_ai_chunk, f"\n[ERROR]: {str(e)}")
            finally:
                self.root.after(0, self._finish_ai_stream)

        threading.Thread(target=fetch_stream, daemon=True).start()

    # ==========================================
    # LOG SYSTEM (NOW IN ITS OWN VIEW)
    # ==========================================
    def build_log_view(self):
        log_frame = ctk.CTkFrame(self.log_view_frame, corner_radius=10, fg_color="transparent")
        log_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        header_frame = ctk.CTkFrame(log_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(header_frame, text="System Activity Log", font=ctk.CTkFont(size=30, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame, text="Export Log to TXT", font=self.f_text, command=self.export_log, fg_color="#3b3b3b", hover_color="#4b4b4b").pack(side="right")
        ctk.CTkButton(header_frame, text="Clear Log", font=self.f_text, command=self.clear_log, fg_color="#ef4444", hover_color="#dc2626", width=100).pack(side="right", padx=10)

        self.log_textbox = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=16), wrap="word")
        self.log_textbox.pack(expand=True, fill="both")
        self.log_textbox.configure(state="disabled")

    def log_event(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        if hasattr(self, 'log_textbox') and self.log_textbox.winfo_exists():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", log_line)
            self.log_textbox.see("end") 
            self.log_textbox.configure(state="disabled")
            
    def export_log(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=f"EcoRetina_Log_{datetime.now().strftime('%Y%m%d')}.txt", filetypes=[("Text File", "*.txt")])
        if filepath:
            try:
                log_content = self.log_textbox.get("1.0", "end-1c")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(log_content)
                self.log_event(f"Activity log successfully exported to: {filepath}")
                messagebox.showinfo("Export Success", f"Activity log saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save log:\n{str(e)}")

    def clear_log(self):
        if messagebox.askyesno("Clear Log", "Are you sure you want to clear the activity log?"):
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.configure(state="disabled")
            self.log_event("Log cleared by user.")

    # ==========================================
    # TUTORIAL VIEW (REDESIGNED)
    # ==========================================
    def build_tutorial_view(self):
        tutorial_frame = ctk.CTkFrame(self.tutorial_view_frame, corner_radius=10, fg_color="transparent")
        tutorial_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(tutorial_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header_frame, text="Tutorial & Documentation", font=ctk.CTkFont(size=30, weight="bold")).pack(side="left")

        # Scrollable container for text cards
        scroll_content = ctk.CTkScrollableFrame(tutorial_frame, fg_color="#1e1e1e", corner_radius=10)
        scroll_content.pack(expand=True, fill="both")

        # Structured Tutorial Content (English)
        tutorial_sections = [
            (
                "Welcome to EcoRETINA ML Workbench PRO", "🚀",
                "This platform is designed to streamline your Machine Learning and econometrics pipeline, while automatically tracking the carbon footprint of your models (via CodeCarbon).\nHere is how to get started:"
            ),
            (
                "1. Data & Pre-Processing", "📊",
                "• Import your dataset (CSV, Excel, JSON).\n"
                "• Choose your data splitting strategy (Train/Test Split or K-Fold Cross Validation).\n"
                "• Manage outliers by capping (Clip) or removing (Drop) rows outside thresholds.\n"
                "• Encode your categorical variables (One-Hot Encoding) with one click.\n"
                "• Tip: Ask the AI Assistant to profile your data and suggest necessary cleaning steps!"
            ),
            (
                "2. Algorithms & Hyperparameters", "⚙️",
                "• Select an algorithm from the available options (EcoRETINA, OLS, XGBoost, Random Forest, Neural Networks, etc.).\n"
                "• Choose the Target Variable (Y) at the top of the parameter page.\n"
                "• Expand the 'Show Hyperparameters' menu to fine-tune the model configuration.\n"
                "• Select your predictor variables (Features) from the scrollable lists.\n"
                "• Start training with the 'Run Model' button."
            ),
            (
                "3. Compare Results", "📈",
                "• Once training is complete, benchmark results appear in this dashboard.\n"
                "• The best metrics (R², MAPE, RMSE, CO2 Emissions) are automatically highlighted in green with an asterisk (*).\n"
                "• Magic Action: RIGHT-CLICK (or two-finger tap on Mac) a row in the table to open the Detailed Statistical Report (visual charts, feature importance, and mathematical equations)."
            ),
            (
                "4. Predict (Inference on New Data)", "🔮",
                "• Load a new dataset (which does not contain the target variable).\n"
                "• Select a previously trained model from the dropdown menu.\n"
                "• Click 'Run Prediction'. The software will apply the model's learned coefficients/rules to predict the missing values.\n"
                "• Visualize or export the new dataset (with the appended predictions) as a CSV."
            ),
            (
                "5. AI Copilot Assistant", "🤖",
                "• Click 'AI Assistant' in the top right to open the side panel.\n"
                "• Select your preferred AI Engine (Gemini, OpenAI, Anthropic, Mistral, Groq) and paste your API key.\n"
                "• The AI can analyze your training metrics, advise you on hyperparameter tuning, or even execute real-time data cleaning commands for you within the interface!"
            )
        ]

        for title, icon, body in tutorial_sections:
            card = ctk.CTkFrame(scroll_content, fg_color="#2b2b2b", corner_radius=10)
            card.pack(fill="x", pady=10, padx=10)

            ctk.CTkLabel(card, text=f"{icon}  {title}", font=self.f_subtitle, text_color="#68B946", anchor="w", justify="left").pack(fill="x", padx=20, pady=(15, 5))
            
            ctk.CTkLabel(card, text=body, font=self.f_text, justify="left", anchor="w").pack(fill="x", padx=20, pady=(0, 20))
    # ==========================================
    # UNDO / REDO LOGIC
    # ==========================================
    def save_state(self, action_name="Data modified"):
        if self.df is not None:
            self.df_history.append((action_name, self.df.copy()))
            if len(self.df_history) > 15:
                self.df_history.pop(0)
            self.df_future.clear() 
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        if hasattr(self, 'btn_undo'):
            self.btn_undo.configure(state="normal" if self.df_history else "disabled")
            self.btn_redo.configure(state="normal" if self.df_future else "disabled")

    def undo(self):
        if self.df_history:
            action_name, old_df = self.df_history.pop()
            self.df_future.append((action_name, self.df.copy()))
            self.df = old_df
            self.refresh_workspace()
            self.update_undo_redo_buttons()
            self.log_event(f"Undo action applied: Reversed '{action_name}'.")

    def redo(self):
        if self.df_future:
            action_name, future_df = self.df_future.pop()
            self.df_history.append((action_name, self.df.copy()))
            self.df = future_df
            self.refresh_workspace()
            self.update_undo_redo_buttons()
            self.log_event(f"Redo action applied: Restored '{action_name}'.")

    # ==========================================
    # HELPER FOR COLLAPSIBLE MENU
    # ==========================================
    def create_collapsible(self, parent, title_text):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill='x', pady=5, padx=10)
        
        btn = ctk.CTkButton(
            container, 
            text=f"▶  {title_text}", 
            anchor="w", 
            font=self.f_subtitle, 
            fg_color="#1f252b", 
            text_color="white", 
            hover_color="#2b3238"
        )
        btn.pack(fill='x')
        
        content_frame = ctk.CTkFrame(container, fg_color="#2b2b2b", corner_radius=5)
        
        def toggle():
            if content_frame.winfo_ismapped():
                content_frame.pack_forget()
                btn.configure(text=f"▶  {title_text}")
            else:
                content_frame.pack(fill='x', pady=(2,0))
                btn.configure(text=f"▼  {title_text}")
                
        btn.configure(command=toggle)
        return content_frame

    # ==========================================
    # TAB 1: DATA IMPORT & PRE-PROCESSING
    # ==========================================
    def build_data_tab(self):
        self.data_scroll = ctk.CTkScrollableFrame(self.tab_data, fg_color="transparent")
        self.data_scroll.pack(fill='both', expand=True)

        self.load_frame = ctk.CTkFrame(self.data_scroll, corner_radius=10)
        self.load_frame.pack(fill='x', padx=10, pady=10)
        
        ctk.CTkLabel(self.load_frame, text="1. Dataset Import", font=self.f_subtitle).pack(anchor="center", pady=(15, 5))
        
        inner_load = ctk.CTkFrame(self.load_frame, fg_color="transparent")
        inner_load.pack(expand=True, fill='none', pady=(10, 20))
        
        btn_load = ctk.CTkButton(inner_load, text="Browse Dataset", font=ctk.CTkFont(size=18, weight="bold"), command=self.load_file, fg_color="#68B946", hover_color="#539438")
        btn_load.grid(row=0, column=0, padx=10)
        
        ctk.CTkLabel(inner_load, text="Sep:", font=self.f_text).grid(row=0, column=1, padx=(10, 5))
        self.csv_sep = ctk.CTkComboBox(inner_load, values=["Auto", ",", ";", "\t", "|"], font=self.f_text, width=80)
        self.csv_sep.set("Auto")
        self.csv_sep.grid(row=0, column=2, padx=5)
        
        self.lbl_file = ctk.CTkLabel(inner_load, text="No file loaded.", font=self.f_text, text_color="gray")
        self.lbl_file.grid(row=0, column=3, padx=15)

        self.preprocessing_frame = ctk.CTkFrame(self.data_scroll, fg_color="transparent")

        self.frame_config = self.create_collapsible(self.preprocessing_frame, "2. Sample Split parameters")
        grid_frame = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        grid_frame.pack(fill='x', padx=20, pady=15)
        grid_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(grid_frame, text="Split Strategy:", font=self.f_text).grid(row=0, column=0, padx=(0, 10), pady=10, sticky='w')
        self.strategy_var = tk.StringVar(value="Train/Test Split")
        self.strategy_menu = ctk.CTkOptionMenu(grid_frame, values=["Train/Test Split", "K-Fold Cross Validation"], variable=self.strategy_var, command=self.on_strategy_change, font=self.f_text)
        self.strategy_menu.grid(row=0, column=1, pady=10, sticky='w')

        self.dynamic_params_frame = ctk.CTkFrame(grid_frame, fg_color="transparent")
        self.dynamic_params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.dynamic_params_frame.grid_columnconfigure(1, weight=1)

        self.train_test_frame = ctk.CTkFrame(self.dynamic_params_frame, fg_color="transparent")
        self.train_test_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.train_test_frame, text="Train Split (%):", font=self.f_text).grid(row=0, column=0, padx=(0, 10), pady=10, sticky='w')
        slider_frame = ctk.CTkFrame(self.train_test_frame, fg_color="transparent")
        slider_frame.grid(row=0, column=1, pady=10, sticky='ew')
        self.split_var = tk.DoubleVar(value=80.0)
        self.split_str_var = tk.StringVar(value="80.0")
        ctk.CTkLabel(slider_frame, text="%", font=ctk.CTkFont(size=16, weight="bold")).pack(side='right', padx=(2, 0))
        self.split_entry = ctk.CTkEntry(slider_frame, textvariable=self.split_str_var, font=self.f_text, width=70, justify="center")
        self.split_entry.pack(side='right', padx=(10, 0))
        split_slider = ctk.CTkSlider(slider_frame, from_=50, to=100, variable=self.split_var, command=self.update_split_from_slider)
        split_slider.pack(side='left', fill='x', expand=True)
        self.split_str_var.trace_add("write", self.update_split_from_entry)

        self.kfold_frame = ctk.CTkFrame(self.dynamic_params_frame, fg_color="transparent")
        self.kfold_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.kfold_frame, text="Number of Folds (K):", font=self.f_text).grid(row=0, column=0, padx=(0, 10), pady=10, sticky='w')
        self.k_var = tk.IntVar(value=5)
        self.k_entry = ctk.CTkEntry(self.kfold_frame, textvariable=self.k_var, font=self.f_text, width=70, justify="center")
        self.k_entry.grid(row=0, column=1, pady=10, sticky='w')

        self.on_strategy_change(self.strategy_var.get())

        self.frame_outliers = self.create_collapsible(self.preprocessing_frame, "3. Outlier Management")
        ctk.CTkLabel(self.frame_outliers, text="Numeric Variable:", font=self.f_text).pack(anchor="w", padx=20, pady=(15, 0))
        self.outlier_var_combo = ctk.CTkComboBox(self.frame_outliers, font=self.f_text, values=[], command=self.on_outlier_var_select)
        self.outlier_var_combo.pack(fill='x', padx=20, pady=5)
        self.outlier_info_text = ctk.CTkTextbox(self.frame_outliers, height=130, font=ctk.CTkFont(family="Consolas", size=16))        
        self.outlier_info_text.pack(fill='x', padx=20, pady=5)
        self.outlier_info_text.insert("1.0", "Data distribution statistics will appear here...")
        self.outlier_info_text.configure(state="disabled")

        bounds_frame = ctk.CTkFrame(self.frame_outliers, fg_color="transparent")
        bounds_frame.pack(fill='x', padx=20, pady=5)
        ctk.CTkLabel(bounds_frame, text="Lower Bound:", font=self.f_text).grid(row=0, column=0, sticky='w', pady=5)
        self.outlier_min_entry = ctk.CTkEntry(bounds_frame, font=self.f_text, width=120)
        self.outlier_min_entry.grid(row=0, column=1, padx=10, pady=5)
        ctk.CTkLabel(bounds_frame, text="Upper Bound:", font=self.f_text).grid(row=1, column=0, sticky='w', pady=5)
        self.outlier_max_entry = ctk.CTkEntry(bounds_frame, font=self.f_text, width=120)
        self.outlier_max_entry.grid(row=1, column=1, padx=10, pady=5)

        self.outlier_action = ctk.CTkComboBox(self.frame_outliers, font=self.f_text, values=["Clip (Cap values at bounds)", "Drop (Delete rows outside bounds)"])
        self.outlier_action.set("Clip (Cap values at bounds)")
        self.outlier_action.pack(fill='x', padx=20, pady=5)
        ctk.CTkButton(self.frame_outliers, text="Apply Thresholds", font=self.f_text, command=self.apply_outliers, fg_color="#3b3b3b", hover_color="#4b4b4b").pack(pady=(10,20), padx=20, fill='x')

        self.frame_cat = self.create_collapsible(self.preprocessing_frame, "4. String / Categorical Variables")
        ctk.CTkLabel(self.frame_cat, text="String Variable:", font=self.f_text).pack(anchor="w", padx=20, pady=(15, 0))
        self.cat_var_combo = ctk.CTkComboBox(self.frame_cat, font=self.f_text, values=[], command=self.on_cat_var_select)
        self.cat_var_combo.pack(fill='x', padx=20, pady=5)
        
        self.cat_info_text = ctk.CTkTextbox(self.frame_cat, height=140, font=ctk.CTkFont(family="Consolas", size=16))
        self.cat_info_text.pack(fill='x', padx=20, pady=5)
        self.cat_info_text.insert("1.0", "Select a variable to view instances...")
        self.cat_info_text.configure(state="disabled")

        action_cat_frame = ctk.CTkFrame(self.frame_cat, fg_color="transparent")
        action_cat_frame.pack(fill='x', padx=20, pady=5)
        ctk.CTkLabel(action_cat_frame, text="Reference Category:", font=self.f_text).grid(row=0, column=0, sticky='w', pady=5)
        self.cat_ref_combo = ctk.CTkComboBox(action_cat_frame, font=self.f_text, values=[])
        self.cat_ref_combo.grid(row=0, column=1, padx=(10,0), pady=5, sticky='ew')
        action_cat_frame.columnconfigure(1, weight=1)

        btn_row = ctk.CTkFrame(self.frame_cat, fg_color="transparent")
        btn_row.pack(fill='x', padx=20, pady=(10, 20))
        ctk.CTkButton(btn_row, text="Drop Column", font=self.f_text, fg_color="#ef4444", hover_color="#dc2626", command=lambda: self.apply_categoricals("drop")).pack(side='left', expand=True, fill='x', padx=(0, 5))
        ctk.CTkButton(btn_row, text="Encode (Dummies)", font=self.f_text, fg_color="#3b3b3b", hover_color="#4b4b4b", command=lambda: self.apply_categoricals("encode")).pack(side='left', expand=True, fill='x', padx=(5, 0))

        self.frame_drop = self.create_collapsible(self.preprocessing_frame, "5. Drop Any Variable(s)")
        drop_inner = ctk.CTkFrame(self.frame_drop, fg_color="transparent")
        drop_inner.pack(fill='x', padx=20, pady=15)
        ctk.CTkLabel(drop_inner, text="Select Variable(s):\n(Hold Ctrl/Shift)", font=self.f_text).pack(side='left', anchor='n', padx=(0, 10))
        listbox_config = {"bg": "#2b2b2b", "fg": "white", "selectbackground": "#ef4444", "selectforeground": "white", "bd": 0, "highlightthickness": 1, "highlightbackground": "#3b3b3b", "highlightcolor": "#ef4444", "font": ("Segoe UI", 15), "height": 5}
        list_frame = ctk.CTkFrame(drop_inner, fg_color="transparent")
        list_frame.pack(side='left', fill='x', expand=True, padx=10)
        self.drop_var_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, exportselection=0, **listbox_config)
        drop_scroll = ctk.CTkScrollbar(list_frame, orientation="vertical", command=self.drop_var_list.yview)
        self.drop_var_list.configure(yscrollcommand=drop_scroll.set)
        drop_scroll.pack(side="right", fill="y")
        self.drop_var_list.pack(side="left", fill="both", expand=True)
        self.bind_hover_scroll(self.drop_var_list) 
        ctk.CTkButton(drop_inner, text="Drop Selected", font=self.f_text, fg_color="#ef4444", hover_color="#dc2626", command=self.apply_drop_any).pack(side='left', anchor='n', padx=20)

        self.frame_scale = self.create_collapsible(self.preprocessing_frame, "6. Scaling / Normalization")
        scale_inner = ctk.CTkFrame(self.frame_scale, fg_color="transparent")
        scale_inner.pack(fill='x', padx=20, pady=15)
        
        ctk.CTkLabel(scale_inner, text="Variables to scale:", font=self.f_text).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scale_var_type = ctk.CTkComboBox(scale_inner, values=["All Numeric Predictors (No Target)", "Target Variable ONLY", "All Numeric Variables"], font=self.f_text, width=250)
        self.scale_var_type.grid(row=0, column=1, padx=15, pady=5)
        
        ctk.CTkLabel(scale_inner, text="Method:", font=self.f_text).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.scale_method = ctk.CTkComboBox(scale_inner, values=["StandardScaler (Z-Score)", "MinMaxScaler (0-1)"], font=self.f_text, width=250)
        self.scale_method.grid(row=1, column=1, padx=15, pady=5)
        
        ctk.CTkButton(scale_inner, text="Apply Scaling", font=self.f_text, fg_color="#f59e0b", hover_color="#d97706", command=self.apply_scaling).grid(row=2, column=0, columnspan=2, pady=15)

        export_frame = ctk.CTkFrame(self.preprocessing_frame, corner_radius=10, fg_color="transparent")
        export_frame.pack(fill='x', padx=10, pady=(20, 10))
        action_btns_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        action_btns_frame.pack(side='right', pady=10)

        ctk.CTkButton(action_btns_frame, text="Visualize Data", command=lambda: self.visualize_dataframe(self.df, "Dataset Visualization Table"), font=ctk.CTkFont(size=16, weight="bold"), fg_color="#1f6aa5", hover_color="#144870").pack(side='top', fill='x', pady=(0, 5))
        ctk.CTkButton(action_btns_frame, text="Save Processed Dataset as CSV", command=self.export_processed_data, font=ctk.CTkFont(size=16, weight="bold")).pack(side='top', fill='x', pady=(5, 0))
        ctk.CTkLabel(export_frame, text="Note: Transformations automatically update Algorithm selection lists.", font=self.f_text, text_color="#1f6aa5").pack(side='right', padx=20)
        ctk.CTkButton(action_btns_frame, text="🧠 Ask AI to Profile Data", command=self.profile_dataset, font=ctk.CTkFont(size=16, weight="bold"), fg_color="#68B946", hover_color="#539438").pack(side='top', fill='x', pady=(15, 0))
        
    def apply_scaling(self):
        if self.df is None: return
        method_str = self.scale_method.get()
        var_type = self.scale_var_type.get()
        target_col = self.target_var.get()
        
        scaler = StandardScaler() if "StandardScaler" in method_str else MinMaxScaler()
        
        try:
            self.save_state(f"Applied {method_str.split()[0]}")
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            
            if var_type == "Target Variable ONLY":
                if target_col in numeric_cols:
                    self.df[[target_col]] = scaler.fit_transform(self.df[[target_col]])
                    msg = f"Target variable '{target_col}' has been scaled."
                else:
                    raise ValueError("Target is not numeric or not selected.")
            elif var_type == "All Numeric Predictors (No Target)":
                cols_to_scale = [c for c in numeric_cols if c != target_col]
                if cols_to_scale:
                    self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
                    msg = f"Scaled {len(cols_to_scale)} numeric predictors."
            else:
                self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
                msg = f"Scaled all {len(numeric_cols)} numeric variables."
            
            self.log_event(msg)
            message_box_msg = msg
            if "Target" in var_type or "All Numeric Variables" in var_type:
                message_box_msg += "\n\n⚠️ WARNING: By scaling the Target Variable around zero, your MAPE metric might explode during training! Rely on RMSE instead."
            
            messagebox.showinfo("Scaling Successful", message_box_msg)
            self.refresh_workspace()
            
        except Exception as e:
            self.log_event(f"Error during scaling: {str(e)}")
            messagebox.showerror("Scaling Error", f"Could not apply scaling:\n{str(e)}")

    def visualize_dataframe(self, dataframe, title):
        if dataframe is None:
            messagebox.showwarning("Data Required", "Please load a dataset first.")
            return

        # 1. Main visualization window creation
        top = ctk.CTkToplevel(self.root)
        top.title(title)
        top.geometry("1100x650")
        top.minsize(700, 450)
        
        # SECURITY: Force window to open in foreground
        top.lift()
        top.attributes('-topmost', True)
        top.after(200, lambda: top.attributes('-topmost', False))
        top.focus_force()

        # 2. Tab system creation (Tabview)
        notebook = ctk.CTkTabview(top, corner_radius=10)
        notebook.pack(expand=True, fill="both", padx=20, pady=20)
        
        tab_stats = notebook.add("Statistical description")
        tab_data = notebook.add("Raw Data (All Data)")

        # ----------------------------------------------------
        # TAB 1: DESCRIPTIVE STATISTICS
        # ----------------------------------------------------
        numeric_df = dataframe.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # Calculate indicators via Pandas
            stats_df = numeric_df.describe().T[['mean', 'max', 'min', 'std']].round(4)
            stats_df.reset_index(inplace=True)
            stats_df.rename(columns={'index': 'Variable', 'mean': 'Average', 'max': 'Max', 'min': 'Min', 'std': 'Std Dev'}, inplace=True)
            
            frame_stats = ctk.CTkFrame(tab_stats, fg_color="transparent")
            frame_stats.pack(expand=True, fill="both", padx=10, pady=10)
            
            scroll_y_stats = ctk.CTkScrollbar(frame_stats, orientation="vertical")
            scroll_y_stats.pack(side="right", fill="y")
            scroll_x_stats = ctk.CTkScrollbar(frame_stats, orientation="horizontal")
            scroll_x_stats.pack(side="bottom", fill="x")
            
            cols_stats = list(stats_df.columns)
            tree_stats = ttk.Treeview(frame_stats, columns=cols_stats, show='headings', yscrollcommand=scroll_y_stats.set, xscrollcommand=scroll_x_stats.set)
            scroll_y_stats.configure(command=tree_stats.yview)
            scroll_x_stats.configure(command=tree_stats.xview)
            
            for col in cols_stats:
                tree_stats.heading(col, text=col)
                width = 250 if col == 'Variable' else 150
                tree_stats.column(col, width=width, anchor='center')
                
            for idx, row in stats_df.iterrows():
                tree_stats.insert("", tk.END, values=list(row))
                
            tree_stats.pack(side="left", expand=True, fill="both")
            self.bind_hover_scroll(tree_stats)
        else:
            lbl = ctk.CTkLabel(tab_stats, text="No numeric variables detected to calculate statistics.", font=self.f_text)
            lbl.pack(expand=True)

        # ----------------------------------------------------
        # TAB 2: RAW DATA (ALL DATA)
        # ----------------------------------------------------
        frame_data = ctk.CTkFrame(tab_data, fg_color="transparent")
        frame_data.pack(expand=True, fill="both", padx=10, pady=10)
        
        scroll_y_data = ctk.CTkScrollbar(frame_data, orientation="vertical")
        scroll_y_data.pack(side="right", fill="y")
        scroll_x_data = ctk.CTkScrollbar(frame_data, orientation="horizontal")
        scroll_x_data.pack(side="bottom", fill="x")
        
        columns_data = list(dataframe.columns)
        tree_data = ttk.Treeview(frame_data, columns=columns_data, show='headings', yscrollcommand=scroll_y_data.set, xscrollcommand=scroll_x_data.set)
        scroll_y_data.configure(command=tree_data.yview)
        scroll_x_data.configure(command=tree_data.xview)
        
        for col in columns_data:
            tree_data.heading(col, text=col)
            tree_data.column(col, width=150, anchor='center')
            
        for idx, row in dataframe.iterrows():
            tree_data.insert("", tk.END, values=list(row))
            
        tree_data.pack(side="left", expand=True, fill="both")
        self.bind_hover_scroll(tree_data)


    def show_statistics_table(self):
        if self.df is None:
            messagebox.showwarning("Data Required", "Please load a dataset first.")
            return

        # Filter only numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showinfo("Info", "No numeric variables to analyze.")
            return

        # Calculate stats and format DataFrame
        stats_df = numeric_df.describe().T[['mean', 'max', 'min', 'std']].round(4)
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'Variable', 'mean': 'Average', 'max': 'Max', 'min': 'Min', 'std': 'Std Dev'}, inplace=True)

        # Create Window
        top = ctk.CTkToplevel(self.root)
        top.title("Descriptive Statistics")
        top.geometry("900x500")
        top.lift()
        top.attributes('-topmost', True)
        top.after(200, lambda: top.attributes('-topmost', False))
        top.focus_force()
        
        # FIX: Force window to foreground
        top.lift()
        top.attributes('-topmost', True)
        top.after(200, lambda: top.attributes('-topmost', False))
        top.focus_force()

        table_frame = ctk.CTkFrame(top, corner_radius=10)
        table_frame.pack(expand=True, fill="both", padx=20, pady=20)

        columns = list(stats_df.columns)
        scroll_y = ctk.CTkScrollbar(table_frame, orientation="vertical")
        scroll_y.pack(side="right", fill="y")

        tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scroll_y.set)
        scroll_y.configure(command=tree.yview)

        # Column configuration
        for col in columns:
            tree.heading(col, text=col)
            width = 250 if col == 'Variable' else 150
            tree.column(col, width=width, anchor='center')

        # Insert Data
        for idx, row in stats_df.iterrows():
            tree.insert("", tk.END, values=list(row))

        tree.pack(side="left", expand=True, fill="both")
        self.bind_hover_scroll(tree)

    def update_split_from_slider(self, value):
        self.split_str_var.set(f"{float(value):.1f}")
    def update_split_from_entry(self, *args):
        try:
            val = float(self.split_str_var.get())
            if 50.0 <= val <= 100.0:
                self.split_var.set(val)
        except ValueError: pass 
    def on_strategy_change(self, selected_strategy):
        self.train_test_frame.grid_remove()
        self.kfold_frame.grid_remove()
        if selected_strategy == "Train/Test Split":
            self.train_test_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        elif selected_strategy == "K-Fold Cross Validation":
            self.kfold_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xlsx *.json"), ("All files", "*.*")])
        self.root.focus_force()
        self.root.update_idletasks()
        if not filepath: return
        try:
            sep_choice = self.csv_sep.get()
            if filepath.endswith('.csv'):
                if sep_choice == "Auto":
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        first_lines = f.readline() + "\n" + f.readline()
                    possible_seps = [',', ';', '\t', '|']
                    sep = ',' 
                    max_count = 0
                    for s in possible_seps:
                        count = first_lines.count(s)
                        if count > max_count:
                            max_count = count; sep = s
                    self.csv_sep.set(sep)
                else: sep = sep_choice
                try: self.df = pd.read_csv(filepath, sep=sep, encoding='utf-8')
                except UnicodeDecodeError: self.df = pd.read_csv(filepath, sep=sep, encoding='latin1')
            elif filepath.endswith('.xlsx'): self.df = pd.read_excel(filepath)
            elif filepath.endswith('.json'): self.df = pd.read_json(filepath)
            self.lbl_file.configure(text=f"Loaded: {filepath.split('/')[-1]}  |  Rows: {len(self.df):,}", text_color="#1f6aa5")
            self.df_history.clear(); self.df_future.clear(); self.update_undo_redo_buttons()
            self.log_event(f"Loaded dataset '{filepath.split('/')[-1]}' successfully. Dimensions: {len(self.df)}x{len(self.df.columns)}.")
            if not self.preprocessing_frame.winfo_ismapped(): self.preprocessing_frame.pack(fill='both', expand=True, pady=(0, 10))
            messagebox.showinfo("Success", "Data loaded!")
            self.root.focus_force()
            self.refresh_workspace()
        except Exception as e:
            self.log_event(f"Error loading file: {str(e)}")
            messagebox.showerror("Parsing Error", f"Could not read the file:\n{str(e)}")
            self.root.focus_force()

    def update_zero_handling_state(self, *args):
        """Vérifie si le dataset contient des zéros pour rendre l'UI transparente si ce n'est pas le cas. 
           Gère aussi l'activation du paramètre Epsilon."""
        try:
            if hasattr(self, 'eco_handle_zeros') and self.eco_handle_zeros.winfo_exists():
                has_zeros = False
                if self.df is not None:
                    numeric_cols = self.df.select_dtypes(include=np.number).columns
                    has_zeros = (self.df[numeric_cols] == 0).any().any()
                
                if has_zeros:
                    self.eco_handle_zeros.configure(state="readonly")
                    
                    # On active Epsilon SEULEMENT si l'option est "translate"
                    if self.eco_handle_zeros.get() == "translate":
                        self.eco_epsilon.configure(state="normal")
                    else:
                        self.eco_epsilon.configure(state="disabled")
                else:
                    self.eco_handle_zeros.configure(state="disabled")
                    self.eco_epsilon.configure(state="disabled")
        except Exception:
            pass

    def refresh_workspace(self):
        if self.df is None:
            if hasattr(self, 'target_combo'): self.target_combo.configure(values=[])
            if hasattr(self, 'outlier_var_combo'): self.outlier_var_combo.configure(values=[])
            if hasattr(self, 'cat_var_combo'): self.cat_var_combo.configure(values=[])
            return
        columns = [str(col) for col in self.df.columns]
        if self.target_var.get() not in columns and columns: self.target_var.set(columns[0])
        try:
            if hasattr(self, 'list_cont') and self.list_cont.winfo_exists():
                self.list_cont.delete(0, tk.END)
                for col in columns: self.list_cont.insert(tk.END, col)
            if hasattr(self, 'list_dummy') and self.list_dummy.winfo_exists():
                self.list_dummy.delete(0, tk.END)
                for col in columns: self.list_dummy.insert(tk.END, col)
            if hasattr(self, 'list_features') and self.list_features.winfo_exists():
                self.list_features.delete(0, tk.END)
                for col in columns: self.list_features.insert(tk.END, col)
            if hasattr(self, 'target_combo') and self.target_combo.winfo_exists():
                self.target_combo.configure(values=columns)
                self.target_combo.set(self.target_var.get())
        except Exception: pass
        if hasattr(self, 'drop_var_list') and self.drop_var_list.winfo_exists():
            self.drop_var_list.delete(0, tk.END)
            for col in columns: self.drop_var_list.insert(tk.END, col)

        numeric_cols_raw = self.df.select_dtypes(include=np.number).columns.tolist()
        continuous_cols = [str(col) for col in numeric_cols_raw if self.df[col].nunique() > 2]
        cat_cols = [str(col) for col in self.df.select_dtypes(include=['object', 'category']).columns.tolist()]

        if continuous_cols:
            self.outlier_var_combo.configure(values=continuous_cols)
            self.outlier_var_combo.set(continuous_cols[0])
            self.on_outlier_var_select(continuous_cols[0])
        else:
            self.outlier_var_combo.configure(values=["No continuous vars"]); self.outlier_var_combo.set("No continuous vars")
            self.outlier_info_text.configure(state="normal"); self.outlier_info_text.delete("1.0", tk.END); self.outlier_info_text.insert("1.0", "No continuous variable detected."); self.outlier_info_text.configure(state="disabled")

        if cat_cols:
            self.cat_var_combo.configure(values=cat_cols)
            self.cat_var_combo.set(cat_cols[0])
            self.on_cat_var_select(cat_cols[0])
        else:
            self.cat_var_combo.configure(values=["No string vars"]); self.cat_var_combo.set("No string vars")
            self.cat_info_text.configure(state="normal"); self.cat_info_text.delete("1.0", tk.END); self.cat_info_text.insert("1.0", "All variables are currently numeric."); self.cat_info_text.configure(state="disabled")
            self.cat_ref_combo.configure(values=["N/A"]); self.cat_ref_combo.set("N/A")
            
        self.update_zero_handling_state()

    def on_outlier_var_select(self, choice):
        if self.df is None or choice not in self.df.columns: return
        stats_desc = self.df[choice].describe()
        info_str = f"  Mean       : {stats_desc.get('mean', 0):<12.2f} |   Min : {stats_desc.get('min', 0):.2f}\n  Median     : {stats_desc.get('50%', 0):<12.2f} |   Q1  : {stats_desc.get('25%', 0):.2f}\n  Std Dev    : {stats_desc.get('std', 0):<12.2f} |   Q3  : {stats_desc.get('75%', 0):.2f}\n  Count      : {int(stats_desc.get('count', 0)):<12} |   Max : {stats_desc.get('max', 0):.2f}"        
        self.outlier_info_text.configure(state="normal"); self.outlier_info_text.delete("1.0", tk.END); self.outlier_info_text.insert("1.0", info_str); self.outlier_info_text.configure(state="disabled")

    def apply_outliers(self):
        if self.df is None: return
        col = self.outlier_var_combo.get()
        if col not in self.df.columns: return
        try:
            self.save_state(f"Manage outliers on '{col}'")
            min_val, max_val = float(self.outlier_min_entry.get()), float(self.outlier_max_entry.get())
            if "Clip" in self.outlier_action.get():
                self.df[col] = self.df[col].clip(lower=min_val, upper=max_val)
                msg = f"Values in '{col}' capped between {min_val} and {max_val}."
            else:
                initial_len = len(self.df)
                self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
                msg = f"Dropped {initial_len - len(self.df)} rows outside bounds [{min_val}, {max_val}]."
            messagebox.showinfo("Success", msg)
            self.refresh_workspace()
        except Exception as e: messagebox.showerror("Error", str(e))

    def on_cat_var_select(self, choice):
        if self.df is None or choice not in self.df.columns: return
        counts = self.df[choice].value_counts()
        info_str = f"Total Unique Instances: {len(counts)}\n\nValue Frequencies:\n"
        for val, cnt in counts.items(): info_str += f" - {val} : {cnt}\n"
        self.cat_info_text.configure(state="normal"); self.cat_info_text.delete("1.0", tk.END); self.cat_info_text.insert("1.0", info_str); self.cat_info_text.configure(state="disabled")
        str_indexes = [str(x) for x in counts.index]
        self.cat_ref_combo.configure(values=str_indexes)
        if str_indexes: self.cat_ref_combo.set(str_indexes[0])

    def apply_categoricals(self, action):
        if self.df is None: return
        col = self.cat_var_combo.get()
        if col not in self.df.columns: return
        try:
            if action == "drop":
                self.save_state(f"Drop categorical '{col}'")
                self.df = self.df.drop(columns=[col])
                msg = f"Column '{col}' successfully dropped."
            elif action == "encode":
                self.save_state(f"Encode categorical '{col}'")
                ref = self.cat_ref_combo.get()
                dummies = pd.get_dummies(self.df[col], prefix=col).astype(int)
                if f"{col}_{ref}" in dummies.columns: dummies = dummies.drop(columns=[f"{col}_{ref}"])
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                msg = f"Column '{col}' encoded into {len(dummies.columns)} dummy variables.\nReference '{ref}' was dropped."
            messagebox.showinfo("Success", msg); self.refresh_workspace()
        except Exception as e: messagebox.showerror("Error", str(e))

    def apply_drop_any(self):
        if self.df is None: return
        selected_indices = self.drop_var_list.curselection()
        if not selected_indices: return messagebox.showwarning("No Selection", "Please select at least one column to drop.")
        valid_cols = [c for c in [self.drop_var_list.get(i) for i in selected_indices] if c in self.df.columns]
        if not valid_cols: return
        try:
            self.save_state(f"Drop columns: {', '.join(valid_cols)}")
            self.df = self.df.drop(columns=valid_cols)
            messagebox.showinfo("Success", f"Successfully dropped {len(valid_cols)} column(s)."); self.refresh_workspace()
        except Exception as e: messagebox.showerror("Error", str(e))

    def export_processed_data(self):
        if self.df is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="processed_data.csv", filetypes=[("CSV", "*.csv")])
        if filepath:
            try: self.df.to_csv(filepath, index=False); messagebox.showinfo("Export Saved", f"Processed dataset saved to:\n{filepath}")
            except Exception as e: messagebox.showerror("Export Error", str(e))

    # ==========================================
    # TAB 2: ALGORITHMS & HYPERPARAMETERS
    # ==========================================
    def build_algo_tab(self):
        try:
            self.icon_eco = ctk.CTkImage(Image.open(get_resource_path("images/ecoretina.png")), size=(70, 70))
            self.icon_linear = ctk.CTkImage(Image.open(get_resource_path("images/linear-regression.png")), size=(70, 70))
            self.icon_tree = ctk.CTkImage(Image.open(get_resource_path("images/tree.png")), size=(70, 70))
            self.icon_nn = ctk.CTkImage(Image.open(get_resource_path("images/deep-learning.png")), size=(70, 70))
            self.icon_xgboost = ctk.CTkImage(Image.open(get_resource_path("images/xgboost.png")), size=(70, 70)) 
            self.icon_lasso = ctk.CTkImage(Image.open(get_resource_path("images/lasso.png")), size=(70, 70))
            self.icon_ridge = ctk.CTkImage(Image.open(get_resource_path("images/ridge.png")), size=(70, 70))
            self.icon_elasticnet = ctk.CTkImage(Image.open(get_resource_path("images/elasticnet.png")), size=(70, 70))
        except Exception:
            self.icon_eco = self.icon_linear = self.icon_tree = self.icon_nn = self.icon_xgboost = self.icon_lasso = self.icon_ridge = self.icon_elasticnet = None

        green_color = (104, 185, 70, 255)
        self.algo_icons_green = {
            "EcoRETINA": create_tinted_icon("images/ecoretina.png", (60, 60), green_color), "OLS": create_tinted_icon("images/linear-regression.png", (60, 60), green_color),
            "Lasso": create_tinted_icon("images/lasso.png", (60, 60), green_color), "Ridge": create_tinted_icon("images/ridge.png", (60, 60), green_color),
            "ElasticNet": create_tinted_icon("images/elasticnet.png", (60, 60), green_color), "XGBoost": create_tinted_icon("images/xgboost.png", (60, 60), green_color),
            "Random Forest": create_tinted_icon("images/tree.png", (60, 60), green_color), "Neural Network": create_tinted_icon("images/deep-learning.png", (60, 60), green_color)
        }

        self.action_frame = ctk.CTkFrame(self.tab_algo, fg_color="transparent")
        self.action_frame.pack(fill='x', side='bottom', padx=20, pady=15)
        self.grid_container = ctk.CTkFrame(self.tab_algo, fg_color="transparent")
        self.grid_container.pack(expand=True, fill="both", side='top', padx=40, pady=40)
        self.grid_container.columnconfigure((0, 1, 2, 3), weight=1, uniform="equal")
        self.grid_container.rowconfigure((0, 1), weight=1, uniform="equal")

        algos = [("EcoRETINA", self.icon_eco, 0, 0), ("OLS", self.icon_linear, 0, 1), ("Lasso", self.icon_lasso, 0, 2), ("Ridge", self.icon_ridge, 0, 3), ("ElasticNet", self.icon_elasticnet, 1, 0), ("XGBoost", self.icon_xgboost, 1, 1), ("Random Forest", self.icon_tree, 1, 2), ("Neural Network", self.icon_nn, 1, 3)]
        for name, icon, row, col in algos:
            card = ctk.CTkButton(self.grid_container, text=f"\n{name}", image=icon, compound="top", anchor="center", command=lambda n=name: self.show_hyperparams_page(n), font=ctk.CTkFont(size=20, weight="bold"), fg_color="#164A31", hover_color="#68B946", text_color="#FFFFFF", corner_radius=15)
            card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

        left_status_frame = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        left_status_frame.pack(side='left', fill='y', pady=5)
        self.progress = ctk.CTkProgressBar(left_status_frame, mode='indeterminate', width=200); self.progress.set(0)
        self.lbl_status = ctk.CTkLabel(left_status_frame, text="Ready.", font=self.f_text, text_color="gray")
        self.lbl_status.pack(side='left', padx=(0, 20))


    def show_hyperparams_page(self, algo_name):
        self.grid_container.pack_forget()
        self.active_algo = algo_name 
        if hasattr(self, "algo_detail_frame"): self.algo_detail_frame.destroy()

        self.algo_detail_frame = ctk.CTkFrame(self.tab_algo, fg_color="transparent")
        self.algo_detail_frame.pack(expand=True, fill="both", side='top', padx=10, pady=(10, 0), before=self.action_frame)

        fixed_header = ctk.CTkFrame(self.algo_detail_frame, fg_color="transparent")
        fixed_header.pack(fill="x", pady=(0, 10))
        fixed_header.columnconfigure(0, weight=1, uniform="col"); fixed_header.columnconfigure(1, weight=1, uniform="col"); fixed_header.columnconfigure(2, weight=1, uniform="col")

        ctk.CTkButton(fixed_header, text="◀  Back to Algorithms", command=self.hide_hyperparams_page, font=self.f_text, fg_color="#000000", text_color="white", hover_color="#333333", width=200).grid(row=0, column=0, sticky="nw", padx=10)
        title_frame = ctk.CTkFrame(fixed_header, fg_color="transparent")
        title_frame.grid(row=0, column=1, sticky="n")
        ctk.CTkLabel(title_frame, text=f"{algo_name}", font=ctk.CTkFont(size=40, weight="bold"), text_color="#68B946").pack()

        icon = getattr(self, "algo_icons_green", {}).get(algo_name)
        if icon: ctk.CTkLabel(title_frame, text="", image=icon).pack(pady=(5, 0))
        # --- NOUVEAU : BOUTONS EN HAUT À DROITE ---
        right_header_frame = ctk.CTkFrame(fixed_header, fg_color="transparent")
        # On le place dans la colonne 2 (à droite) avec un ancrage au nord-est ("ne")
        right_header_frame.grid(row=0, column=2, sticky="ne", padx=10)
        
        top_btn_row = ctk.CTkFrame(right_header_frame, fg_color="transparent")
        top_btn_row.pack(side='top', fill='x', pady=(0, 5))

        self.btn_run = ctk.CTkButton(top_btn_row, text="► Run model", font=ctk.CTkFont(weight="bold", size=18), height=40, fg_color="#68B946", hover_color="#539438", text_color="#FFFFFF", command=self.start_training)
        self.btn_run.pack(side='left', expand=True, fill='x', padx=(0, 10))

        # Icône du bouton stop
        icon_size = 14; stop_pil = Image.new("RGBA", (icon_size, icon_size), (255, 255, 255, 0)); draw = ImageDraw.Draw(stop_pil); draw.rectangle((0, 0, icon_size-1, icon_size-1), outline="#FFFFFF", width=2); stop_img = ctk.CTkImage(stop_pil, size=(icon_size, icon_size))
        
        self.btn_stop = ctk.CTkButton(top_btn_row, text="", image=stop_img, width=40, height=40, corner_radius=8, fg_color="#ef4444", hover_color="#dc2626", command=self.stop_training, state="disabled")
        self.btn_stop.pack(side='right')

        self.btn_summary = ctk.CTkButton(right_header_frame, text="View Report", font=self.f_text, command=self.view_summary, fg_color="#1f6aa5", hover_color="#144870", height=40)
        self.btn_summary.pack(side='top', fill='x')
        # ----------------------------------------

        self.params_scroll = ctk.CTkScrollableFrame(self.algo_detail_frame, fg_color="transparent")
        self.params_scroll.pack(expand=True, fill="both")

        target_frame = ctk.CTkFrame(self.params_scroll, fg_color="transparent")
        target_frame.pack(fill="x", pady=(5, 10), padx=10)
        ctk.CTkLabel(target_frame, text="Target Variable (Y):", font=self.f_subtitle).pack(side="left", padx=(10, 10))
        cols = [str(col) for col in self.df.columns] if self.df is not None else ["No Data"]
        self.target_combo = ctk.CTkComboBox(target_frame, values=cols, variable=self.target_var, font=self.f_text, state="readonly", width=350)
        if self.target_var.get() == "" and cols and cols[0] != "No Data": self.target_var.set(cols[0])
        self.target_combo.set(self.target_var.get() if self.target_var.get() else cols[0]); self.target_combo.pack(side="left")

        # Directly show hyperparameters frame
        self.algo_params_frame = ctk.CTkFrame(self.params_scroll, fg_color="#2b2b2b", corner_radius=10)
        self.algo_params_frame.pack(fill="x", padx=10, pady=(10, 5))

        # Directly show variables frame
        self.algo_vars_frame = ctk.CTkFrame(self.params_scroll, fg_color="#2b2b2b", corner_radius=10)
        self.algo_vars_frame.pack(fill="both", expand=True, padx=10, pady=5)

        if algo_name == "EcoRETINA": self.build_ecoretina_params(self.algo_params_frame)
        elif algo_name == "OLS": self.build_ols_params(self.algo_params_frame)
        elif algo_name == "Lasso": self.build_lasso_params(self.algo_params_frame)
        elif algo_name == "Ridge": self.build_ridge_params(self.algo_params_frame)
        elif algo_name == "ElasticNet": self.build_elasticnet_params(self.algo_params_frame)
        elif algo_name == "XGBoost": self.build_xgboost_params(self.algo_params_frame)
        elif algo_name == "Random Forest": self.build_rf_params(self.algo_params_frame)
        elif algo_name == "Neural Network": self.build_nn_params(self.algo_params_frame)
            
        self.build_feature_selection(self.algo_vars_frame, is_eco=(algo_name == "EcoRETINA"))
        self.root.update_idletasks()

    def hide_hyperparams_page(self):
        if hasattr(self, "algo_detail_frame"): self.algo_detail_frame.pack_forget()
        self.grid_container.pack(expand=True, fill="both", side='top', padx=40, pady=40, before=self.action_frame)

    def build_feature_selection(self, parent, is_eco):
        var_frame = ctk.CTkFrame(parent, fg_color="transparent")
        var_frame.pack(fill='both', expand=True, padx=10, pady=10)
        header_area = ctk.CTkFrame(var_frame, fg_color="transparent")
        header_area.pack(fill='x', padx=15, pady=(5, 5))
        cols = list(self.df.columns) if self.df is not None else []
        lb_height = min(20, max(12, len(cols)))
        lists_container = ctk.CTkFrame(var_frame, fg_color="transparent")
        lists_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        if is_eco:
            self.select_all_cont_var = ctk.IntVar(value=0); self.select_all_dum_var = ctk.IntVar(value=0)
            ctk.CTkCheckBox(header_area, text="Select All Continuous", font=self.f_text, variable=self.select_all_cont_var, command=self.toggle_cont).pack(side="left", padx=10)
            ctk.CTkCheckBox(header_area, text="Select All Dummies", font=self.f_text, variable=self.select_all_dum_var, command=self.toggle_dum).pack(side="right", padx=10)
            lists_container.columnconfigure(0, weight=1); lists_container.columnconfigure(1, weight=1)

            cont_col = ctk.CTkFrame(lists_container, fg_color="transparent"); cont_col.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
            ctk.CTkLabel(cont_col, text="Continuous Variables", font=self.f_subtitle).pack(anchor='w', pady=(0, 5))
            self.list_cont = tk.Listbox(cont_col, selectmode=tk.EXTENDED, exportselection=0, font=("Segoe UI", 15), height=lb_height, bg="#2b2b2b", fg="white", selectbackground="#1f6aa5")
            cont_scroll = ctk.CTkScrollbar(cont_col, orientation="vertical", command=self.list_cont.yview); self.list_cont.configure(yscrollcommand=cont_scroll.set); cont_scroll.pack(side="right", fill="y"); self.list_cont.pack(side="left", fill="both", expand=True)
            self.bind_hover_scroll(self.list_cont); self.list_cont.bind('<<ListboxSelect>>', self.on_cont_select)

            dummy_col = ctk.CTkFrame(lists_container, fg_color="transparent"); dummy_col.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
            ctk.CTkLabel(dummy_col, text="Dummy Variables", font=self.f_subtitle).pack(anchor='w', pady=(0, 5))
            self.list_dummy = tk.Listbox(dummy_col, selectmode=tk.EXTENDED, exportselection=0, font=("Segoe UI", 15), height=lb_height, bg="#2b2b2b", fg="white", selectbackground="#1f6aa5")
            dummy_scroll = ctk.CTkScrollbar(dummy_col, orientation="vertical", command=self.list_dummy.yview); self.list_dummy.configure(yscrollcommand=dummy_scroll.set); dummy_scroll.pack(side="right", fill="y"); self.list_dummy.pack(side="left", fill="both", expand=True)
            self.bind_hover_scroll(self.list_dummy); self.list_dummy.bind('<<ListboxSelect>>', self.on_dummy_select)
            for col in cols: self.list_cont.insert(tk.END, col); self.list_dummy.insert(tk.END, col)
        else:
            self.select_all_feat_var = ctk.IntVar(value=0)
            ctk.CTkCheckBox(header_area, text="Select All Predictors", font=self.f_text, variable=self.select_all_feat_var, command=self.toggle_features).pack(side="right", padx=10)
            lists_container.columnconfigure(0, weight=1)
            feat_col = ctk.CTkFrame(lists_container, fg_color="transparent"); feat_col.grid(row=0, column=0, sticky='nsew')
            ctk.CTkLabel(feat_col, text="Predictor Variables (Features)", font=self.f_subtitle).pack(anchor='w', pady=(0, 5))
            self.list_features = tk.Listbox(feat_col, selectmode=tk.EXTENDED, exportselection=0, font=("Segoe UI", 15), height=lb_height, bg="#2b2b2b", fg="white", selectbackground="#1f6aa5")
            feat_scroll = ctk.CTkScrollbar(feat_col, orientation="vertical", command=self.list_features.yview); self.list_features.configure(yscrollcommand=feat_scroll.set); feat_scroll.pack(side="right", fill="y"); self.list_features.pack(side="left", fill="both", expand=True)
            self.bind_hover_scroll(self.list_features)
            for col in cols: self.list_features.insert(tk.END, col)

    def on_cont_select(self, event):
        if not hasattr(self, 'list_dummy'): return
        conflits = set(self.list_cont.curselection()).intersection(set(self.list_dummy.curselection()))
        for i in conflits: self.list_dummy.select_clear(i)

    def on_dummy_select(self, event):
        if not hasattr(self, 'list_cont'): return
        conflits = set(self.list_dummy.curselection()).intersection(set(self.list_cont.curselection()))
        for i in conflits: self.list_cont.select_clear(i)

    def toggle_cont(self):
        if not hasattr(self, 'list_cont') or not hasattr(self, 'list_dummy'): return
        if self.select_all_cont_var.get() == 1:
            for i in range(self.list_cont.size()):
                if self.list_cont.get(i) != self.target_var.get() and i not in set(self.list_dummy.curselection()): self.list_cont.select_set(i)
        else: self.list_cont.select_clear(0, tk.END)

    def toggle_dum(self):
        if not hasattr(self, 'list_dummy') or not hasattr(self, 'list_cont'): return
        if self.select_all_dum_var.get() == 1:
            for i in range(self.list_dummy.size()):
                if self.list_dummy.get(i) != self.target_var.get() and i not in set(self.list_cont.curselection()): self.list_dummy.select_set(i)
        else: self.list_dummy.select_clear(0, tk.END)
            
    def toggle_features(self):
        if not hasattr(self, 'list_features'): return
        if self.select_all_feat_var.get() == 1:
            for i in range(self.list_features.size()):
                if self.list_features.get(i) != self.target_var.get(): self.list_features.select_set(i)
        else: self.list_features.select_clear(0, tk.END)

    def build_ecoretina_params(self, parent):
        hyper_frame = ctk.CTkFrame(parent, fg_color="transparent"); hyper_frame.pack(fill='both', expand=True, padx=10, pady=5)
        for i in range(8): hyper_frame.columnconfigure(i, weight=1)
        ctk.CTkLabel(hyper_frame, text="Loss:", font=self.f_text).grid(row=0, column=0, padx=5, pady=8, sticky='e'); self.eco_loss = ctk.CTkComboBox(hyper_frame, font=self.f_text, values=['mse', 'mae', 'MAPE', 'AIC', 'BIC'], state="readonly", width=120); self.eco_loss.set('mse'); self.eco_loss.grid(row=0, column=1, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Reg Type:", font=self.f_text).grid(row=0, column=2, padx=5, pady=8, sticky='e'); self.eco_reg_type = ctk.CTkComboBox(hyper_frame, font=self.f_text, values=['linear', 'logit', 'probit' ], state="readonly", width=120); self.eco_reg_type.set('linear'); self.eco_reg_type.grid(row=0, column=3, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Cross Dummy:", font=self.f_text).grid(row=0, column=4, padx=5, pady=8, sticky='e'); self.eco_cross_dummy_cb = ctk.CTkComboBox(hyper_frame, font=self.f_text, values=['False', 'True'], state="readonly", width=120); self.eco_cross_dummy_cb.set('False'); self.eco_cross_dummy_cb.grid(row=0, column=5, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Cov Type:", font=self.f_text).grid(row=0, column=6, padx=5, pady=8, sticky='e'); self.eco_cov_type = ctk.CTkComboBox(hyper_frame, font=self.f_text, values=['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'], state="readonly", width=120); self.eco_cov_type.set('nonrobust'); self.eco_cov_type.grid(row=0, column=7, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Params (list):", font=self.f_text).grid(row=1, column=0, padx=5, pady=8, sticky='e'); self.eco_params = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_params.insert(0, "[-1.0, 0.0, 1.0]"); self.eco_params.grid(row=1, column=1, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Max R²:", font=self.f_text).grid(row=1, column=2, padx=5, pady=8, sticky='e'); self.eco_max_r2 = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_max_r2.insert(0, "0.99"); self.eco_max_r2.grid(row=1, column=3, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Grid:", font=self.f_text).grid(row=1, column=4, padx=5, pady=8, sticky='e'); self.eco_grid = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_grid.insert(0, "0.005"); self.eco_grid.grid(row=1, column=5, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Seed:", font=self.f_text).grid(row=1, column=6, padx=5, pady=8, sticky='e'); self.eco_seed = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_seed.insert(0, "8"); self.eco_seed.grid(row=1, column=7, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Max Inst.:", font=self.f_text).grid(row=2, column=0, padx=5, pady=8, sticky='e'); self.eco_max_instances = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_max_instances.insert(0, "100000"); self.eco_max_instances.grid(row=2, column=1, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Max Reg:", font=self.f_text).grid(row=2, column=2, padx=5, pady=8, sticky='e'); self.eco_max_reg = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_max_reg.insert(0, "100"); self.eco_max_reg.grid(row=2, column=3, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Chunk Size:", font=self.f_text).grid(row=2, column=4, padx=5, pady=8, sticky='e'); self.eco_chunk_size = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_chunk_size.insert(0, "500"); self.eco_chunk_size.grid(row=2, column=5, padx=5, pady=8, sticky='ew')
        ctk.CTkLabel(hyper_frame, text="Model Step:", font=self.f_text).grid(row=2, column=6, padx=5, pady=8, sticky='e'); self.eco_model_step = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120); self.eco_model_step.insert(0, "1"); self.eco_model_step.grid(row=2, column=7, padx=5, pady=8, sticky='ew')

        # --- NOUVEAU: GESTION DES ZÉROS ---
        ctk.CTkLabel(hyper_frame, text="Handle Zeros:", font=self.f_text).grid(row=3, column=0, padx=5, pady=8, sticky='e')
        self.eco_handle_zeros = ctk.CTkComboBox(
            hyper_frame, 
            font=self.f_text, 
            values=['prevent_division', 'translate', 'drop_rows'], 
            state="readonly", 
            width=120,
            command=self.update_zero_handling_state
        )
        self.eco_handle_zeros.set('prevent_division')
        self.eco_handle_zeros.grid(row=3, column=1, padx=5, pady=8, sticky='ew')
        
        ctk.CTkLabel(hyper_frame, text="Epsilon (Shift):", font=self.f_text).grid(row=3, column=2, padx=5, pady=8, sticky='e')
        self.eco_epsilon = ctk.CTkEntry(hyper_frame, font=self.f_text, width=120)
        self.eco_epsilon.insert(0, "auto")
        self.eco_epsilon.grid(row=3, column=3, padx=5, pady=8, sticky='ew')
        
        # Mettre à jour l'opacité selon le dataset actuel
        self.update_zero_handling_state()

    def build_ols_params(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill='both', expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="Fit Intercept (Add Constant):", font=self.f_text).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.ols_fit_intercept = ctk.CTkComboBox(f, font=self.f_text, values=['True', 'False'], width=150); self.ols_fit_intercept.set('True'); self.ols_fit_intercept.grid(row=0, column=1, padx=10, pady=10)

    def build_xgboost_params(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill='both', expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="N Estimators:", font=self.f_text).grid(row=0, column=0, padx=10, pady=10, sticky='w'); self.xgb_n_estimators = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_n_estimators.insert(0, "100"); self.xgb_n_estimators.grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Max Depth:", font=self.f_text).grid(row=0, column=2, padx=10, pady=10, sticky='w'); self.xgb_max_depth = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_max_depth.insert(0, "6"); self.xgb_max_depth.grid(row=0, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Learning Rate:", font=self.f_text).grid(row=1, column=0, padx=10, pady=10, sticky='w'); self.xgb_lr = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_lr.insert(0, "0.1"); self.xgb_lr.grid(row=1, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Subsample:", font=self.f_text).grid(row=1, column=2, padx=10, pady=10, sticky='w'); self.xgb_subsample = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_subsample.insert(0, "1.0"); self.xgb_subsample.grid(row=1, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Colsample By Tree:", font=self.f_text).grid(row=2, column=0, padx=10, pady=10, sticky='w'); self.xgb_colsample = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_colsample.insert(0, "1.0"); self.xgb_colsample.grid(row=2, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Gamma (Min Loss Red):", font=self.f_text).grid(row=2, column=2, padx=10, pady=10, sticky='w'); self.xgb_gamma = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_gamma.insert(0, "0.0"); self.xgb_gamma.grid(row=2, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Reg Alpha (L1):", font=self.f_text).grid(row=3, column=0, padx=10, pady=10, sticky='w'); self.xgb_alpha = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_alpha.insert(0, "0.0"); self.xgb_alpha.grid(row=3, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Reg Lambda (L2):", font=self.f_text).grid(row=3, column=2, padx=10, pady=10, sticky='w'); self.xgb_lambda = ctk.CTkEntry(f, font=self.f_text, width=150); self.xgb_lambda.insert(0, "1.0"); self.xgb_lambda.grid(row=3, column=3, padx=10, pady=10)

    def build_rf_params(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill='both', expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="N Estimators:", font=self.f_text).grid(row=0, column=0, padx=10, pady=10, sticky='w'); self.rf_n_estimators = ctk.CTkEntry(f, font=self.f_text, width=150); self.rf_n_estimators.insert(0, "100"); self.rf_n_estimators.grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Max Depth (0=Unlimited):", font=self.f_text).grid(row=0, column=2, padx=10, pady=10, sticky='w'); self.rf_max_depth = ctk.CTkEntry(f, font=self.f_text, width=150); self.rf_max_depth.insert(0, "0"); self.rf_max_depth.grid(row=0, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Min Samples Split:", font=self.f_text).grid(row=1, column=0, padx=10, pady=10, sticky='w'); self.rf_min_split = ctk.CTkEntry(f, font=self.f_text, width=150); self.rf_min_split.insert(0, "2"); self.rf_min_split.grid(row=1, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Min Samples Leaf:", font=self.f_text).grid(row=1, column=2, padx=10, pady=10, sticky='w'); self.rf_min_leaf = ctk.CTkEntry(f, font=self.f_text, width=150); self.rf_min_leaf.insert(0, "1"); self.rf_min_leaf.grid(row=1, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Max Features:", font=self.f_text).grid(row=2, column=0, padx=10, pady=10, sticky='w'); self.rf_max_features = ctk.CTkComboBox(f, font=self.f_text, values=['1.0', 'sqrt', 'log2'], width=150); self.rf_max_features.set('1.0'); self.rf_max_features.grid(row=2, column=1, padx=10, pady=10)

    def build_regularization_params(self, parent, prefix, default_alpha="0.01"):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill='both', expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="Alpha (Penalty):", font=self.f_text).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        entry_a = ctk.CTkEntry(f, font=self.f_text, width=150); entry_a.insert(0, default_alpha); entry_a.grid(row=0, column=1, padx=10, pady=10); setattr(self, f"{prefix}_alpha", entry_a)
        ctk.CTkLabel(f, text="Fit Intercept:", font=self.f_text).grid(row=0, column=2, padx=10, pady=10, sticky='w')
        combo_i = ctk.CTkComboBox(f, font=self.f_text, values=['True', 'False'], width=150); combo_i.set('True'); combo_i.grid(row=0, column=3, padx=10, pady=10); setattr(self, f"{prefix}_fit_intercept", combo_i)
        ctk.CTkLabel(f, text="Max Iterations:", font=self.f_text).grid(row=1, column=0, padx=10, pady=10, sticky='w')
        entry_i = ctk.CTkEntry(f, font=self.f_text, width=150); entry_i.insert(0, "1000"); entry_i.grid(row=1, column=1, padx=10, pady=10); setattr(self, f"{prefix}_max_iter", entry_i)
        ctk.CTkLabel(f, text="Tolerance:", font=self.f_text).grid(row=1, column=2, padx=10, pady=10, sticky='w')
        entry_t = ctk.CTkEntry(f, font=self.f_text, width=150); entry_t.insert(0, "0.0001"); entry_t.grid(row=1, column=3, padx=10, pady=10); setattr(self, f"{prefix}_tol", entry_t)

    def build_lasso_params(self, parent):
        self.build_regularization_params(parent, "lasso", "0.01")

    def build_ridge_params(self, parent):
        self.build_regularization_params(parent, "ridge", "1.0")
        f = parent.winfo_children()[0] 
        ctk.CTkLabel(f, text="Solver:", font=self.f_text).grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.ridge_solver = ctk.CTkComboBox(f, font=self.f_text, values=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], width=150); self.ridge_solver.set('auto'); self.ridge_solver.grid(row=2, column=1, padx=10, pady=10)

    def build_elasticnet_params(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill='both', expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="Alpha:", font=self.f_text).grid(row=0, column=0, padx=10, pady=10, sticky='w'); self.en_alpha = ctk.CTkEntry(f, font=self.f_text, width=150); self.en_alpha.insert(0, "0.01"); self.en_alpha.grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="L1 Ratio:", font=self.f_text).grid(row=0, column=2, padx=10, pady=10, sticky='w'); self.en_l1_ratio = ctk.CTkEntry(f, font=self.f_text, width=150); self.en_l1_ratio.insert(0, "0.5"); self.en_l1_ratio.grid(row=0, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Fit Intercept:", font=self.f_text).grid(row=1, column=0, padx=10, pady=10, sticky='w'); self.en_fit_intercept = ctk.CTkComboBox(f, font=self.f_text, values=['True', 'False'], width=150); self.en_fit_intercept.set('True'); self.en_fit_intercept.grid(row=1, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Max Iterations:", font=self.f_text).grid(row=1, column=2, padx=10, pady=10, sticky='w'); self.en_max_iter = ctk.CTkEntry(f, font=self.f_text, width=150); self.en_max_iter.insert(0, "1000"); self.en_max_iter.grid(row=1, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Tolerance:", font=self.f_text).grid(row=2, column=0, padx=10, pady=10, sticky='w'); self.en_tol = ctk.CTkEntry(f, font=self.f_text, width=150); self.en_tol.insert(0, "0.0001"); self.en_tol.grid(row=2, column=1, padx=10, pady=10)

    def build_nn_params(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill='both', expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="Hidden Layers (e.g., 100,50):", font=self.f_text).grid(row=0, column=0, padx=10, pady=10, sticky='w'); self.nn_hidden_layers = ctk.CTkEntry(f, font=self.f_text, width=150); self.nn_hidden_layers.insert(0, "100, 50"); self.nn_hidden_layers.grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Activation:", font=self.f_text).grid(row=0, column=2, padx=10, pady=10, sticky='w'); self.nn_activation = ctk.CTkComboBox(f, font=self.f_text, values=['relu', 'tanh', 'logistic', 'identity'], width=150); self.nn_activation.set('relu'); self.nn_activation.grid(row=0, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Solver:", font=self.f_text).grid(row=1, column=0, padx=10, pady=10, sticky='w'); self.nn_solver = ctk.CTkComboBox(f, font=self.f_text, values=['adam', 'sgd', 'lbfgs'], width=150); self.nn_solver.set('adam'); self.nn_solver.grid(row=1, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Alpha (L2 Penalty):", font=self.f_text).grid(row=1, column=2, padx=10, pady=10, sticky='w'); self.nn_alpha = ctk.CTkEntry(f, font=self.f_text, width=150); self.nn_alpha.insert(0, "0.0001"); self.nn_alpha.grid(row=1, column=3, padx=10, pady=10)
        ctk.CTkLabel(f, text="Learning Rate Init:", font=self.f_text).grid(row=2, column=0, padx=10, pady=10, sticky='w'); self.nn_lr_init = ctk.CTkEntry(f, font=self.f_text, width=150); self.nn_lr_init.insert(0, "0.001"); self.nn_lr_init.grid(row=2, column=1, padx=10, pady=10)
        ctk.CTkLabel(f, text="Max Iterations:", font=self.f_text).grid(row=2, column=2, padx=10, pady=10, sticky='w'); self.nn_max_iter = ctk.CTkEntry(f, font=self.f_text, width=150); self.nn_max_iter.insert(0, "200"); self.nn_max_iter.grid(row=2, column=3, padx=10, pady=10)

    # ==========================================
    # WORKER THREAD REGRESSION LOOP
    # ==========================================
    def stop_training(self):
        if messagebox.askyesno("Stop Pipeline", "Are you sure you want to abort the current training process?"):
            self.stop_requested = True
            self.lbl_status.configure(text="Aborting ...", text_color="#ef4444")
            if hasattr(self, 'btn_stop'): self.btn_stop.configure(state="disabled")
            self.log_event("User requested pipeline abort.")
            
    def start_training(self):
        if self.df is None: return messagebox.showwarning("Data Required", "Please load a dataset first.")

        def safe_get(widget_name, default_val):
            try:
                widget = getattr(self, widget_name, None)
                if widget and widget.winfo_exists(): return widget.get()
            except Exception: pass
            return default_val

        try:
            active_algo = getattr(self, 'active_algo', "EcoRETINA")
            target_col = self.target_var.get()
            if not target_col: return messagebox.showwarning("Target Missing", "Please select a Target Variable (Y).")
                
            split_strategy = self.strategy_var.get()
            split_ratio = self.split_var.get() / 100.0
            k_folds = self.k_var.get()

            cont_names = [self.list_cont.get(i) for i in self.list_cont.curselection()] if hasattr(self, 'list_cont') and self.list_cont.winfo_exists() else []
            dummy_names = [self.list_dummy.get(i) for i in self.list_dummy.curselection()] if hasattr(self, 'list_dummy') and self.list_dummy.winfo_exists() else []
            standard_names = [self.list_features.get(i) for i in self.list_features.curselection()] if hasattr(self, 'list_features') and self.list_features.winfo_exists() else []
            
            raw_params = safe_get('eco_params', "[-1.0, 0.0, 1.0]").strip("[]")
            eco_params_list = [float(x.strip()) for x in raw_params.split(',')]
            reg_t = safe_get('eco_reg_type', 'ols')
            
            # --- PARAMÈTRES POUR LES ZÉROS ---
            raw_eps = safe_get('eco_epsilon', 'auto')
            try:
                eps_val = float(raw_eps) if str(raw_eps).lower() != 'auto' else 'auto'
            except ValueError:
                eps_val = 'auto'
            
            eco_kwargs = {
                "cross_dummy": True if safe_get('eco_cross_dummy_cb', 'False') == 'True' else False,
                "max_r2": float(safe_get('eco_max_r2', 0.50)), "grid": float(safe_get('eco_grid', 0.005)),
                "reg_type": 'linear' if reg_t == 'ols' else reg_t, "loss": safe_get('eco_loss', 'mse'), 
                "max_instances": int(safe_get('eco_max_instances', 100000)), "max_reg": int(safe_get('eco_max_reg', 100)), 
                "model_step": int(safe_get('eco_model_step', 1)), "chunk_size": int(safe_get('eco_chunk_size', 500)), 
                "seed": int(safe_get('eco_seed', 8)), "cov_type": safe_get('eco_cov_type', 'nonrobust'),
                "handle_zeros": safe_get('eco_handle_zeros', 'prevent_division'),
                "epsilon": eps_val
            }
            other_kwargs = {
                "ols_fit_intercept": True if safe_get('ols_fit_intercept', 'True') == 'True' else False,
                "xgb_n": int(safe_get('xgb_n_estimators', 100)), "xgb_depth": int(safe_get('xgb_max_depth', 6)),
                "xgb_lr": float(safe_get('xgb_lr', 0.1)), "xgb_subsample": float(safe_get('xgb_subsample', 1.0)),
                "xgb_colsample": float(safe_get('xgb_colsample', 1.0)), "xgb_gamma": float(safe_get('xgb_gamma', 0.0)),
                "xgb_alpha": float(safe_get('xgb_alpha', 0.0)), "xgb_lambda": float(safe_get('xgb_lambda', 1.0)),
                "rf_n": int(safe_get('rf_n_estimators', 100)), "rf_depth": int(safe_get('rf_max_depth', 0)),
                "rf_split": int(safe_get('rf_min_split', 2)), "rf_leaf": int(safe_get('rf_min_leaf', 1)),
                "rf_max_feat": None if safe_get('rf_max_features', '1.0') == '1.0' else safe_get('rf_max_features', 'sqrt'),
                "lasso_alpha": float(safe_get('lasso_alpha', 0.01)), "lasso_fit": True if safe_get('lasso_fit_intercept', 'True') == 'True' else False,
                "lasso_iter": int(safe_get('lasso_max_iter', 1000)), "lasso_tol": float(safe_get('lasso_tol', 0.0001)),
                "ridge_alpha": float(safe_get('ridge_alpha', 1.0)), "ridge_fit": True if safe_get('ridge_fit_intercept', 'True') == 'True' else False,
                "ridge_iter": int(safe_get('ridge_max_iter', 1000)), "ridge_tol": float(safe_get('ridge_tol', 0.0001)), "ridge_solver": safe_get('ridge_solver', 'auto'),
                "en_alpha": float(safe_get('en_alpha', 0.01)), "en_l1": float(safe_get('en_l1_ratio', 0.5)),
                "en_fit": True if safe_get('en_fit_intercept', 'True') == 'True' else False, "en_iter": int(safe_get('en_max_iter', 1000)), "en_tol": float(safe_get('en_tol', 0.0001)),
                "nn_layers": safe_get('nn_hidden_layers', "100"), "nn_act": safe_get('nn_activation', "relu"), "nn_sol": safe_get('nn_solver', "adam"),
                "nn_alpha": float(safe_get('nn_alpha', 0.0001)), "nn_lr": float(safe_get('nn_lr_init', 0.001)), "nn_iter": int(safe_get('nn_max_iter', 200))
            }
            config = {
                "algo": active_algo, "target_col": target_col, "split_strategy": split_strategy, "split_ratio": split_ratio, "k_folds": k_folds,
                "cont_names": cont_names, "dummy_names": dummy_names, "standard_names": standard_names,
                "eco_params_list": eco_params_list, "eco_kwargs": eco_kwargs, "other_kwargs": other_kwargs
            }
            self.log_event(f"Initializing pipeline for: {active_algo}. Strategy: {split_strategy}. Target: '{target_col}'.")
        except Exception as e:
            self.log_event(f"Config setup failed: {str(e)}")
            return messagebox.showerror("Config Error", f"Failed to extract settings:\n{str(e)}")
        
        self.stop_requested = False
        if hasattr(self, 'btn_run'): self.btn_run.configure(state="disabled")
        if hasattr(self, 'btn_stop'): self.btn_stop.configure(state="normal")
        self.progress.pack(side='left', padx=15); self.progress.start()
        self.lbl_status.configure(text=f"Running {active_algo}...", text_color="#1f6aa5")
        threading.Thread(target=self.run_training_logic, args=(config,), daemon=True).start()

    def run_training_logic(self, config):
        try:
            start_time = time.time()
            target_col, split_ratio, split_strategy, k_folds = config["target_col"], config["split_ratio"], config["split_strategy"], config["k_folds"]
            if target_col not in self.df.columns: raise ValueError("Target variable missing.")
            
            df_clean = self.df.dropna(subset=[target_col]).fillna(0)
            if len(df_clean) < 10: raise ValueError("Clean dataset has <10 rows.")
            
            y = df_clean[target_col].values
            model = None
            features_list = []
            selected_features = []

            if config["algo"] == 'EcoRETINA':
                if not ECO_RETINA_AVAILABLE: raise ImportError("eco_retina.py is missing.")
                raw_cont = [c for c in config["cont_names"] if c != target_col]
                raw_dummy = [d for d in config["dummy_names"] if d != target_col]
                clean_dummy = [d for d in raw_dummy if d not in raw_cont]
                selected_features = list(dict.fromkeys(raw_cont + clean_dummy))
                if not selected_features: raise ValueError("Select at least one valid predictor.")
                
                X_eco_df = df_clean[selected_features]
                con_cols_indices = [selected_features.index(c) for c in raw_cont]
                dummy_cols_indices = [selected_features.index(d) for d in clean_dummy]
                X_eco = X_eco_df.values
                
                # Ancienne logique commentée car maintenant gérée par EcoRETINA en interne
                # if con_cols_indices:
                #     mask = (X_eco[:, con_cols_indices] != 0).all(axis=1)
                #     X_eco, y_eco = X_eco[mask], y[mask]
                #     if len(y_eco) < 10: raise ValueError(f"CRITICAL LOSS: Only {len(y_eco)} rows remain.")
                # else: y_eco = y
                
                y_eco = y
                
                if split_strategy == "K-Fold Cross Validation":
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    r2_tr, mape_tr, rmse_tr = [], [], []
                    r2_te, mape_te, rmse_te = [], [], []
                    for train_idx, test_idx in kf.split(X_eco):
                        if getattr(self, 'stop_requested', False): raise InterruptedError()
                        X_train, X_test, y_train, y_test = X_eco[train_idx], X_eco[test_idx], y_eco[train_idx], y_eco[test_idx]
                        model = EcoRETINA()
                        model.fit(y=y_train, X=X_train, con_cols_indices=con_cols_indices, dummy_cols_indices=dummy_cols_indices, col_names=selected_features, params=config["eco_params_list"], **config["eco_kwargs"])
                        y_train_pred = model.predict(X_train)
                        from utils import precompute_powers, generate_features 
                        X_test_transformed = generate_features(X_test, model.combinations, precompute_powers(X_test, model.params), model.params, model.chunk_size)
                        y_test_pred = model.sm_model.predict(X_test_transformed)
                        r2_tr.append(r2_score(y_train, y_train_pred)); mape_tr.append(mean_absolute_percentage_error(y_train, y_train_pred) * 100); rmse_tr.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
                        r2_te.append(r2_score(y_test, y_test_pred)); mape_te.append(mean_absolute_percentage_error(y_test, y_test_pred) * 100); rmse_te.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
                    r2_train, mape_train, rmse_train = np.mean(r2_tr), np.mean(mape_tr), np.mean(rmse_tr)
                    r2_test, mape_test, rmse_test = np.mean(r2_te), np.mean(mape_te), np.mean(rmse_te)
                    X_test_for_viz = X_test_transformed; features_list = list(model.sm_model.params.index)
                else:
                    X_train, X_test, y_train, y_test = (X_eco, X_eco, y_eco, y_eco) if split_ratio >= 1.0 else train_test_split(X_eco, y_eco, train_size=split_ratio, random_state=42)
                    model = EcoRETINA()
                    model.fit(y=y_train, X=X_train, con_cols_indices=con_cols_indices, dummy_cols_indices=dummy_cols_indices, col_names=selected_features, params=config["eco_params_list"], **config["eco_kwargs"])
                    if getattr(self, 'stop_requested', False): raise InterruptedError()
                    y_train_pred = model.predict(X_train)
                    from utils import precompute_powers, generate_features 
                    X_test_transformed = generate_features(X_test, model.combinations, precompute_powers(X_test, model.params), model.params, model.chunk_size)
                    y_test_pred = model.sm_model.predict(X_test_transformed)
                    X_test_for_viz = X_test_transformed; features_list = list(model.sm_model.params.index)
                    r2_train, mape_train, rmse_train = r2_score(y_train, y_train_pred), mean_absolute_percentage_error(y_train, y_train_pred) * 100, np.sqrt(mean_squared_error(y_train, y_train_pred))
                    r2_test, mape_test, rmse_test = r2_score(y_test, y_test_pred), mean_absolute_percentage_error(y_test, y_test_pred) * 100, np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            else:
                raw_features = config.get("standard_names", [])
                selected_features = [f for f in raw_features if f != target_col]
                if not selected_features: raise ValueError("Select at least one predictor.")
                features_list = selected_features
                X_encoded = df_clean[features_list].values
                kwargs = config["other_kwargs"]
                
                if split_strategy == "K-Fold Cross Validation":
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    r2_tr, mape_tr, rmse_tr = [], [], []
                    r2_te, mape_te, rmse_te = [], [], []
                    for train_idx, test_idx in kf.split(X_encoded):
                        if getattr(self, 'stop_requested', False): raise InterruptedError()
                        X_train, X_test, y_train, y_test = X_encoded[train_idx], X_encoded[test_idx], y[train_idx], y[test_idx]
                        
                        if config["algo"] == 'OLS':
                            if kwargs["ols_fit_intercept"]: X_train_fit, X_test_fit, f_list = sm.add_constant(X_train, has_constant='add'), sm.add_constant(X_test, has_constant='add'), (["const"] + features_list if "const" not in features_list else features_list)
                            else: X_train_fit, X_test_fit, f_list = X_train, X_test, features_list
                            sm_res = sm.OLS(y_train, pd.DataFrame(X_train_fit, columns=f_list)).fit()
                            model, y_train_pred, y_test_pred, X_test_for_viz, features_list = OLSWrapper(sm_res), sm_res.predict(pd.DataFrame(X_train_fit, columns=f_list)), sm_res.predict(pd.DataFrame(X_test_fit, columns=f_list)), X_test_fit, f_list
                        
                        # --- POST-ESTIMATION MODIFICATION HERE ---
                        elif config["algo"] in ['Lasso', 'Ridge', 'ElasticNet']:
                            if config["algo"] == 'Lasso': model_pen = Lasso(alpha=kwargs["lasso_alpha"], fit_intercept=kwargs["lasso_fit"], max_iter=kwargs["lasso_iter"], tol=kwargs["lasso_tol"]); fit_int = kwargs["lasso_fit"]
                            elif config["algo"] == 'Ridge': model_pen = Ridge(alpha=kwargs["ridge_alpha"], fit_intercept=kwargs["ridge_fit"], max_iter=kwargs["ridge_iter"], tol=kwargs["ridge_tol"], solver=kwargs["ridge_solver"]); fit_int = kwargs["ridge_fit"]
                            elif config["algo"] == 'ElasticNet': model_pen = ElasticNet(alpha=kwargs["en_alpha"], l1_ratio=kwargs["en_l1"], fit_intercept=kwargs["en_fit"], max_iter=kwargs["en_iter"], tol=kwargs["en_tol"]); fit_int = kwargs["en_fit"]
                            
                            model_pen.fit(X_train, y_train)
                            
                            # We only keep significant coefficients (> 1e-5)
                            sel_idx = np.where(np.abs(model_pen.coef_) > 1e-5)[0]
                            if len(sel_idx) == 0: sel_idx = np.arange(X_train.shape[1]) # Anti-crash if it removes everything
                            
                            X_tr_sel, X_te_sel = X_train[:, sel_idx], X_test[:, sel_idx]
                            f_list = [features_list[i] for i in sel_idx]
                            selected_features = f_list.copy()
                            
                            if fit_int:
                                X_tr_fit, X_te_fit = sm.add_constant(X_tr_sel, has_constant='add'), sm.add_constant(X_te_sel, has_constant='add')
                                f_list = (["const"] + f_list if "const" not in f_list else f_list)
                            else: 
                                X_tr_fit, X_te_fit = X_tr_sel, X_te_sel
                                
                            X_tr_fit_df = pd.DataFrame(X_tr_fit, columns=f_list)
                            X_te_fit_df = pd.DataFrame(X_te_fit, columns=f_list)

                            sm_res = sm.OLS(y_train, X_tr_fit_df).fit()
                            model = OLSWrapper(sm_res)
                            y_train_pred = sm_res.predict(X_tr_fit_df)
                            y_test_pred = sm_res.predict(X_te_fit_df)
                            X_test_for_viz = X_te_fit
                            features_list = f_list

                        else:
                            if config["algo"] == 'XGBoost': model = xgb.XGBRegressor(n_estimators=kwargs["xgb_n"], max_depth=kwargs["xgb_depth"], learning_rate=kwargs["xgb_lr"], subsample=kwargs["xgb_subsample"], colsample_bytree=kwargs["xgb_colsample"], gamma=kwargs["xgb_gamma"], reg_alpha=kwargs["xgb_alpha"], reg_lambda=kwargs["xgb_lambda"], random_state=42)
                            elif config["algo"] == 'Random Forest': depth = kwargs["rf_depth"]; model = RandomForestRegressor(n_estimators=kwargs["rf_n"], max_depth=None if depth == 0 else depth, min_samples_split=kwargs["rf_split"], min_samples_leaf=kwargs["rf_leaf"], max_features=kwargs["rf_max_feat"], random_state=42)
                            elif config["algo"] == 'Neural Network': layers = tuple(int(x.strip()) for x in kwargs["nn_layers"].split(',')); model = MLPRegressor(hidden_layer_sizes=layers, activation=kwargs["nn_act"], solver=kwargs["nn_sol"], alpha=kwargs["nn_alpha"], learning_rate_init=kwargs["nn_lr"], max_iter=kwargs["nn_iter"], random_state=42)
                            model.fit(X_train, y_train)
                            y_train_pred, y_test_pred, X_test_for_viz = model.predict(X_train), model.predict(X_test), X_test
                        
                        r2_tr.append(r2_score(y_train, y_train_pred)); mape_tr.append(mean_absolute_percentage_error(y_train, y_train_pred) * 100); rmse_tr.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
                        r2_te.append(r2_score(y_test, y_test_pred)); mape_te.append(mean_absolute_percentage_error(y_test, y_test_pred) * 100); rmse_te.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
                    r2_train, mape_train, rmse_train = np.mean(r2_tr), np.mean(mape_tr), np.mean(rmse_tr)
                    r2_test, mape_test, rmse_test = np.mean(r2_te), np.mean(mape_te), np.mean(rmse_te)
                else:
                    X_train, X_test, y_train, y_test = (X_encoded, X_encoded, y, y) if split_ratio >= 1.0 else train_test_split(X_encoded, y, train_size=split_ratio, random_state=42)
                    X_test_for_viz = X_test
                    
                    if config["algo"] == 'OLS':
                        if kwargs["ols_fit_intercept"]:
                            X_train_fit, X_test_fit = sm.add_constant(X_train, has_constant='add'), sm.add_constant(X_test, has_constant='add')
                            X_test_for_viz = X_test_fit
                            if "const" not in features_list: features_list = ["const"] + features_list
                        else: X_train_fit, X_test_fit = X_train, X_test
                        X_train_fit_df, X_test_fit_df = pd.DataFrame(X_train_fit, columns=features_list), pd.DataFrame(X_test_fit, columns=features_list)
                        tracker = EmissionsTracker(tracking_mode='process', output_file='standard_emissions.csv', project_name='OLS', log_level="error")
                        tracker.start(); sm_res = sm.OLS(y_train, X_train_fit_df).fit(); tracker.stop()
                        if getattr(self, 'stop_requested', False): raise InterruptedError()
                        model, y_train_pred, y_test_pred = OLSWrapper(sm_res), sm_res.predict(X_train_fit_df), sm_res.predict(X_test_fit_df)

                    # --- POST-ESTIMATION MODIFICATION HERE ---
                    elif config["algo"] in ['Lasso', 'Ridge', 'ElasticNet']:
                        if config["algo"] == 'Lasso': model_pen = Lasso(alpha=kwargs["lasso_alpha"], fit_intercept=kwargs["lasso_fit"], max_iter=kwargs["lasso_iter"], tol=kwargs["lasso_tol"]); fit_int = kwargs["lasso_fit"]
                        elif config["algo"] == 'Ridge': model_pen = Ridge(alpha=kwargs["ridge_alpha"], fit_intercept=kwargs["ridge_fit"], max_iter=kwargs["ridge_iter"], tol=kwargs["ridge_tol"], solver=kwargs["ridge_solver"]); fit_int = kwargs["ridge_fit"]
                        elif config["algo"] == 'ElasticNet': model_pen = ElasticNet(alpha=kwargs["en_alpha"], l1_ratio=kwargs["en_l1"], fit_intercept=kwargs["en_fit"], max_iter=kwargs["en_iter"], tol=kwargs["en_tol"]); fit_int = kwargs["en_fit"]

                        tracker = EmissionsTracker(tracking_mode='process', output_file='standard_emissions.csv', project_name=config["algo"], log_level="error")
                        tracker.start()
                        model_pen.fit(X_train, y_train)
                        
                        sel_idx = np.where(np.abs(model_pen.coef_) > 1e-5)[0]
                        if len(sel_idx) == 0: raise ValueError(f"The penalty (Alpha) of algorithm {config['algo']} is too strong. All coefficients have been reduced to zero.")
                        
                        X_tr_sel, X_te_sel = X_train[:, sel_idx], X_test[:, sel_idx]
                        f_list = [features_list[i] for i in sel_idx]
                        selected_features = f_list.copy() 
                        
                        if fit_int:
                            X_tr_fit = sm.add_constant(X_tr_sel, has_constant='add')
                            X_te_fit = sm.add_constant(X_te_sel, has_constant='add')
                            f_list = ["const"] + f_list if "const" not in f_list else f_list
                        else:
                            X_tr_fit, X_te_fit = X_tr_sel, X_te_sel

                        X_train_fit_df = pd.DataFrame(X_tr_fit, columns=f_list)
                        X_test_fit_df = pd.DataFrame(X_te_fit, columns=f_list)

                        sm_res = sm.OLS(y_train, X_train_fit_df).fit()
                        tracker.stop()
                        if getattr(self, 'stop_requested', False): raise InterruptedError()
                        
                        model = OLSWrapper(sm_res)
                        y_train_pred = sm_res.predict(X_train_fit_df)
                        y_test_pred = sm_res.predict(X_test_fit_df)
                        X_test_for_viz = X_te_fit
                        features_list = f_list

                    else:
                        if config["algo"] == 'XGBoost': model = xgb.XGBRegressor(n_estimators=kwargs["xgb_n"], max_depth=kwargs["xgb_depth"], learning_rate=kwargs["xgb_lr"], subsample=kwargs["xgb_subsample"], colsample_bytree=kwargs["xgb_colsample"], gamma=kwargs["xgb_gamma"], reg_alpha=kwargs["xgb_alpha"], reg_lambda=kwargs["xgb_lambda"], random_state=42)
                        elif config["algo"] == 'Random Forest': depth = kwargs["rf_depth"]; model = RandomForestRegressor(n_estimators=kwargs["rf_n"], max_depth=None if depth == 0 else depth, min_samples_split=kwargs["rf_split"], min_samples_leaf=kwargs["rf_leaf"], max_features=kwargs["rf_max_feat"], random_state=42)
                        elif config["algo"] == 'Lasso': model = Lasso(alpha=kwargs["lasso_alpha"], fit_intercept=kwargs["lasso_fit"], max_iter=kwargs["lasso_iter"], tol=kwargs["lasso_tol"])
                        elif config["algo"] == 'Ridge': model = Ridge(alpha=kwargs["ridge_alpha"], fit_intercept=kwargs["ridge_fit"], max_iter=kwargs["ridge_iter"], tol=kwargs["ridge_tol"], solver=kwargs["ridge_solver"])
                        elif config["algo"] == 'ElasticNet': model = ElasticNet(alpha=kwargs["en_alpha"], l1_ratio=kwargs["en_l1"], fit_intercept=kwargs["en_fit"], max_iter=kwargs["en_iter"], tol=kwargs["en_tol"])
                        elif config["algo"] == 'Neural Network': layers = tuple(int(x.strip()) for x in kwargs["nn_layers"].split(',')); model = MLPRegressor(hidden_layer_sizes=layers, activation=kwargs["nn_act"], solver=kwargs["nn_sol"], alpha=kwargs["nn_alpha"], learning_rate_init=kwargs["nn_lr"], max_iter=kwargs["nn_iter"], random_state=42)

                        tracker = EmissionsTracker(tracking_mode='process', output_file='standard_emissions.csv', project_name=config["algo"], log_level="error")
                        tracker.start(); model.fit(X_train, y_train)
                        if getattr(self, 'stop_requested', False): raise InterruptedError()
                        tracker.stop(); y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)

                    r2_train, mape_train, rmse_train = r2_score(y_train, y_train_pred), mean_absolute_percentage_error(y_train, y_train_pred) * 100, np.sqrt(mean_squared_error(y_train, y_train_pred))
                    r2_test, mape_test, rmse_test = r2_score(y_test, y_test_pred), mean_absolute_percentage_error(y_test, y_test_pred) * 100, np.sqrt(mean_squared_error(y_test, y_test_pred))

            n_train, p_train = X_train.shape[0], len(features_list)
            n_test = X_test.shape[0]
            adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p_train - 1) if n_train > p_train + 1 else float('nan')
            adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_train - 1) if n_test > p_train + 1 else float('nan')

            residuals = y_test - y_test_pred
            norm_p_value = stats.shapiro(residuals)[1] if len(residuals) >= 3 else float('nan')

            emissions_val, energy_val = 0.0, 0.0
            try:
                if config["algo"] == 'EcoRETINA': em_df = model.load_emissions_report()
                else: 
                    em_df = pd.read_csv('standard_emissions.csv')
                    em_df = em_df[em_df['project_name'] == config["algo"]]
                if not em_df.empty:
                    last_run = em_df.iloc[-1]
                    emissions_val, energy_val = float(last_run.get('emissions', 0.0)), float(last_run.get('energy_consumed', 0.0))
            except Exception: pass

            r2_carbon_ratio = (r2_test / emissions_val) if emissions_val > 0 else float('inf')
            elapsed_time = time.time() - start_time
            
            all_metrics = {
                "R2_Train": r2_train, "Adj_R2_Train": adj_r2_train, "MAPE_Train": mape_train, "RMSE_Train": rmse_train,
                "R2_Test": r2_test, "Adj_R2_Test": adj_r2_test, "MAPE_Test": mape_test, "RMSE_Test": rmse_test,
                "Norm_P_Value": norm_p_value, "Emissions": emissions_val, "Energy": energy_val, "R2_CO2_Ratio": r2_carbon_ratio,
                "Time": elapsed_time
            }

            if self.root.winfo_exists():
                self.root.after(0, lambda: self.training_complete(config["algo"], config["target_col"], all_metrics, len(features_list), model, features_list, X_test_for_viz, y_test, y_test_pred, selected_features, config))
        
        except Exception as e:
            error_msg = traceback.format_exc()
            if self.root.winfo_exists():
                self.root.after(0, lambda: self.btn_run.configure(state="normal") if hasattr(self, 'btn_run') else None)
                self.root.after(0, lambda: self.btn_stop.configure(state="disabled") if hasattr(self, 'btn_stop') else None)
                self.root.after(0, self.progress.stop); self.root.after(0, self.progress.pack_forget)
                
                if isinstance(e, InterruptedError):
                    self.root.after(0, lambda: self.log_event("Execution gracefully aborted by user."))
                    self.root.after(0, lambda: self.lbl_status.configure(text="Aborted.", text_color="#ef4444"))
                else:
                    self.root.after(0, lambda: self.log_event(f"CRITICAL ERROR during execution of {config['algo']}: {str(e)}"))
                    self.root.after(0, lambda: messagebox.showerror("Pipeline Execution Error", f"{str(e)}\n\nTechnical:\n{error_msg}"))
                    self.root.after(0, lambda: self.lbl_status.configure(text="Pipeline failed.", text_color="#ef4444"))

    def run_prediction(self):
        if getattr(self, 'df_predict', None) is None: return messagebox.showwarning("Error", "Please load a dataset for prediction first.")
        selected_run = self.predict_model_combo.get()
        if not selected_run or selected_run not in self.run_history: return messagebox.showwarning("Error", "Please select a valid trained model from the list.")

        run_data = self.run_history[selected_run]
        model, model_name, raw_features, target_col = run_data["model"], run_data["model_name"], run_data.get("raw_features", []), run_data.get("target_col", "Target")

        missing_cols = [col for col in raw_features if col not in self.df_predict.columns]
        if missing_cols: return messagebox.showerror("Missing Variables", f"Missing columns in new dataset:\n\n{chr(10).join(f'- {c}' for c in missing_cols)}")

        try:
            X_new_df = self.df_predict[raw_features].copy().fillna(0)
            X_new = X_new_df.values

            if model_name == 'EcoRETINA': predictions = model.predict(X_new)
            # --- MODIFICATION FOR POST-LASSO PREDICTIONS HERE ---
            elif model_name in ['OLS', 'Lasso', 'Ridge', 'ElasticNet']:
                features_list = run_data["feature_names"]
                X_new_fit = sm.add_constant(X_new, has_constant='add') if "const" in features_list else X_new
                predictions = model.predict(pd.DataFrame(X_new_fit, columns=features_list))
            else: predictions = model.predict(X_new)

            pred_col_name = f"Predicted_{target_col}_{selected_run}"
            self.df_predict[pred_col_name] = predictions
            self.log_event(f"Prediction successful for run '{selected_run}'.")
            messagebox.showinfo("Success", f"Predictions added to column:\n'{pred_col_name}'")
        except Exception as e: messagebox.showerror("Prediction Error", str(e))

    def training_complete(self, model_name, target_col, metrics, features_count, model, features_list, X_test, y_test, y_test_pred, raw_features, algo_config=None):
        if hasattr(self, 'btn_run'): self.btn_run.configure(state="normal")
        if hasattr(self, 'btn_stop'): self.btn_stop.configure(state="disabled")
        self.progress.stop(); self.progress.pack_forget()
        self.lbl_status.configure(text=f"Last execution: {model_name} (Success)", text_color="#22c55e")
        
        run_name = f"Run_{time.strftime('%H%M%S')}"
        self.run_history[run_name] = {
            "model": model, "model_name": model_name, "target_col": target_col, "feature_names": features_list, 
            "raw_features": raw_features, "metrics": metrics, "X_test": X_test, "y_test": y_test, 
            "y_test_pred": y_test_pred, "config": algo_config 
        }
        self.latest_run_by_algo[model_name] = run_name 
        self.log_event(f"Execution successful: {model_name} ({run_name}). Test R2: {metrics['R2_Test']:.4f}, Test MAPE: {metrics['MAPE_Test']:.2f}%")
        
        self.add_comparison_result(run_name, model_name, metrics, features_count)
        self.tabview.set("3. Compare Results")
        self.refresh_predict_models()

        warnings_list = []
        if model_name in ["Lasso", "Ridge", "ElasticNet"] and -0.05 <= metrics['R2_Test'] <= 0.05:
            warnings_list.append("⚠️ LINEAR COLLAPSE: The R² is close to 0. The algorithm was probably penalized too heavily. Go to 'Hyperparameters' and reduce the 'Alpha' value (e.g., 0.01 or 0.001).")
        if metrics['R2_Train'] > 0.8 and metrics['R2_Test'] < 0.4:
            warnings_list.append("⚠️ OVERFITTING: The model learns your training data perfectly but collapses on new data (Test). Try reducing model complexity (e.g., max depth for trees).")
        if metrics['MAPE_Test'] > 1e4:
            warnings_list.append("⚠️ MAPE EXPLOSION: The MAPE shows a gigantic number. This happens mathematically when dividing by a true value equal to (or very close to) zero. Ignore this metric and rely on RMSE (or undo the standardization of the target).")
        
        if warnings_list:
            warning_msg = "\n\n".join(warnings_list)
            messagebox.showwarning("Automatic Analysis & Warnings", warning_msg)

    # ==========================================
    # PREDICTION TAB (INFERENCE ON NEW DATA)
    # ==========================================
    def build_predict_tab(self):
        container = ctk.CTkFrame(self.tab_predict, fg_color="transparent")
        container.pack(expand=True, fill='both', padx=20, pady=20)

        load_frame = ctk.CTkFrame(container, corner_radius=10); load_frame.pack(fill='x', pady=(0, 20))
        ctk.CTkLabel(load_frame, text="1. Load New Dataset (For Inference)", font=self.f_subtitle).pack(anchor="w", padx=20, pady=(15, 5))

        inner_load = ctk.CTkFrame(load_frame, fg_color="transparent"); inner_load.pack(expand=True, fill='none', pady=(10, 20))
        ctk.CTkButton(inner_load, text="Browse New Dataset", font=self.f_text, command=self.load_predict_file, fg_color="#3b3b3b", hover_color="#4b4b4b").pack(side='left', padx=10)
        self.lbl_predict_file = ctk.CTkLabel(inner_load, text="No prediction file loaded.", font=self.f_text, text_color="gray"); self.lbl_predict_file.pack(side='left', padx=15)

        model_frame = ctk.CTkFrame(container, corner_radius=10); model_frame.pack(fill='x', pady=(0, 20))
        ctk.CTkLabel(model_frame, text="2. Select Trained Model", font=self.f_subtitle).pack(anchor="w", padx=20, pady=(15, 5))

        inner_model = ctk.CTkFrame(model_frame, fg_color="transparent"); inner_model.pack(fill='x', padx=20, pady=(10, 20))
        self.predict_model_combo = ctk.CTkComboBox(inner_model, values=list(self.run_history.keys()), font=self.f_text, state="readonly", width=350); self.predict_model_combo.pack(side='left')
        ctk.CTkButton(inner_model, text="Refresh List", font=self.f_text, width=120, command=self.refresh_predict_models, fg_color="#3b3b3b", hover_color="#4b4b4b").pack(side='left', padx=15)

        action_frame = ctk.CTkFrame(container, corner_radius=10); action_frame.pack(fill='x', pady=(0, 20))
        ctk.CTkLabel(action_frame, text="3. Execution & Export", font=self.f_subtitle).pack(anchor="w", padx=20, pady=(15, 5))

        inner_action = ctk.CTkFrame(action_frame, fg_color="transparent"); inner_action.pack(fill='x', padx=20, pady=(10, 20))
        ctk.CTkButton(inner_action, text="Run Prediction", font=ctk.CTkFont(size=18, weight="bold"), command=self.run_prediction, fg_color="#68B946", hover_color="#539438").pack(side='left')
        ctk.CTkButton(inner_action, text="Visualize Results", font=self.f_text, command=self.visualize_predict_data, fg_color="#1f6aa5", hover_color="#144870").pack(side='left', padx=15)
        ctk.CTkButton(inner_action, text="Export Predictions to CSV", font=self.f_text, command=self.export_predict_data).pack(side='right')

    def load_predict_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xlsx *.json"), ("All files", "*.*")])
        if not filepath: return
        try:
            if filepath.endswith('.csv'): self.df_predict = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'): self.df_predict = pd.read_excel(filepath)
            elif filepath.endswith('.json'): self.df_predict = pd.read_json(filepath)

            self.lbl_predict_file.configure(text=f"Loaded: {os.path.basename(filepath)} | Rows: {len(self.df_predict)}", text_color="#1f6aa5")
            self.log_event(f"Prediction dataset '{os.path.basename(filepath)}' loaded successfully.")
            messagebox.showinfo("Success", "Prediction dataset loaded successfully.")
        except Exception as e:
            self.log_event(f"Error loading prediction file: {str(e)}")
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def refresh_predict_models(self):
        if hasattr(self, 'predict_model_combo'):
            runs = list(self.run_history.keys())
            self.predict_model_combo.configure(values=runs)
            if runs and not self.predict_model_combo.get(): self.predict_model_combo.set(runs[-1])
            self.log_event("Refreshed prediction models list.")

    

    def visualize_predict_data(self):
        if getattr(self, 'df_predict', None) is None: return messagebox.showwarning("No Data", "Please load a prediction dataset first.")
        self.visualize_dataframe(self.df_predict, "Prediction Dataset Visualization")

    def export_predict_data(self):
        if getattr(self, 'df_predict', None) is None: return messagebox.showwarning("No Data", "Please load a prediction dataset first.")
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="predictions.csv", filetypes=[("CSV", "*.csv")])
        if filepath:
            try: self.df_predict.to_csv(filepath, index=False); messagebox.showinfo("Export Saved", f"Predictions saved to:\n{filepath}")
            except Exception as e: messagebox.showerror("Export Error", str(e))

    # ==========================================
    # STATISTICAL DETAILED SUMMARY REPORT WINDOW
    # ==========================================
    def view_summary(self):
        active_tab = getattr(self, 'active_algo', "EcoRETINA")
        run_name = self.latest_run_by_algo.get(active_tab)
        if not run_name: return messagebox.showinfo("No Run Available", f"No saved execution found for '{active_tab}'.")
        self.show_summary_window(run_name)

    def show_summary_window(self, run_name):
        run_data = self.run_history.get(run_name)
        if not run_data: return

        model, model_name, feature_names, metrics = run_data["model"], run_data["model_name"], run_data["feature_names"], run_data["metrics"]
        X_test, y_test, y_test_pred = run_data["X_test"], run_data["y_test"], run_data["y_test_pred"]

        top = ctk.CTkToplevel(self.root)
        top.title(f"Detailed Analysis - {run_name} ({model_name})")
        top.geometry("1400x850") 
        top.lift()
        top.attributes('-topmost', True)
        top.after(200, lambda: top.attributes('-topmost', False))
        top.focus_force()

        header = ctk.CTkFrame(top, height=60, corner_radius=0, fg_color="#1a1a1a"); header.pack(fill='x'); header.pack_propagate(False)
        ctk.CTkLabel(header, text=f"Statistical Report: {run_name}", font=self.f_subtitle, text_color="white").pack(side='left', padx=20)
        ctk.CTkLabel(header, text=model_name, font=self.f_subtitle, text_color="#1f6aa5").pack(side='right', padx=20)

        report_tabs = ctk.CTkTabview(top)
        report_tabs._segmented_button.configure(font=ctk.CTkFont(size=16, weight="bold"))
        report_tabs.pack(expand=True, fill='both', padx=20, pady=10)
        
        tab_stats = report_tabs.add("Statistical Report")
        tab_formula = report_tabs.add("Visual Formula")
        tab_graphs = report_tabs.add("Graphical Analysis")

        text_area = ctk.CTkTextbox(tab_stats, font=ctk.CTkFont(family="Consolas", size=16), wrap="none"); text_area.pack(expand=True, fill='both', pady=(0, 10))

        summary_lines = []
        data_rows = []
        y_mean = np.mean(y_test) if np.mean(y_test) != 0 else 1e-9 

        try:
            summary_lines.append("="*140)
            summary_lines.append(f"{'PIPELINE PERFORMANCE METRICS':^140}")
            summary_lines.append("="*140)
            summary_lines.append(f" TRAIN -> R-squared: {metrics['R2_Train']:>9.6f} | Adj R2: {metrics['Adj_R2_Train']:>9.6f} | RMSE: {metrics['RMSE_Train']:>10.4f} | MAPE: {metrics['MAPE_Train']:>8.2f}%")
            summary_lines.append(f" TEST  -> R-squared: {metrics['R2_Test']:>9.6f} | Adj R2: {metrics['Adj_R2_Test']:>9.6f} | RMSE: {metrics['RMSE_Test']:>10.4f} | MAPE: {metrics['MAPE_Test']:>8.2f}%")
            summary_lines.append(f" Residuals Normality (Shapiro-Wilk) : P-Value = {metrics['Norm_P_Value']:.6f}  (>0.05 implies normal distribution)")
            summary_lines.append(f" Processing Time                    : {metrics['Time']:.3f} seconds\n")

            if model_name in ['EcoRETINA', 'OLS', 'Lasso', 'Ridge', 'ElasticNet'] and (hasattr(model, 'coef_') or hasattr(model, 'sm_model')):
                summary_lines.append("="*140)
                summary_lines.append(f"{'ENGINEERED FEATURES, DERIVATIVES & ELASTICITIES (SORTED BY IMPACT)':^140}")
                summary_lines.append("="*140)
                summary_lines.append(f" {'Variable':<35} | {'Coefficient (Marginal Effect)':>30} | {'Mean Elasticity':>20} | {'T-Stat':>10} | {'P-Value':>10}")
                summary_lines.append("-" * 140)
                
                is_sm = hasattr(model, 'sm_model')
                params_dict = model.sm_model.params if is_sm else dict(zip(feature_names, model.coef_))
                conf_int_df = model.sm_model.conf_int(alpha=0.05) if is_sm else None
                
                sorted_feats = sorted(feature_names, key=lambda f: abs(params_dict.get(f, 0)), reverse=True)

                for i, feat in enumerate(sorted_feats):
                    coef = params_dict.get(feat, 0.0)
                    tstat = model.sm_model.tvalues.get(feat, float('nan')) if is_sm else float('nan')
                    pval = model.sm_model.pvalues.get(feat, float('nan')) if is_sm else float('nan')
                    try: ci_lower, ci_upper = conf_int_df.loc[feat][0], conf_int_df.loc[feat][1]
                    except: ci_lower, ci_upper = float('nan'), float('nan')
                    try: elasticity = coef * (np.mean(X_test[:, feature_names.index(feat)]) / y_mean)
                    except: elasticity = float('nan')
                    
                    summary_lines.append(f" {feat[:35]:<35} | {coef:>30.6f} | {elasticity:>20.6f} | {tstat:>10.3f} | {pval:>10.4f}")
                    data_rows.append({"Variable": feat, "Coefficient": coef, "Elasticity": elasticity, "T-Stat": tstat, "P-Value": pval, "CI_Lower": ci_lower, "CI_Upper": ci_upper})
            else:
                summary_lines.append("="*140)
                summary_lines.append(f"{'FEATURE IMPORTANCES (TREE-BASED / NEURAL NETWORKS - SORTED)':^140}")
                summary_lines.append("="*140)
                summary_lines.append(f" {'Variable':<45} | {'Relative Importance Magnitude':>30}")
                summary_lines.append("-" * 140)
                
                importances = []
                if model_name == 'Neural Network' and hasattr(model, 'coefs_'): importances = np.mean(np.abs(model.coefs_[0]), axis=1)
                elif hasattr(model, 'feature_importances_'): importances = model.feature_importances_
                
                importances = np.array(importances).flatten()
                for feat, val in sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True):
                    summary_lines.append(f" {feat[:45]:<45} | {float(val):>30.6f}")
                    data_rows.append({"Variable": feat, "Importance": val})

            summary_lines.append("\n" + "="*140); summary_lines.append(f"{'CODECARBON EMISSIONS TELEMETRY REPORT':^140}"); summary_lines.append("="*140)
            if metrics.get('Emissions', 0.0) > 0:
                summary_lines.append(f" Ratio R² / Carbon    : {metrics.get('R2_CO2_Ratio', 0):.2f}")
                summary_lines.append(f" Emissions (kgCO2eq)  : {metrics.get('Emissions', 0):.8f}")
                summary_lines.append(f" Energy Consumed (kWh): {metrics.get('Energy', 0):.8f}")
            else: summary_lines.append(" Emissions tracking logger sheet was empty for this model.")
        except Exception: summary_lines.append(traceback.format_exc())

        text_area.insert("1.0", "\n".join(summary_lines)); text_area.configure(state="disabled")

        formula_text = ctk.CTkTextbox(tab_formula, font=ctk.CTkFont(family="Consolas", size=18), wrap="word"); formula_text.pack(expand=True, fill='both', pady=10)
        eq = "Y = \n"
        
        # Ce snippet pour la formule vient du code précédent (bien que partiellement tronqué visuellement dans la source d'origine)
        try:
            if val_key == "Coefficient":
                for i, row in enumerate(data_rows):
                    if i > 0: eq += " + \n"
                    eq += f"({row['Coefficient']:.6f} * {row['Variable']})"
        except:
            eq += "[Formula not available for this algorithm]"
            
        formula_text.insert("1.0", eq); formula_text.configure(state="disabled")

        # --- GRAPHICS ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#2b2b2b'); fig.tight_layout(pad=4.0)

        axes[0].scatter(y_test, y_test_pred, color='#1f6aa5', alpha=0.6, edgecolors='white', linewidth=0.5)
        axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#ef4444', linestyle='--', linewidth=2, label='Perfect Fit')
        
        std_resid = np.std(y_test - y_test_pred); x_line = np.linspace(min(y_test), max(y_test), 100)
        axes[0].fill_between(x_line, x_line - 1.96 * std_resid, x_line + 1.96 * std_resid, color='#22c55e', alpha=0.15, label='95% CI Band')
        axes[0].set_title('Actual vs Predicted (Test Set)', color='white'); axes[0].set_xlabel('Actual Values', color='white'); axes[0].set_ylabel('Predicted Values', color='white')
        axes[0].tick_params(colors='white'); axes[0].set_facecolor('#2b2b2b'); axes[0].legend(facecolor='#1a1a1a', edgecolor='none', labelcolor='white', loc='upper left', fontsize=8)

        residuals = y_test - y_test_pred
        axes[1].hist(residuals, bins=20, color='#ef4444', alpha=0.7, edgecolor='white')
        axes[1].set_title('Residuals Distribution', color='white'); axes[1].set_xlabel('Residual ', color='white'); axes[1].tick_params(colors='white'); axes[1].set_facecolor('#2b2b2b')
        p_val = metrics.get('Norm_P_Value', float('nan'))
        axes[1].text(0.95, 0.95, f"P-Value: {p_val:.4f}" if not np.isnan(p_val) else "P-Value: N/A", transform=axes[1].transAxes, color='white', fontsize=10, ha='right', va='top', bbox=dict(facecolor='#1a1a1a', alpha=0.8, edgecolor='none'))

        plot_data = [d for d in data_rows if d["Variable"] != "Intercept"]
        val_key = "Importance" if len(plot_data) > 0 and "Importance" in plot_data[0] else "Coefficient" if len(plot_data) > 0 else None
            
        if val_key:
            top_feats = sorted(plot_data, key=lambda x: abs(x[val_key]), reverse=True)[:10]
            y_labels = [(d["Variable"][:15] + "..") if len(d["Variable"]) > 15 else d["Variable"] for d in top_feats][::-1]
            x_vals = [d[val_key] for d in top_feats][::-1]
            
            if val_key == "Coefficient" and model_name in ['EcoRETINA', 'OLS']:
                x_err_lower = [d["Coefficient"] - d["CI_Lower"] for d in top_feats][::-1]
                x_err_upper = [d["CI_Upper"] - d["Coefficient"] for d in top_feats][::-1]
                axes[2].errorbar(x_vals, y_labels, xerr=[x_err_lower, x_err_upper], fmt='o', color='#22c55e', ecolor='#ef4444', elinewidth=2, capsize=4)
                axes[2].axvline(x=0, color='white', linestyle=':', alpha=0.5)
                axes[2].set_title('Top 10 Coefficients (95% CI)', color='white')
            else:
                colors = ['#22c55e' if v >= 0 else '#ef4444' for v in x_vals]
                axes[2].barh(y_labels, x_vals, color=colors)
                axes[2].set_title(f"Top 10 Variables ({model_name})", color='white')
                
            axes[2].tick_params(colors='white'); axes[2].set_facecolor('#2b2b2b')
        else:
            axes[2].text(0.5, 0.5, "No Feature Data Available", color='white', ha='center', va='center'); axes[2].set_facecolor('#2b2b2b'); axes[2].axis('off')

        canvas = FigureCanvasTkAgg(fig, master=tab_graphs); canvas.draw(); canvas.get_tk_widget().pack(expand=True, fill='both')

    # ==========================================
    # TAB 3: COMPARISON METRICS TABLE BOARD
    # ==========================================
    def build_compare_tab(self):
        # --- 1. ZONE HAUTE : TABLEAU DE COMPARAISON ---
        self.table_container = ctk.CTkFrame(self.tab_compare, corner_radius=10)
        self.table_container.pack(expand=True, fill='both', padx=20, pady=(20, 10))
        
        columns = ("Run Name", "Algo", "Train R²", "Train MAPE", "Test R²", "Test Adj R²", "Test RMSE", "Test MAPE", "R²/CO2", "CO2 (kg)")
        tree_frame = ctk.CTkFrame(self.table_container, fg_color="transparent")
        tree_frame.pack(expand=True, fill='both', padx=20, pady=(20, 10))
        
        tree_scroll = ctk.CTkScrollbar(tree_frame, orientation="vertical")
        tree_scroll.pack(side="right", fill="y")
        
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', yscrollcommand=tree_scroll.set)
        tree_scroll.configure(command=self.tree.yview)
        
        for col in columns:
            self.tree.heading(col, text=col)
            width = 110 if col not in ["Run Name", "Algo"] else 140
            self.tree.column(col, width=width, anchor='center')
        
        self.tree.pack(side="left", expand=True, fill='both')

        # Zebra striping
        self.tree.tag_configure('oddrow', background="#2b2b2b")
        self.tree.tag_configure('evenrow', background="#353535")
        
        # Events
        if sys.platform == "darwin":
            self.tree.bind("<Button-2>", self.open_tree_popup); self.tree.bind("<Button-3>", self.open_tree_popup) ; self.tree.bind("<ButtonRelease-1>", self.fix_mac_left_click)
        else:
            self.tree.bind("<Button-3>", self.open_tree_popup) 
        self.tree.bind("<Delete>", self.delete_selected_tree_rows)
        self.tree.bind("<BackSpace>", self.delete_selected_tree_rows)
        
        btn_frame = ctk.CTkFrame(self.table_container, fg_color="transparent")
        btn_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Boutons
        ctk.CTkButton(btn_frame, text="Clear Table", font=self.f_text, fg_color="#ef4444", hover_color="#dc2626", command=self.clear_comparison_table).pack(side='left', padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Export CSV", font=self.f_text, command=self.export_compare, fg_color="#3b3b3b", hover_color="#4b4b4b").pack(side='right', padx=(10, 0))
        ctk.CTkButton(btn_frame, text="📋 Copy LaTeX", font=self.f_text, command=self.copy_latex_to_clipboard, fg_color="#1f6aa5", hover_color="#144870").pack(side='right', padx=5)
        ctk.CTkButton(btn_frame, text="📋 Copy Markdown", font=self.f_text, command=self.copy_markdown_to_clipboard, fg_color="#68B946", hover_color="#539438").pack(side='right', padx=5)

        # --- 2. ZONE BASSE : GRAPHIQUES DE COMPARAISON VISUELLE ---
        self.chart_container = ctk.CTkFrame(self.tab_compare, corner_radius=10, fg_color="#2b2b2b")
        self.chart_container.pack(expand=True, fill='both', padx=20, pady=(10, 20))
        
        # Initialisation du canvas Matplotlib
        self.compare_fig, self.compare_axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='#2b2b2b')
        self.compare_fig.tight_layout(pad=4.0)
        self.compare_canvas = FigureCanvasTkAgg(self.compare_fig, master=self.chart_container)
        self.compare_canvas.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)
        
        self.update_comparison_chart() # Dessin initial (vide)
        
    def update_comparison_chart(self):
        if not hasattr(self, 'compare_axes'): return
        
        self.compare_axes[0].clear()
        self.compare_axes[1].clear()
        self.compare_axes[0].set_facecolor('#2b2b2b')
        self.compare_axes[1].set_facecolor('#2b2b2b')
        
        runs = []
        r2_scores = []
        mape_scores = []
        
        # Extraction des données directement depuis le tableau
        for item in self.tree.get_children():
            vals = self.tree.item(item)['values']
            run_name = str(vals[0]).replace(" *", "")
            algo_name = str(vals[1])
            runs.append(f"{algo_name}\n({run_name.split('_')[-1]})") # Ex: XGBoost (142033)
            
            try: r2_scores.append(float(str(vals[4]).replace(" *", "")))
            except: r2_scores.append(0.0)
                
            try: mape_scores.append(float(str(vals[7]).replace(" *", "").replace("%", "")))
            except: mape_scores.append(0.0)
                
        if not runs:
            self.compare_axes[0].text(0.5, 0.5, "No models trained yet.", color='gray', ha='center', va='center')
            self.compare_axes[1].text(0.5, 0.5, "No models trained yet.", color='gray', ha='center', va='center')
            self.compare_canvas.draw()
            return
            
        # Graphique 1 : Test R² (Le plus grand = Vert)
        best_r2 = max(r2_scores)
        colors_r2 = ['#22c55e' if r == best_r2 else '#1f6aa5' for r in r2_scores]
        bars1 = self.compare_axes[0].bar(runs, r2_scores, color=colors_r2, edgecolor='white', linewidth=0.5)
        self.compare_axes[0].set_title('Test R² Score (Higher is Better)', color='white', pad=15)
        self.compare_axes[0].tick_params(colors='white')
        
        # Graphique 2 : Test MAPE (Le plus petit = Vert)
        valid_mape = [m for m in mape_scores if m > 0]
        min_mape = min(valid_mape) if valid_mape else -1
        colors_mape = ['#22c55e' if m == min_mape and m > 0 else '#ef4444' for m in mape_scores]
        bars2 = self.compare_axes[1].bar(runs, mape_scores, color=colors_mape, edgecolor='white', linewidth=0.5)
        self.compare_axes[1].set_title('Test MAPE % (Lower is Better)', color='white', pad=15)
        self.compare_axes[1].tick_params(colors='white')
        
        # Rotation des labels pour éviter qu'ils ne se superposent si on a beaucoup de runs
        for ax in self.compare_axes:
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels(runs, rotation=25, ha='right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#555555')
            ax.spines['bottom'].set_color('#555555')
        
        self.compare_fig.tight_layout()
        self.compare_canvas.draw()
        
    def open_tree_popup(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item); self.tree.focus(item)
            popup_menu = tk.Menu(self.root, tearoff=0, bg="#1a1a1a", fg="white", activebackground="#1f6aa5", activeforeground="white", font=("Segoe UI", 14))
            popup_menu.add_command(label="View Statistical Report", command=lambda: self.view_summary_from_tree(item))
            
            # --- NOUVEAU : Bouton de suppression dans le menu ---
            popup_menu.add_separator()
            popup_menu.add_command(label="Delete Row", command=self.delete_selected_tree_rows)
            # ----------------------------------------------------
            
            popup_menu.bind("<FocusOut>", lambda e: popup_menu.unpost())
            self.root.bind("<Button-1>", lambda e: popup_menu.unpost(), add="+")
            try: popup_menu.tk_popup(event.x_root, event.y_root)
            finally: popup_menu.grab_release()

    def fix_mac_left_click(self, event):
        item = self.tree.identify_row(event.y)
        if item: self.tree.selection_set(item); self.tree.focus(item); self.tree.grab_release()

    def view_summary_from_tree(self, item):
        row_values = self.tree.item(item)['values']
        if row_values:
            run_name = str(row_values[0]).replace(" *", "")
            self.show_summary_window(run_name)
            
    def delete_selected_tree_rows(self, event=None):
        selected_items = self.tree.selection()
        if not selected_items: 
            return
        
        if messagebox.askyesno("Delete Run", "Are you sure you want to delete the selected run(s)?"):
            for item in selected_items:
                row_values = self.tree.item(item)['values']
                if row_values:
                    # On nettoie le nom de la run (pour enlever l'étoile verte s'il y en a une)
                    run_name = str(row_values[0]).replace(" *", "")
                    
                    # 1. On la supprime de l'historique mémoire
                    if run_name in self.run_history:
                        del self.run_history[run_name]
                        
                    # 2. On la supprime du tableau visuel
                    self.tree.delete(item)
                    self.log_event(f"Run '{run_name}' deleted from comparison board.")
            
            # 3. On recalcule les meilleurs scores (car le meilleur a peut-être été supprimé !)
            self.highlight_best_metrics()
            
            # 4. On met à jour la liste déroulante dans l'onglet "Predict"
            self.refresh_predict_models()

    def highlight_best_metrics(self):
        if not self.tree_tags_configured:
            for i in range(2, 10): self.tree.tag_configure(f'green_col_{i}', foreground='#22c55e')
            self.tree_tags_configured = True
        col_optimals = {2: True, 3: False, 4: True, 5: True, 6: False, 7: False, 8: True, 9: False}
        all_items = self.tree.get_children()
        if not all_items: return

        for item in all_items:
            vals = list(self.tree.item(item)['values'])
            for i in range(len(vals)):
                if isinstance(vals[i], str) and " *" in vals[i]: vals[i] = vals[i].replace(" *", "")
            self.tree.item(item, values=vals)

        for col_idx, is_max in col_optimals.items():
            best_val = -float('inf') if is_max else float('inf')
            best_items = []
            for item in all_items:
                val_str = str(self.tree.item(item)['values'][col_idx]).replace(" *", "").replace("%", "").replace("s", "")
                if val_str in ["N/A", "nan"]: continue
                try:
                    val = float(val_str)
                    if is_max and val > best_val: best_val = val; best_items = [item]
                    elif not is_max and val < best_val: best_val = val; best_items = [item]
                    elif val == best_val: best_items.append(item)
                except ValueError: pass
                
            for b_item in best_items:
                vals = list(self.tree.item(b_item)['values'])
                vals[col_idx] = str(vals[col_idx]) + " *"
                self.tree.item(b_item, values=vals)
        self.update_comparison_chart()
        
    def clear_comparison_table(self):
        if messagebox.askyesno("Clear Board", "Are you sure you want to delete ALL comparison data?"):
            self.tree.delete(*self.tree.get_children())
            self.run_history.clear()
            self.update_comparison_chart() # Met à jour le graphique (qui deviendra vide)
            self.refresh_predict_models()
            self.log_event("Comparison board completely cleared.")

    def add_comparison_result(self, run_name, algo, metrics, features_count):
        co2_str = f"{metrics.get('Emissions', 0.0):.2e}" if metrics.get('Emissions', 0.0) > 0 else "N/A"
        ratio_str = f"{metrics.get('R2_CO2_Ratio', 0.0):.2f}" if metrics.get('Emissions', 0.0) > 0 else "N/A"
        
        # Déterminer si la ligne est paire ou impaire
        row_count = len(self.tree.get_children())
        row_tag = 'evenrow' if row_count % 2 == 0 else 'oddrow'

        self.tree.insert("", tk.END, values=(
            run_name, algo, f"{metrics['R2_Train']:.4f}", f"{metrics['MAPE_Train']:.2f}%", 
            f"{metrics['R2_Test']:.4f}", f"{metrics['Adj_R2_Test']:.4f}", f"{metrics['RMSE_Test']:.4f}", 
            f"{metrics['MAPE_Test']:.2f}%", ratio_str, co2_str
        ), tags=(row_tag,))
        self.highlight_best_metrics()

    def export_compare(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")], title="Save Run Logs Metrics as CSV")
        if not filepath: return
        try:
            with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([self.tree.heading(col)["text"] for col in self.tree["columns"]])
                for item in self.tree.get_children():
                    clean_row = [str(v).replace(" *", "") for v in self.tree.item(item)["values"]]
                    writer.writerow(clean_row)
            self.log_event(f"Successfully exported comparison matrix to: {filepath}")
            messagebox.showinfo("Export Success", f"Historical benchmark comparison matrix sheet saved to:\n{filepath}")
        except Exception as e:
            self.log_event(f"Error exporting comparison matrix: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export comparison matrix:\n{str(e)}")
    
    def copy_latex_to_clipboard(self):
        if not self.tree.get_children():
            messagebox.showwarning("Table Empty", "There is no data to copy.")
            return

        columns = [self.tree.heading(col)["text"] for col in self.tree["columns"]]
        # Configuration standard pour papier de recherche (nécessite \usepackage{booktabs})
        latex_str = "\\begin{table}[htbp]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{" + "l" + "c"*(len(columns)-1) + "}\n\\toprule\n"
        latex_str += " & ".join(columns) + " \\\\\n\\midrule\n"
        
        for item in self.tree.get_children():
            row_vals = []
            for val in self.tree.item(item)["values"]:
                val_str = str(val)
                # Conversion des meilleurs scores (*) en gras LaTeX
                if " *" in val_str:
                    clean_val = val_str.replace(" *", "")
                    row_vals.append(f"\\textbf{{{clean_val}}}")
                else:
                    row_vals.append(val_str)
            latex_str += " & ".join(row_vals) + " \\\\\n"
            
        latex_str += "\\bottomrule\n\\end{tabular}\n}\n\\caption{Machine Learning Models Benchmark Comparison}\n\\label{tab:model_comparison}\n\\end{table}"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(latex_str)
        self.log_event("Copied comparison table to clipboard as LaTeX.")
        messagebox.showinfo("Copied to Clipboard", "LaTeX code copied!\n\nPro Tip: Ensure you have \\usepackage{booktabs} and \\usepackage{graphicx} in your LaTeX preamble.")

    def copy_markdown_to_clipboard(self):
        if not self.tree.get_children():
            messagebox.showwarning("Table Empty", "There is no data to copy.")
            return

        columns = [self.tree.heading(col)["text"] for col in self.tree["columns"]]
        md_str = "| " + " | ".join(columns) + " |\n"
        md_str += "|" + "|".join(["---"]*len(columns)) + "|\n"
        
        for item in self.tree.get_children():
            row_vals = []
            for val in self.tree.item(item)["values"]:
                val_str = str(val)
                # Conversion des meilleurs scores (*) en gras Markdown
                if " *" in val_str:
                    clean_val = val_str.replace(" *", "")
                    row_vals.append(f"**{clean_val}**")
                else:
                    row_vals.append(val_str)
            md_str += "| " + " | ".join(row_vals) + " |\n"
            
        self.root.clipboard_clear()
        self.root.clipboard_append(md_str)
        self.log_event("Copied comparison table to clipboard as Markdown.")
        messagebox.showinfo("Copied to Clipboard", "Markdown table copied!\nReady to paste in Notion, Obsidian, or Jupyter.")

def force_mac_focus(root):
    if sys.platform == "darwin":
        try:
            subprocess.call([
                "osascript", "-e",
                f'tell application "System Events" to set frontmost of first process whose unix id is {os.getpid()} to true'
            ])
        except Exception: pass
        root.lift()
        root.attributes('-topmost', True)
        root.after(150, lambda: root.attributes('-topmost', False))
        root.focus_force()

import multiprocessing # <-- TO BE ADDED

if __name__ == "__main__":
    # BLOCK UNWANTED WINDOW CREATION ON WINDOWS
    multiprocessing.freeze_support() # <-- CRITICAL: MUST BE THE VERY FIRST LINE HERE
    
    root = ctk.CTk()
    app = EcoRetinaApp(root)
    force_mac_focus(root)
    root.mainloop()