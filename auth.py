import sqlite3
import hashlib
import streamlit as st

def init_db():
    conn = sqlite3.connect('factory_users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def add_userdata(username, password):
    conn = sqlite3.connect('factory_users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username,password) VALUES (?,?)', (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('factory_users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def login_page():
    init_db()
    st.markdown('<h1 class="glow-text" style="text-align:center;">FactoryMind AI</h1>', unsafe_allow_html=True)
    
    cols = st.columns([1, 1.5, 1])
    with cols[1]:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        choice = st.radio("Access Control", ["Login", "Sign Up"], horizontal=True)
        
        if choice == "Login":
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.button("Enter Dashboard"):
                if login_user(user, make_hashes(pw)):
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = user
                    st.rerun()
                else:
                    st.error("Invalid credentials")
                    
        else:
            new_user = st.text_input("Create Username")
            new_pw = st.text_input("Create Password", type="password")
            if st.button("Register Account"):
                if add_userdata(new_user, make_hashes(new_pw)):
                    st.success("Account created! Please switch to Login.")
                else:
                    st.error("Username already taken.")
        st.markdown('</div>', unsafe_allow_html=True)