import time
from pathlib import Path
import base64
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from assistant.ingest import ingest_corpus
from assistant.qa import answer_question

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "unt_dast_logo.png"

DOCS_DIR = APP_DIR / "data" / "docs"
PDF_MAP = {
    "Academic Integrity Policy": DOCS_DIR / "Academic_Integrity.pdf",
    "MS ADTA Handbook": DOCS_DIR / "ADTA Graduate School Handbook Academic Year 25-26.pdf",
}

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="DAST FAQ ASSISTANT (RAG)",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------
# Helpers
# ---------------------------
def format_sources(docs):
    seen = set()
    out = []
    for doc in docs or []:
        src = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        label = f"{src} — page {page}" if page is not None else src
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


@st.cache_data(show_spinner=False)
def get_logo_b64(path: Path) -> str:
    img = Image.open(path).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pdf_preview(pdf_path: Path, height: int = 900):
    """Inline PDF preview using a base64 iframe (no extra libraries)."""
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%"
            height="{height}"
            style="border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 12px;"
        ></iframe>
        """,
        unsafe_allow_html=True,
    )


def doc_downloads_in_left_rail():
    st.markdown("**Documents**")
    for label, path in PDF_MAP.items():
        if path.exists():
            with open(path, "rb") as f:
                st.download_button(
                    label=f"⬇️ {label}",
                    data=f,
                    file_name=path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.warning(f"Missing: {path.name}")


# ---------------------------
# Vector store caching
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_vector_store():
    return ingest_corpus()


# ---------------------------
# CSS (sticky input + clean layout)
# ---------------------------
THEME_CSS = """
<style>
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] > section > div:first-child { padding-top: 0 !important; }

.card {
  background: #ffffff;
  border-radius: 18px;
  border: 1px solid rgba(49, 51, 63, 0.12);
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
  padding: 18px 18px;
}
.muted { opacity: 0.75; }

.item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin: 12px 0;
}
.blue-banner {
  background: #0B63B6;
  color: white;
  border-radius: 14px;
  padding: 14px 16px;
  font-weight: 650;
  margin: 14px 0 16px 0;
}

/* ✅ STICKY CHAT INPUT (TOP) */
div[data-testid="stChatInput"] {
  position: sticky;
  top: 110px;             /* adjust if needed */
  z-index: 1000;
  background: #ffffff;
  padding-top: 10px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(49, 51, 63, 0.10);
}

/* input styling */
div[data-testid="stChatInput"] textarea { border-radius: 18px; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------
# Header
# ---------------------------
def top_header():
    if not LOGO_PATH.exists():
        st.error(f"Logo not found: {LOGO_PATH}")
        return

    logo_b64 = get_logo_b64(LOGO_PATH)

    st.markdown(
        f"""
        <style>
          .custom-header-wrapper {{
              position: relative;
              left: 50%;
              right: 50%;
              margin-left: -50vw;
              margin-right: -50vw;
              width: 100vw;
              background: #0B4F3A;
              padding: 12px 0;
              box-sizing: border-box;
          }}
          .custom-header-inner {{
              max-width: 1400px;
              margin: 0 auto;
              padding: 0 60px;
              box-sizing: border-box;
              position: relative;
              height: 70px;
          }}
          .header-logo {{
              position: absolute;
              left: 60px;
              top: 50%;
              transform: translateY(-50%);
              display: flex;
              align-items: center;
              padding: 6px 10px;
              border-radius: 10px;
              background: rgba(255,255,255,0.06);
              border: 1px solid rgba(255,255,255,0.20);
          }}
          .header-logo img {{
              height: 44px;
              width: auto;
              display: block;
          }}
          .header-title {{
              position: absolute;
              left: 50%;
              top: 50%;
              transform: translate(-50%, -50%);
              color: #ffffff;
              font-size: 32px;
              font-weight: 800;
              font-family: "Arial Narrow", "Helvetica Neue Condensed", Arial, sans-serif;
              letter-spacing: 0.8px;
              white-space: nowrap;
              text-align: center;
          }}
        </style>

        <div class="custom-header-wrapper">
          <div class="custom-header-inner">
            <div class="header-logo">
              <img src="data:image/png;base64,{logo_b64}" alt="UNT DAST Logo" />
            </div>
            <div class="header-title"> AskDAST FAQ Assistant</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Left rail
# ---------------------------
def left_rail():
    st.markdown(
        """
        <div class="card">
          <div style="font-weight:900; font-size:20px; font-family: Arial, sans-serif; color:#114B3A;">Scope</div>
          <div class="item">
            <div>Answers questions using only approved department documents with citations</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("⟳ Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.prefill = ""
        st.rerun()

    st.markdown("")
    st.markdown("**Sample Question**")
    example = "What are the mandatory courses for ADTA Grad Program?"
    if st.button(example, use_container_width=True):
        st.session_state.prefill = example
        st.rerun()

    st.markdown("")
    doc_downloads_in_left_rail()


# ---------------------------
# Main intro
# ---------------------------
def main_intro_card():
    st.markdown(
        """
        <div class="card">
          <div class="muted" style="font-size:18px; line-height:1.55;">
            A Retrieval-Augmented Generation (RAG) assistant grounded in department-approved documents.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# App
# ---------------------------
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prefill" not in st.session_state:
        st.session_state.prefill = ""

    top_header()

    st.markdown('<div style="padding: 1.5rem 3rem 1rem 3rem;">', unsafe_allow_html=True)

    left, right = st.columns([0.33, 0.67], gap="large")

    with left:
        left_rail()

    with right:
        tabs = st.tabs(["Chat", "Document Preview"])

        # ------------------ CHAT TAB ------------------
        with tabs[0]:
            main_intro_card()

            with st.spinner("Preparing knowledge base..."):
                t0 = time.time()
                vector_store = load_vector_store()
                load_s = time.time() - t0

            st.success(f"Knowledge base ready (loaded in {load_s:.2f}s).")

            st.markdown(
                """
                <div class="blue-banner">
                  Ask your question about MS ADTA program or Academic Integrity policy.
                  You will get a grounded answer plus citations.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ✅ Anchor + auto-scroll so it doesn't "go down" after rerun
            st.markdown('<div id="chat-input-anchor"></div>', unsafe_allow_html=True)
            components.html(
                """
                <script>
                  const el = window.parent.document.getElementById("chat-input-anchor");
                  if (el) { el.scrollIntoView({behavior: "instant", block: "start"}); }
                </script>
                """,
                height=0,
            )

            # ✅ Prompt box stays at top (sticky via CSS)
            user_input = st.chat_input("Type your question…")

            # If user asked something, answer it and append
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})

                result = answer_question(vector_store, user_input)
                answer = (result.get("answer") or "").strip()
                sources = format_sources(result.get("sources", []))

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

                st.rerun()

            # ✅ Replies appear below input
            for msg in reversed(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander(f"Sources ({len(msg['sources'])})"):
                            for s in msg["sources"]:
                                st.write(f"- {s}")

        # ------------------ PREVIEW TAB ------------------
        with tabs[1]:
            st.markdown("### Document Preview")

            selected_doc = st.selectbox(
                "Select a document to preview",
                list(PDF_MAP.keys()),
            )

            pdf_path = PDF_MAP[selected_doc]

            if not pdf_path.exists():
                st.error(f"PDF not found: {pdf_path}")
            else:
                pdf_preview(pdf_path, height=900)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
