"""
ממשק גרפי לסוכן טיפול באפאזיה - שיטת CIAT
Gradio GUI for Aphasia Therapy Agent (CIAT)

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python aphasia_therapy/app.py
"""

import os
import sys
import anthropic
import gradio as gr
from aphasia_therapy.word_bank import WORD_BANK, DIFFICULTY_CONFIGS, format_word_bank_display

# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """אתה קלינאי תקשורת מומחה המתמחה בטיפול בנפגעי אפאזיה בשיטת CIAT
(Constraint-Induced Aphasia Therapy - טיפול אפאזיה מבוסס אילוץ).

## עקרונות CIAT:

### אילוץ (Constraint)
- אינך מקבל תקשורת שאינה מילולית
- אם המטופלת מנסה בדרך אחרת: "נסי לומר את זה במילים, את יכולה"
- כל ניסיון מילולי הוא הצלחה

### עיצוב (Shaping) - רמות רמז:
1. המטופלת מצליחה → "כל הכבוד! " + המשך
2. מתקשה → רמז סמנטי: "זה משהו שאוכלים/שותים/בבית..."
3. עדיין מתקשה → רמז פונולוגי: "זה מתחיל באות..."
4. עדיין מתקשה → הברה ראשונה: "מ... / ל... / ש..."
5. עדיין מתקשה → "בואי נגיד יחד: [מילה]" ← חיקוי
6. תמיד חזרה על המילה המלאה בסוף

### חיזוק חיובי
- "כל הכבוד!", "יפה מאוד!", "מצוין!", "נהדר!"
- אחרי קושי: "כמעט! עוד ניסיון, את יכולה"
- אחרי הצלחה: "מעולה! המילה היא [מילה]"

### התקדמות הדרגתית
- מילה בודדת → ביטוי → משפט פשוט → משפט מורכב → שיחה

## כללים:
- דבר בעברית תמיד
- היה סבלני - תן זמן לשליפה
- משפטים קצרים וברורים
- אל תתקן בצורה גסה - עדד ועצב
- אם המטופלת מתוסכלת - הפחת קושי מיד
- עבוד עם המילים שמוצגות בצד (מאגר המילים היומיומיות)
"""

# ─── Session State ─────────────────────────────────────────────────────────────

class TherapyState:
    def __init__(self):
        self.messages: list[dict] = []
        self.difficulty: int = 1
        self.turn_count: int = 0
        self.current_category: str = list(WORD_BANK.keys())[0]
        self.patient_name: str = ""
        self.successes: int = 0
        self.session_active: bool = False

    def reset(self):
        self.messages = []
        self.difficulty = 1
        self.turn_count = 0
        self.successes = 0
        self.session_active = False

# Global state (Gradio shares state via gr.State)
def make_state():
    return TherapyState()

# ─── Core Logic ────────────────────────────────────────────────────────────────

def get_therapist_response(state: TherapyState, user_text: str) -> tuple[str, TherapyState]:
    """Send patient input and get therapist response (streaming collected)."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    state.turn_count += 1

    # Inject word bank context into first user message
    if state.turn_count == 1:
        category_words = ", ".join(WORD_BANK[state.current_category]["words"][:12])
        context = (
            f"[רמת קושי: {state.difficulty}/5 - {DIFFICULTY_CONFIGS[state.difficulty]['label']}]\n"
            f"[קטגוריה נוכחית: {state.current_category}]\n"
            f"[מילים לתרגול: {category_words}]\n"
            f"[שם המטופלת: {state.patient_name or 'לא צוין'}]\n\n"
        )
        content = context + user_text
    else:
        content = user_text

    state.messages.append({"role": "user", "content": content})

    # Collect full response (streaming)
    response_text = ""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=state.messages,
    ) as stream:
        for text in stream.text_stream:
            response_text += text

    state.messages.append({"role": "assistant", "content": response_text})

    # Count successes from positive signals
    positive = ["כל הכבוד", "מצוין", "יפה מאוד", "נהדר", "מעולה", ""]
    if any(p in response_text for p in positive):
        state.successes += 1

    return response_text, state


def format_progress_bar(difficulty: int) -> str:
    filled = "🟢" * difficulty
    empty = "⚪" * (5 - difficulty)
    return filled + empty


def build_status_md(state: TherapyState) -> str:
    if not state.session_active:
        return "לחצי **התחל סשן** כדי להתחיל"
    bar = format_progress_bar(state.difficulty)
    level_info = DIFFICULTY_CONFIGS[state.difficulty]
    return (
        f"**רמת קושי:** {bar} ({state.difficulty}/5)\n\n"
        f"**שלב:** {level_info['label']}\n\n"
        f"**פניות:** {state.turn_count}  |  **הצלחות:** {state.successes}\n\n"
        f"**קטגוריה:** {state.current_category}"
    )

# ─── Gradio Callbacks ──────────────────────────────────────────────────────────

def start_session(patient_name: str, category: str, difficulty: int, state: TherapyState):
    """Initialize and start a new therapy session."""
    state.reset()
    state.patient_name = patient_name.strip()
    state.current_category = category
    state.difficulty = difficulty
    state.session_active = True

    opening = (
        "התחל את הסשן עכשיו. הציג את עצמך בחום, שאל שאלת חימום פשוטה אחת, "
        "ואז עבור לתרגול מילים מהקטגוריה שנבחרה."
    )
    response, state = get_therapist_response(state, opening)

    chat_history = [{"role": "assistant", "content": response}]
    word_display = format_word_bank_display(category)
    status = build_status_md(state)

    return chat_history, word_display, status, state


def patient_speaks(
    user_input: str,
    chat_history: list,
    state: TherapyState,
):
    """Handle patient message and get therapist reply."""
    if not state.session_active:
        return chat_history, build_status_md(state), state, ""

    if not user_input.strip():
        return chat_history, build_status_md(state), state, ""

    chat_history.append({"role": "user", "content": user_input})

    response, state = get_therapist_response(state, user_input)

    chat_history.append({"role": "assistant", "content": response})
    status = build_status_md(state)

    return chat_history, status, state, ""


def increase_difficulty(state: TherapyState, chat_history: list):
    """Manually increase difficulty level."""
    if not state.session_active:
        return chat_history, build_status_md(state), state
    if state.difficulty < 5:
        state.difficulty += 1
        msg = f"[רמת קושי עלתה ל-{state.difficulty}/5 - {DIFFICULTY_CONFIGS[state.difficulty]['label']}]"
        response, state = get_therapist_response(
            state,
            f"עלינו לרמת קושי {state.difficulty}. {DIFFICULTY_CONFIGS[state.difficulty]['prompt_suffix']}"
        )
        chat_history.append({"role": "assistant", "content": response})
    return chat_history, build_status_md(state), state


def decrease_difficulty(state: TherapyState, chat_history: list):
    """Manually decrease difficulty level."""
    if not state.session_active:
        return chat_history, build_status_md(state), state
    if state.difficulty > 1:
        state.difficulty -= 1
        response, state = get_therapist_response(
            state,
            f"ירדנו לרמת קושי {state.difficulty}. {DIFFICULTY_CONFIGS[state.difficulty]['prompt_suffix']} בואי נתחיל מחדש עם משהו קל יותר."
        )
        chat_history.append({"role": "assistant", "content": response})
    return chat_history, build_status_md(state), state


def change_category(new_category: str, state: TherapyState, chat_history: list):
    """Switch practice word category mid-session."""
    if not state.session_active:
        word_display = format_word_bank_display(new_category)
        return chat_history, word_display, build_status_md(state), state

    state.current_category = new_category
    category_words = ", ".join(WORD_BANK[new_category]["words"][:10])
    response, state = get_therapist_response(
        state,
        f"עכשיו נעבור לקטגוריה: {new_category}. מילים לתרגול: {category_words}. התחילי עם המילה הראשונה."
    )
    chat_history.append({"role": "assistant", "content": response})
    word_display = format_word_bank_display(new_category)
    return chat_history, word_display, build_status_md(state), state


def end_session(state: TherapyState, chat_history: list):
    """End the session with a summary."""
    if not state.session_active:
        return chat_history, build_status_md(state), state

    response, state = get_therapist_response(
        state,
        f"סיים את הסשן. סכם בחום מה הצליח, ציין {state.successes} הצלחות שנרשמו, ועודד להמשך."
    )
    chat_history.append({"role": "assistant", "content": response})
    state.session_active = False
    return chat_history, build_status_md(state), state


# ─── UI Layout ─────────────────────────────────────────────────────────────────

CSS = """
.chat-container { font-size: 18px !important; }
.message { border-radius: 12px !important; }
.therapist-msg { background: #e8f5e9 !important; }
.patient-msg { background: #e3f2fd !important; }
#status-box { background: #f5f5f5; padding: 12px; border-radius: 8px; }
#word-box { background: #fffde7; padding: 12px; border-radius: 8px; }
.difficulty-btn { font-size: 20px !important; }
footer { display: none !important; }
"""

def build_ui():
    categories = list(WORD_BANK.keys())

    with gr.Blocks(
        title="טיפול באפאזיה - CIAT",
        theme=gr.themes.Soft(
            primary_hue="green",
            font=[gr.themes.GoogleFont("Assistant"), "Arial"],
        ),
        css=CSS,
        rtl=True,
    ) as demo:

        state = gr.State(make_state)

        # ── Header ──
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px; direction: rtl;">
            <h1 style="font-size: 28px; color: #2e7d32;">🗣️ סוכן טיפול בשפה - שיטת CIAT</h1>
            <p style="color: #555; font-size: 16px;">
                Constraint-Induced Aphasia Therapy · שליפת מילים ומשפטים מהחיים היומיומיים
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left Panel: Chat ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="שיחה טיפולית",
                    type="messages",
                    height=480,
                    show_label=True,
                    avatar_images=(None, "🩺"),
                    elem_classes=["chat-container"],
                    rtl=True,
                )

                with gr.Row():
                    patient_input = gr.Textbox(
                        placeholder="הקלידי את התגובה כאן...",
                        show_label=False,
                        scale=4,
                        rtl=True,
                        lines=2,
                    )
                    send_btn = gr.Button("שלח ↵", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    harder_btn = gr.Button("⬆️ קשה יותר", variant="secondary", size="sm")
                    easier_btn = gr.Button("⬇️ קל יותר", variant="secondary", size="sm")
                    end_btn = gr.Button("סיים סשן 🏁", variant="stop", size="sm")

            # ── Right Panel: Settings + Word Bank ──
            with gr.Column(scale=2):

                # Session setup
                with gr.Group():
                    gr.Markdown("### ⚙️ הגדרות סשן")
                    patient_name_in = gr.Textbox(
                        label="שם המטופלת",
                        placeholder="לדוגמה: מרים",
                        rtl=True,
                    )
                    category_in = gr.Dropdown(
                        choices=categories,
                        value=categories[0],
                        label="קטגוריה לתרגול",
                    )
                    difficulty_in = gr.Slider(
                        minimum=1, maximum=5, step=1, value=1,
                        label="רמת קושי התחלתית",
                    )
                    start_btn = gr.Button("▶️ התחל סשן", variant="primary", size="lg")

                # Status display
                gr.Markdown("### 📊 מצב הסשן")
                status_md = gr.Markdown(
                    value="לחצי **התחל סשן** כדי להתחיל",
                    elem_id="status-box",
                )

                # Word bank display
                gr.Markdown("### 📚 מאגר מילים")
                word_bank_md = gr.Markdown(
                    value=format_word_bank_display(categories[0]),
                    elem_id="word-box",
                )

                # Change category mid-session
                change_cat_btn = gr.Button("🔄 החלף קטגוריה", size="sm")

        # ── Difficulty reference ──
        with gr.Accordion("📖 מדריך רמות קושי", open=False):
            rows = []
            for lvl, cfg in DIFFICULTY_CONFIGS.items():
                rows.append(f"**רמה {lvl} - {cfg['label']}:** {cfg['description']} | דוגמה: _{cfg['example']}_")
            gr.Markdown("\n\n".join(rows))

        # ── Event wiring ──────────────────────────────────────────────────────

        start_btn.click(
            fn=start_session,
            inputs=[patient_name_in, category_in, difficulty_in, state],
            outputs=[chatbot, word_bank_md, status_md, state],
        )

        send_btn.click(
            fn=patient_speaks,
            inputs=[patient_input, chatbot, state],
            outputs=[chatbot, status_md, state, patient_input],
        )

        patient_input.submit(
            fn=patient_speaks,
            inputs=[patient_input, chatbot, state],
            outputs=[chatbot, status_md, state, patient_input],
        )

        harder_btn.click(
            fn=increase_difficulty,
            inputs=[state, chatbot],
            outputs=[chatbot, status_md, state],
        )

        easier_btn.click(
            fn=decrease_difficulty,
            inputs=[state, chatbot],
            outputs=[chatbot, status_md, state],
        )

        change_cat_btn.click(
            fn=change_category,
            inputs=[category_in, state, chatbot],
            outputs=[chatbot, word_bank_md, status_md, state],
        )

        category_in.change(
            fn=lambda cat: format_word_bank_display(cat),
            inputs=[category_in],
            outputs=[word_bank_md],
        )

        end_btn.click(
            fn=end_session,
            inputs=[state, chatbot],
            outputs=[chatbot, status_md, state],
        )

    return demo


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ שגיאה: ANTHROPIC_API_KEY לא מוגדר.")
        print("   הגדר: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
