"""
סוכן טיפול באפאזיה בשיטת CIAT
Aphasia Therapy Agent using Constraint-Induced Aphasia Therapy (CIAT)

This agent simulates a speech-language therapist practicing the CIAT method
for word and sentence retrieval with an aphasia patient.

Usage:
    python aphasia_therapy/agent.py
    ANTHROPIC_API_KEY=... python aphasia_therapy/agent.py
"""

import os
import sys
import anthropic

# ─── CIAT Therapist System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """אתה קלינאי תקשורת מומחה המתמחה בטיפול בנפגעי אפאזיה בשיטת CIAT
(Constraint-Induced Aphasia Therapy - טיפול אפאזיה מבוסס אילוץ).

## עקרונות CIAT שאתה מיישם:

### 1. אילוץ (Constraint)
- אינך מקבל תקשורת שאינה מילולית (הצבעה, מחוות, כתיבה, ציור)
- אם המטופלת מנסה לתקשר שלא במילים, אמור בנועם: "נסי לומר את זה במילים"
- עודד ניסיונות מילוליים גם אם לא מושלמים

### 2. עיצוב התנהגות - Shaping (מעיצוב לביצוע מלא)
שלבי עיצוב לשליפת מילה:
1. הצלחה מלאה ← שבח וחזור הלאה
2. קושי → תן רמז סמנטי ("זה משהו שאוכלים", "זה בבית המטבח")
3. עדיין קשה → תן רמז פונולוגי ("זה מתחיל ב...")
4. עדיין קשה → תן את ההברה הראשונה
5. עדיין קשה → אמור את המילה ובקש חזרה (imitation)
6. תמיד חזרה על המילה המלאה בסוף

### 3. תרגול אינטנסיבי
- עבוד על מילים וביטויים חוזרים ונשנים
- עלה בהדרגה: מילה בודדת → ביטוי קצר → משפט פשוט → שיחה

### 4. חיזוק חיובי
- הגיבי בחמימות לכל ניסיון
- חגגי הצלחות: "כל הכבוד!", "יפה מאוד!", "מצוין!"
- לאחר טעות: "כמעט! בואי ננסה שוב"

## מבנה הסשן:

**שלב 1 - חימום (3-5 דקות)**
- שאלות פשוטות שגרתיות (שם, תחביב אחד)
- שיחה קצרה על היום

**שלב 2 - שליפת מילים (15-20 דקות)**
עבוד בקטגוריות:
- חפצים יומיומיים (כוס, כף, מכונית, טלפון, כיסא)
- פעולות (לאכול, לשתות, ללכת, לישון, לדבר)
- תארים (גדול, קטן, חם, קר, יפה)
- מזון ושתייה (לחם, מים, קפה, תפוח, עוגה)

**שלב 3 - שליפת משפטים (15-20 דקות)**
עבוד על מבנים:
- "אני רוצה ___"
- "___ הוא/היא ___"
- "אני [פועל] ___"
- "תן לי ___"

**שלב 4 - שיחה תפקודית (10 דקות)**
- הזמן מסעדה / חנות / בית קפה
- בקש עזרה במשהו
- תאר תמונה

## הנחיות חשובות:
- דבר בעברית תמיד
- היה סבלני ומעודד
- אל תמהר - תן זמן לשליפה
- רשום את ההתקדמות לאורך הסשן
- התאם את הקושי לרמת המטופלת
- אם המטופלת מתוסכלת - הפחת קושי ועבור למשהו קל יותר
- בסוף כל שלב - סכם מה הצליח

## הפעל את הסשן:
פתח כל סשן בהצגה עצמית חמה ושאלות פשוטות.
עקוב אחרי ההתקדמות וציין מה השתפר.
"""

# ─── Difficulty levels ────────────────────────────────────────────────────────

DIFFICULTY_LEVELS = {
    1: "מילים בודדות נפוצות",
    2: "מילים בודדות פחות נפוצות + ביטויים קצרים",
    3: "משפטים פשוטים (נושא + פועל)",
    4: "משפטים עם פרטים",
    5: "שיחה חופשית",
}

# ─── Conversation Manager ─────────────────────────────────────────────────────


class AphasiaTherapySession:
    """Manages a CIAT therapy session for an aphasia patient."""

    def __init__(self, patient_name: str = ""):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.messages: list[dict] = []
        self.patient_name = patient_name
        self.turn_count = 0
        self.difficulty = 1
        self.session_notes: list[str] = []

    def _build_context_message(self) -> str:
        """Build context about current session state for the model."""
        context = f"\n[מצב הסשן: פנייה {self.turn_count + 1}, רמת קושי {self.difficulty}/5 - {DIFFICULTY_LEVELS[self.difficulty]}]"
        if self.patient_name:
            context += f"\n[שם המטופלת: {self.patient_name}]"
        return context

    def send_message(self, user_input: str) -> str:
        """Send a patient message and get therapist response."""
        self.turn_count += 1

        # Add context to first message only (avoids cache invalidation)
        if self.turn_count == 1:
            content = self._build_context_message() + "\n\n" + user_input
        else:
            content = user_input

        self.messages.append({"role": "user", "content": content})

        # Use streaming to handle long responses without timeout
        with self.client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            messages=self.messages,
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text

        print()  # newline after streamed response

        self.messages.append({"role": "assistant", "content": response_text})

        # Auto-adjust difficulty based on turn count
        self._maybe_adjust_difficulty(response_text)

        return response_text

    def _maybe_adjust_difficulty(self, therapist_response: str) -> None:
        """Increase difficulty every ~8 turns if session is going well."""
        if self.turn_count % 8 == 0 and self.difficulty < 5:
            # Check for positive signals in therapist response
            positive_signals = ["כל הכבוד", "מצוין", "יפה מאוד", "נהדר", "מעולה"]
            if any(sig in therapist_response for sig in positive_signals):
                self.difficulty = min(5, self.difficulty + 1)
                note = f"פנייה {self.turn_count}: עלייה לרמה {self.difficulty}"
                self.session_notes.append(note)

    def start_session(self) -> str:
        """Initiate the therapy session with an opening prompt."""
        opening = "התחל את הסשן - הציג את עצמך ופתח עם שאלות חימום פשוטות למטופלת."
        return self.send_message(opening)

    def print_session_summary(self) -> None:
        """Print a summary of the therapy session."""
        print("\n" + "═" * 60)
        print("📋 סיכום סשן טיפולי")
        print("═" * 60)
        print(f"מספר פניות: {self.turn_count}")
        print(f"רמת קושי סופית: {self.difficulty}/5 - {DIFFICULTY_LEVELS[self.difficulty]}")
        if self.session_notes:
            print("\nהתקדמות:")
            for note in self.session_notes:
                print(f"  • {note}")
        print("═" * 60)


# ─── CLI Interface ────────────────────────────────────────────────────────────


def print_header() -> None:
    print("\n" + "═" * 60)
    print("🗣️  סוכן טיפול בשפה - שיטת CIAT")
    print("   Constraint-Induced Aphasia Therapy")
    print("═" * 60)
    print("הקלד את תגובות המטופלת. הקלד 'סיום' לסיום הסשן.\n")


def get_patient_name() -> str:
    try:
        name = input("שם המטופלת (אופציונלי, לחץ Enter לדלג): ").strip()
        return name
    except (EOFError, KeyboardInterrupt):
        return ""


def run_interactive_session() -> None:
    """Run an interactive CLI therapy session."""
    print_header()
    patient_name = get_patient_name()
    print()

    session = AphasiaTherapySession(patient_name=patient_name)

    print("🩺 המטפל:")
    session.start_session()

    while True:
        try:
            print("\n👤 המטופלת: ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("סיום", "bye", "exit", "quit"):
            print("\n🩺 המטפל:")
            session.send_message("סיים את הסשן בחום, סכם מה הצליח ועודד להמשך.")
            break

        print("\n🩺 המטפל:")
        session.send_message(user_input)

    session.print_session_summary()


# ─── Programmatic API ─────────────────────────────────────────────────────────


def create_session(patient_name: str = "") -> AphasiaTherapySession:
    """Create and return a new therapy session (for programmatic use)."""
    return AphasiaTherapySession(patient_name=patient_name)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ שגיאה: ANTHROPIC_API_KEY לא מוגדר.")
        print("   הגדר: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    run_interactive_session()
