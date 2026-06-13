"""
Placement Prediction System — Production-Grade Streamlit Application
Author: Senior ML Engineer / Data Scientist
Description: AI-powered student employability assessment tool using machine learning.
"""

import os
import streamlit as st

# ─── Page Configuration (Must be the first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load and Inject Custom CSS ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def load_css(filename: str) -> None:
    """Inject an external CSS file (resolved next to app.py) into Streamlit."""
    filepath = os.path.join(_HERE, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at: {filepath}")


load_css("style.css")

# ─── Import Prediction Utility ────────────────────────────────────────────────
try:
    from utils import predict_placement
    MODEL_AVAILABLE = True
except (ImportError, AttributeError):
    MODEL_AVAILABLE = False

    def predict_placement(cgpa, internships, projects, workshops,
                          aptitude_score, soft_skills, extracurricular,
                          placement_training, ssc_marks, hsc_marks):
        """Fallback rule-based heuristic function if model utils are missing."""
        score = (
            cgpa * 0.25 +
            internships * 2.0 +
            projects * 1.5 +
            workshops * 1.0 +
            aptitude_score * 0.2 +
            soft_skills * 3.0 +
            extracurricular * 5.0 +         # Already encoded as 0 or 1
            placement_training * 5.0 +      # Already encoded as 0 or 1
            ssc_marks * 0.05 +
            hsc_marks * 0.05
        )
        prob = min(max(score / 60.0, 0.05), 0.99)
        prediction = 1 if prob > 0.5 else 0
        return prediction, prob


# ══════════════════════════════════════════════════════════════════════════════
#  HERO SECTION (FIXED STRUCTURAL ALIGNMENT)
# ══════════════════════════════════════════════════════════════════════════════
def render_hero() -> None:
    """Render the full-width hero banner at the top of the page."""
    st.markdown("""
    <div class="hero-section" style="text-align: center; display: flex; flex-direction: column; align-items: center;">
        <div class="hero-badge">✦ AI-Powered Assessment</div>
        <h1 class="hero-title" style="text-align: center;">
            🎓 Placement Prediction<br>
            <span class="hero-title-accent">System</span>
        </h1>
        <p class="hero-subtitle" style="text-align: center;">
            ML-Powered Student Employability Assessment
        </p>
        <p class="hero-description" style="text-align: center; margin: 0 auto 44px; max-width: 600px; display: block;">
            Don’t guess your recruitment readiness—measure it. Benchmark your profile against real corporate hiring standards,
            identify hidden risks early, and unlock personalized action items to optimize your career path.
        </p>
        <div class="hero-stats" style="display: flex; justify-content: center; align-items: center;">
            <div class="hero-stat">
                <span class="hero-stat-number">10+</span>
                <span class="hero-stat-label">Assessment Parameters</span>
            </div>
            <div class="hero-divider"></div>
            <div class="hero-stat">
                <span class="hero-stat-number">ML</span>
                <span class="hero-stat-label">Powered Engine</span>
            </div>
            <div class="hero-divider"></div>
            <div class="hero-stat">
                <span class="hero-stat-number">Real-time</span>
                <span class="hero-stat-label">Career Insights</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
def render_inputs() -> dict:
    """
    Render the student profile input form.
    Returns a dict of all collected feature values.
    """
    st.markdown(
        '<div class="section-header"><span class="section-tag">PROFILE</span>'
        '<h2>Student Assessment Profile</h2>'
        '<p>Complete all fields for the most accurate placement prediction.</p></div>', 
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    # ── Column 1: Academic & Cognitive ────────────────────────────────────────
    with col1:
        st.markdown('<div class="input-card"><div class="input-card-header">📚 Academic & Cognitive Metrics</div>', unsafe_allow_html=True)
        
        cgpa = st.slider(
            "CGPA (Cumulative GPA)",
            min_value=0.0, max_value=10.0, value=7.5, step=0.1,
            help="Your cumulative GPA on a 10-point scale."
        )
        internships = st.number_input(
            "Internships Completed",
            min_value=0, max_value=10, value=1, step=1,
            help="Number of internships you have completed."
        )
        projects = st.number_input(
            "Academic / Personal Projects",
            min_value=0, max_value=20, value=2, step=1,
            help="Total number of significant projects."
        )
        workshops = st.number_input(
            "Workshops / Certifications",
            min_value=0, max_value=30, value=3, step=1,
            help="Number of workshops or certifications attended."
        )
        aptitude_score = st.slider(
            "Aptitude Test Score (%)",
            min_value=0, max_value=100, value=70, step=1,
            help="Score in aptitude / reasoning tests."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Column 2: Skills & Engagement ─────────────────────────────────────────
    with col2:
        st.markdown('<div class="input-card"><div class="input-card-header">🌟 Skills & Engagement</div>', unsafe_allow_html=True)
        
        soft_skills = st.slider(
            "Soft Skills Rating",
            min_value=1.0, max_value=5.0, value=3.5, step=0.1,
            help="Self-assessed soft skills on a 1–5 scale (communication, teamwork, etc.)."
        )
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=["Yes", "No"],
            help="Have you participated in extracurricular activities?"
        )
        placement_training = st.selectbox(
            "Attended Placement Training",
            options=["Yes", "No"],
            help="Did you attend placement or career training programs?"
        )
        ssc_marks = st.slider(
            "SSC / 10th Grade Marks (%)",
            min_value=0, max_value=100, value=75, step=1,
            help="Marks obtained in 10th grade board exams."
        )
        hsc_marks = st.slider(
            "HSC / 12th Grade Marks (%)",
            min_value=0, max_value=100, value=72, step=1,
            help="Marks obtained in 12th grade board exams."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    return {
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "workshops": workshops,
        "aptitude_score": aptitude_score,
        "soft_skills": soft_skills,
        "extracurricular": extracurricular,
        "placement_training": placement_training,
        "ssc_marks": ssc_marks,
        "hsc_marks": hsc_marks,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION CTA BUTTON
# ══════════════════════════════════════════════════════════════════════════════
def render_predict_button() -> bool:
    """Render the full-width CTA prediction button. Returns True when clicked."""
    st.markdown('<div class="cta-wrapper">', unsafe_allow_html=True)
    clicked = st.button("🚀  Predict Placement Probability", key="predict_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return clicked


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS: Labels & Analytics Bands
# ══════════════════════════════════════════════════════════════════════════════
def get_confidence_label(probability: float) -> tuple[str, str]:
    """Map probability to confidence level label and CSS class."""
    if probability >= 0.75:
        return "High", "confidence-high"
    elif probability >= 0.50:
        return "Medium", "confidence-medium"
    return "Low", "confidence-low"


def get_employability_band(score: float) -> str:
    """Return a descriptive band for the employability score."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    return "Needs Improvement"


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_results(inputs: dict, prediction: int, probability: float) -> None:
    """Render the full results dashboard metric cards and analytics grids."""
    pct = round(probability * 100, 2)
    confidence_label, confidence_class = get_confidence_label(probability)
    emp_band = get_employability_band(pct)

    # ── Verdict Banner ────────────────────────────────────────────────────────
    if prediction == 1:
        verdict_html = f"""
        <div class="verdict placed">
            <div class="verdict-icon">✅</div>
            <div class="verdict-text">
                <h2>Likely To Be Placed</h2>
                <p>Your profile indicates a strong readiness for campus recruitment.</p>
            </div>
            <div class="verdict-badge placed-badge">PLACED</div>
        </div>
        """
    else:
        verdict_html = f"""
        <div class="verdict at-risk">
            <div class="verdict-icon">❌</div>
            <div class="verdict-text">
                <h2>Placement Risk Detected</h2>
                <p>Your profile needs strengthening before campus recruitment.</p>
            </div>
            <div class="verdict-badge risk-badge">AT RISK</div>
        </div>
        """
    st.markdown(verdict_html, unsafe_allow_html=True)

    # ── KPI Cards Grid ────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3 = st.columns(3, gap="medium")
    with kpi1:
        st.markdown(f'<div class="kpi-card kpi-primary"><div class="kpi-label">Placement Probability</div>'
                    f'<div class="kpi-value">{pct}%</div><div class="kpi-sub">Based on ML model</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card kpi-secondary"><div class="kpi-label">Confidence Level</div>'
                    f'<div class="kpi-value {confidence_class}">{confidence_label}</div><div class="kpi-sub">Prediction reliability</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card kpi-accent"><div class="kpi-label">Employability Score</div>'
                    f'<div class="kpi-value">{pct}</div><div class="kpi-sub">{emp_band} readiness</div></div>', unsafe_allow_html=True)

    # ── Animated Probability Progress Bar ──────────────────────────────────────
    bar_color = "#10B981" if prediction == 1 else "#EF4444"
    st.markdown(f"""
    <div class="prob-bar-section">
        <div class="prob-bar-header">
            <span class="prob-bar-title">Placement Probability Tracking</span>
            <span class="prob-bar-pct" style="color:{bar_color}">{pct}%</span>
        </div>
        <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{int(pct)}%; background: linear-gradient(90deg, {bar_color}aa, {bar_color});"></div>
        </div>
        <div class="prob-bar-labels">
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Insights & Action Items Grid ──────────────────────────────────────────
    ins_col, rec_col = st.columns(2, gap="large")
    with ins_col:
        render_insights(inputs)
    with rec_col:
        render_recommendations(inputs, pct)


# ══════════════════════════════════════════════════════════════════════════════
#  CAREER INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def render_insights(inputs: dict) -> None:
    """Dynamically compile evaluation insights from student profile features."""
    insights = []

    if inputs["cgpa"] >= 8.5:
        insights.append(("🏆", "Outstanding Academic Performance", "Your CGPA places you in the top tier of candidates recruiters actively seek."))
    elif inputs["cgpa"] >= 7.5:
        insights.append(("📘", "Solid Academic Record", "Your academic performance is competitive and meets most recruiter benchmarks."))
    else:
        insights.append(("📖", "Academic Improvement Potential", "Pushing your CGPA higher could significantly improve your placement chances."))

    if inputs["aptitude_score"] >= 85:
        insights.append(("🧠", "Strong Analytical Ability", "Your aptitude score signals excellent problem-solving and quantitative skills."))
    elif inputs["aptitude_score"] >= 65:
        insights.append(("🔢", "Good Aptitude Foundation", "You have a solid aptitude base—targeted practice can push you to the top."))

    if inputs["soft_skills"] >= 4.5:
        insights.append(("🗣️", "Exceptional Communication Skills", "High soft skills rating is a major differentiator in competitive hiring."))
    elif inputs["soft_skills"] >= 3.5:
        insights.append(("💬", "Adequate Interpersonal Skills", "Your communication skills are good; refining them further will set you apart."))

    if inputs["internships"] >= 2:
        insights.append(("🏢", "Strong Industry Exposure", "Multiple internships demonstrate real-world experience that recruiters highly value."))
    elif inputs["internships"] == 1:
        insights.append(("🔍", "Industry Exposure Identified", "Your internship experience gives you a practical edge over campus-only candidates."))

    if inputs["projects"] >= 4:
        insights.append(("💻", "Rich Project Portfolio", "A strong project portfolio demonstrates initiative and practical technical depth."))
    if inputs["workshops"] >= 5:
        insights.append(("🎯", "Continuous Learner", "Multiple certifications signal a growth mindset—exactly what modern employers value."))
    if inputs["placement_training"] == "Yes":
        insights.append(("🎓", "Placement-Ready", "Placement training gives you a structured edge in interviews and aptitude rounds."))
    if inputs["extracurricular"] == "Yes":
        insights.append(("⚡", "Well-Rounded Profile", "Extracurricular involvement signals leadership, teamwork, and time management skills."))

    st.markdown('<div class="insights-section"><div class="insights-header">💡 Career Insights</div>', unsafe_allow_html=True)
    for icon, title, body in insights[:5]:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-icon">{icon}</div>
            <div class="insight-content">
                <div class="insight-title">{title}</div>
                <div class="insight-body">{body}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def render_recommendations(inputs: dict, pct: float) -> None:
    """Generate conditional, priority-labeled action items for career paths."""
    recs = []

    if pct >= 80:
        recs += [
            ("✅", "High", "Sharpen interview skills", "Practice STAR-method responses and case-study questions for your target companies."),
            ("✅", "High", "Research company culture", "Study the values and tech stack of your top 5 target companies in detail."),
            ("✅", "Medium", "Build a personal brand", "Publish LinkedIn posts, GitHub projects, or a portfolio website to stand out."),
            ("✅", "Medium", "Mock interview sessions", "Schedule at least 3 mock interviews with peers or mentors before D-Day."),
        ]
    elif pct >= 60:
        recs += [
            ("🔶", "High", "Strengthen aptitude preparation", "Dedicate 30 minutes daily to quantitative aptitude and logical reasoning practice."),
            ("🔶", "High", "Complete one more internship", "Even a short virtual internship substantially boosts your employability signals."),
            ("🔶", "Medium", "Earn 2–3 certifications", "Platforms like Coursera and edX offer recruiter-recognised path specializations."),
            ("🔶", "Medium", "Build a capstone project", "A domain-specific end-to-end project demonstrates both skills and initiative."),
        ]
    else:
        recs += [
            ("🔴", "Critical", "Enroll in placement training", "Structured training programmes dramatically improve aptitude and interview performance."),
            ("🔴", "Critical", "Improve academic performance", "Speak with faculty about improvement plans; CGPA above 7 opens significantly more doors."),
            ("🔴", "High", "Complete at least one internship", "Apply to internship portals now—even virtual experience shifts recruiter perception."),
            ("🔴", "High", "Develop communication skills", "Join a public-speaking workshop, peer defense club, or structural technical presentations."),
            ("🔴", "Medium", "Expand your project portfolio", "Build 2–3 projects on GitHub with solid documentation that showcase problem-solving."),
        ]

    if inputs["workshops"] < 2:
        recs.append(("📌", "Medium", "Attend workshops and hackathons", "Industry workshops sharpen skills and expand your professional network."))
    if inputs["extracurricular"] == "No":
        recs.append(("📌", "Low", "Join extracurricular activities", "Leadership roles in clubs and societies signal soft skills to recruiters."))

    st.markdown('<div class="recs-section"><div class="recs-header">🎯 Recommendations</div>', unsafe_allow_html=True)
    for icon, priority, title, body in recs[:5]:
        priority_class = f"priority-{priority.lower()}"
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-priority {priority_class}">{priority}</div>
            <div class="rec-content">
                <div class="rec-title">{icon} {title}</div>
                <div class="rec-body">{body}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
def render_footer() -> None:
    """Render the application branding and structural disclaimer footer."""
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">🎓 Placement Prediction System</div>
        <div class="footer-meta">
            Powered by Machine Learning &nbsp;·&nbsp;
            Built for Campus Recruitment &nbsp;·&nbsp;
            Predictions are probabilistic estimates, not guarantees.
        </div>
        <div class="footer-copy">© 2026 Placement Prediction System. All rights reserved.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Orchestrate state management, layouts, and evaluation executions."""
    render_hero()
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Inform the developer/user if the app falls back to rule-engine logic
    if not MODEL_AVAILABLE:
        st.info("💡 Running engine via static statistical weights wrapper. Connect 'utils.py' for pipeline inference.")

    # Render inputs and grab interactive state parameters
    inputs = render_inputs()
    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
    
    clicked = render_predict_button()

    # Session State Evaluation Matrix processing
    if clicked:
        with st.spinner("Analysing profile indicators with prediction models..."):
            
            # ─── PRE-INFERENCE CATEGORICAL DATA ENCODING ───
            extracurricular_encoded = 1 if inputs["extracurricular"] == "Yes" else 0
            placement_training_encoded = 1 if inputs["placement_training"] == "Yes" else 0

            pred, prob = predict_placement(
                cgpa=inputs["cgpa"],
                internships=inputs["internships"],
                projects=inputs["projects"],
                workshops=inputs["workshops"],
                aptitude_score=inputs["aptitude_score"],
                soft_skills=inputs["soft_skills"],
                extracurricular=extracurricular_encoded,
                placement_training=placement_training_encoded,
                ssc_marks=inputs["ssc_marks"],
                hsc_marks=inputs["hsc_marks"],
            )
            st.session_state["prediction_results"] = {"prediction": pred, "probability": prob}

    # Render the results conditionally if state data exists
    if "prediction_results" in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-tag">RESULTS</span>'
                    '<h2>Your Placement Assessment</h2></div>', unsafe_allow_html=True)
        
        res = st.session_state["prediction_results"]
        render_results(inputs, res["prediction"], res["probability"])

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    render_footer()


if __name__ == "__main__":
    main()