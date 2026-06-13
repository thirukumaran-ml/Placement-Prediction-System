import os
import json
import joblib
import pandas as pd

# ==============================================================================
# PROJECT ARTIFACT PATH RESOLUTION
# ==============================================================================

# Resolves the absolute root path relative to this utility script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "final_scaler.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "model_config.json")

# ==============================================================================
# MACHINE LEARNING PIPELINE ASSET INGESTION
# ==============================================================================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    THRESHOLD = config.get("optimal_threshold", 0.5)

except Exception as e:
    raise FileNotFoundError(
        f"Critical Failure: Unable to load model artifacts from path directory. "
        f"Details: {str(e)}"
    )

# ==============================================================================
# INFERENCE PIPELINE LOGIC
# ==============================================================================

def predict_placement(
    cgpa,
    internships,
    projects,
    workshops,
    aptitude_score,
    soft_skills,
    extracurricular,
    placement_training,
    ssc_marks,
    hsc_marks
) -> tuple[int, float]:
    """
    Processes student profile inputs, handles categorical string mutations,
    applies feature scaling, and returns the placement prediction vector.

    Parameters:
        cgpa (float): Cumulative Grade Point Average.
        internships (int): Count of internships completed.
        projects (int): Count of academic/personal projects.
        workshops (int): Count of workshops/certifications attended.
        aptitude_score (float): Score achieved in analytical aptitude test.
        soft_skills (float): Self-assessed interpersonal communication score.
        extracurricular (str/int): Categorical status or pre-encoded integer.
        placement_training (str/int): Categorical status or pre-encoded integer.
        ssc_marks (float): 10th-grade percentage mark score.
        hsc_marks (float): 12th-grade percentage mark score.

    Returns:
        prediction (int): Binary value -> 1 if likely to be placed, 0 if at risk.
        probability (float): Numerical probability of falling under class 1.
    """

    # ─── EXTRA PROTECTIVE SANITIZATION LAYER ──────────────────────────────────
    # Immunizes the pipeline from crashing if raw strings leak from the UI.
    # Handles string variants like "Yes"/"No" and native numerical primitives.
    sanitizer_matrix = {
        "Yes": 1, "yes": 1, "Y": 1, 1: 1, 1.0: 1,
        "No": 0, "no": 0, "N": 0, 0: 0, 0.0: 0
    }

    clean_extracurricular = sanitizer_matrix.get(extracurricular, 0)
    clean_placement_training = sanitizer_matrix.get(placement_training, 0)

    # ─── CONSTRUCT INPUT MACHINE LEARNING VECTOR DATAFRAME ───────────────────
    # The dictionary keys map identically to the column names your models were trained on.
    input_df = pd.DataFrame({
        "CGPA": [float(cgpa)],
        "Internships": [int(internships)],
        "Projects": [int(projects)],
        "Workshops/Certifications": [int(workshops)],
        "AptitudeTestScore": [float(aptitude_score)],
        "SoftSkillsRating": [float(soft_skills)],
        "ExtracurricularActivities": [clean_extracurricular],
        "PlacementTraining": [clean_placement_training],
        "SSC_Marks": [float(ssc_marks)],
        "HSC_Marks": [float(hsc_marks)]
    })

    # ─── RUN SCALE AND ESTIMATE TRANSFORMS ────────────────────────────────────
    # Standardize variance features via standard scaler transformations
    scaled_input = scaler.transform(input_df)

    # Calculate model probability maps for target Class 1 [Placed Candidate]
    probability_array = model.predict_proba(scaled_input)
    probability = float(probability_array[0][1])

    # Classify candidate classification assignments across the optimal threshold boundary
    prediction = 1 if probability >= THRESHOLD else 0

    return prediction, probability