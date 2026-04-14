import gradio as gr
from core.diagnostics import SymptomDiagnosticEngine
from core.doctors import find_doctors, create_doctor_map
from core.insurance import InsuranceEstimator
from core.memory import generate_patient_id, update_patient_memory, get_patient_history
from core.explainer import AIExplainer

diagnostic_engine = SymptomDiagnosticEngine()
insurance_estimator = InsuranceEstimator()
ai_explainer = AIExplainer()

def calculate_bmi(height_cm, weight_kg):
    height_m = float(height_cm) / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    bmi = float(weight_kg) / (height_m ** 2)
    return round(bmi, 2)

def format_doctors(doctors):
    if not doctors:
        return "No doctors found."
    lines = []
    for doc in doctors:
        lines.append(
            f"{doc['name']} | {doc['specialty']} | {doc['city']} | Rating: {doc['rating']}"
        )
    return "\n".join(lines)

def format_history(history):
    if not history:
        return "No previous patient history found."
    visits = history.get("history", [])
    if not visits:
        return "No previous patient history found."
    lines = []
    lines.append("Patient Visit Timeline")
    lines.append("-" * 40)
    for idx, visit in enumerate(visits, start=1):
        diag = visit.get("diagnosis", {})
        top = diag.get("top_diagnosis", {}) if isinstance(diag, dict) else {}
        demographics = history.get("demographics", {})
        lines.append(f"Visit {idx}")
        lines.append(f"Timestamp: {visit.get('timestamp', 'Unknown')}")
        lines.append(f"Symptoms: {visit.get('symptoms', 'Unknown')}")
        lines.append(f"Health insight: {top.get('diagnosis', 'Unknown')}")
        lines.append(f"Recommended specialty: {top.get('specialty', 'Unknown')}")
        lines.append(f"Insurance estimate: {visit.get('insurance_price', 'Unknown')}")
        if demographics.get("bmi") is not None:
            lines.append(f"Stored BMI: {demographics.get('bmi')}")
        lines.append("-" * 40)
    return "\n".join(lines)

def format_summary(result):
    top = result.get("top_diagnosis")
    if not top:
        return "No health insight found."
    summary_lines = [
        f"Health Insight: {top.get('diagnosis', 'Unknown')}",
        f"Recommended Specialty: {top.get('specialty', 'Unknown')}",
        f"Urgency Level: {top.get('triage_level', 'Unknown')}",
    ]
    if result.get("calculated_bmi") is not None:
        summary_lines.append(f"Calculated BMI: {result.get('calculated_bmi')}")
    return "\n".join(summary_lines)

def format_detailed_analysis(result):
    top = result.get("top_diagnosis")
    if not top:
        return "No detailed analysis found."
    lines = []
    lines.append("Structured Health Analysis")
    lines.append("-" * 40)
    lines.append(f"Matched symptom pattern: {top.get('symptom_pattern', 'Unknown')}")
    lines.append(f"Health insight: {top.get('diagnosis', 'Unknown')}")
    lines.append(f"Recommended specialty: {top.get('specialty', 'Unknown')}")
    lines.append(f"Urgency level: {top.get('triage_level', 'Unknown')}")
    lines.append(f"Suggested tests: {top.get('recommended_tests', 'Unknown')}")
    lines.append(f"Advice: {top.get('advice', 'Unknown')}")
    lines.append(f"Similarity confidence: {top.get('confidence', 0):.4f}")
    if result.get("calculated_bmi") is not None:
        lines.append(f"Calculated BMI: {result.get('calculated_bmi')}")
    if result.get("history_summary"):
        lines.append("")
        lines.append(f"History summary: {result['history_summary']}")
    if result.get("safety_disclaimer"):
        lines.append("")
        lines.append(result["safety_disclaimer"])
    return "\n".join(lines)

def analyze(name, email, age, sex, height_cm, weight_kg, children, smoker, region, symptoms, city, use_ai):
    try:
        if not name.strip() or not email.strip() or not symptoms.strip():
            msg = "Please fill in name, email, and symptoms."
            return msg, msg, msg, msg, msg, msg, "<p>No map available.</p>"
        patient_id = generate_patient_id(name, email)
        result = diagnostic_engine.diagnose(symptoms, patient_id)
        top = result.get("top_diagnosis")
        if not top:
            msg = "No health insight could be generated."
            return msg, msg, msg, msg, msg, msg, "<p>No map available.</p>"
        doctors = find_doctors(top.get("specialty", ""), city)
        bmi = calculate_bmi(height_cm, weight_kg)
        result["calculated_bmi"] = bmi
        insurance_price, insurance_text = insurance_estimator.estimate(
            patient_id=patient_id,
            age=int(age),
            sex=sex,
            bmi=float(bmi),
            children=int(children),
            smoker=smoker,
            region=region,
        )
        if use_ai and not ai_explainer.model_loaded:
            ai_explainer.enable_local_model()
        payload = {
            "top_diagnosis": top,
            "doctors": doctors,
            "insurance_text": insurance_text,
            "history_summary": result.get("history_summary", ""),
        }
        ai_text = ai_explainer.explain(payload, use_ai=use_ai)
        demographics = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "city": city,
        }
        update_patient_memory(
            patient_id=patient_id,
            demographics=demographics,
            symptoms=symptoms,
            diagnosis_result=result,
            insurance_price=insurance_price,
            doctors=doctors,
        )
        history = get_patient_history(patient_id)
        detailed_analysis = format_detailed_analysis(result)
        summary = format_summary(result)
        doctor_text = format_doctors(doctors)
        history_text = format_history(history)
        doctor_map_html = create_doctor_map(doctors)
        return (
            detailed_analysis,
            insurance_text,
            summary,
            ai_text,
            history_text,
            doctor_text,
            doctor_map_html,
        )
    except Exception as e:
        msg = f"Error during analysis: {e}"
        return msg, msg, msg, msg, msg, msg, "<p>No map available.</p>"

def show_history(name, email):
    try:
        if not name.strip() or not email.strip():
            return "Please enter both name and email."
        patient_id = generate_patient_id(name, email)
        history = get_patient_history(patient_id)
        return format_history(history)
    except Exception as e:
        return f"Error while loading history: {e}"

with gr.Blocks(title="Healthcare AI Agent") as demo:
    gr.Markdown("# Healthcare AI Agent")
    gr.Markdown("From Data → Intelligence → AI Product")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Full Name")
            email = gr.Textbox(label="Email")
            age = gr.Number(label="Age", value=35)
            sex = gr.Dropdown(
                choices=["male", "female"],
                value="male",
                label="Sex",
            )
            height_cm = gr.Number(label="Height (cm)", value=170)
            weight_kg = gr.Number(label="Weight (kg)", value=70)
            children = gr.Number(label="Children", value=1)
            smoker = gr.Dropdown(
                choices=["yes", "no"],
                value="no",
                label="Smoker",
            )
            region = gr.Dropdown(
                choices=["northeast", "northwest", "southeast", "southwest"],
                value="southeast",
                label="Region",
            )
            city = gr.Dropdown(
                choices=["Seoul", "Hanoi", "Ho Chi Minh City", "Da Nang"],
                value="Seoul",
                label="City",
            )
            symptoms = gr.Textbox(
                label="Describe Symptoms",
                lines=4,
                placeholder="Example: chest pain and sweating",
            )
            use_ai = gr.Checkbox(
                label="Use local AI explanation (experimental)",
                value=False,
            )
            analyze_button = gr.Button("Analyze")
            history_button = gr.Button("Show patient history")
        with gr.Column():
            detailed_analysis = gr.Textbox(
                label="Full analysis",
                lines=12,
            )
            insurance_box = gr.Textbox(
                label="Insurance estimate",
                lines=2,
            )
            summary_box = gr.Textbox(
                label="Health insight summary",
                lines=4,
            )
            ai_box = gr.Textbox(
                label="Explanation layer output",
                lines=8,
            )
            history_box = gr.Textbox(
                label="Patient history",
                lines=10,
            )
            doctors_box = gr.Textbox(
                label="Doctor recommendations",
                lines=6,
            )
            doctor_map = gr.HTML(label="Doctor map")
    analyze_button.click(
        analyze,
        inputs=[name, email, age, sex, height_cm, weight_kg, children, smoker, region, symptoms, city, use_ai],
        outputs=[
            detailed_analysis,
            insurance_box,
            summary_box,
            ai_box,
            history_box,
            doctors_box,
            doctor_map,
        ],
    )
    history_button.click(
        show_history,
        inputs=[name, email],
        outputs=history_box,
    )

if __name__ == "__main__":
    demo.launch(share=True)