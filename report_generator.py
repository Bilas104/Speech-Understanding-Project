
# report_generator.py

from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import numpy as np

def generate_personality_insights(scores):
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    insights = []
    summary_parts = []

    for i, score in enumerate(scores):
        label = traits[i]
        val = float(score)
        if val >= 0.7:
            level = "High"
        elif val >= 0.4:
            level = "Moderate"
        else:
            level = "Low"

        desc = interpret_trait(label, val)
        insights.append(f"{label} ({val:.2f}) - {level}: {desc}")

        if label == "Extraversion" and val > 0.7:
            summary_parts.append("outgoing")
        if label == "Openness" and val > 0.7:
            summary_parts.append("imaginative")
        if label == "Neuroticism" and val < 0.4:
            summary_parts.append("emotionally stable")
        if label == "Conscientiousness" and val > 0.7:
            summary_parts.append("well-organized")

    summary = "You appear to be " + ", ".join(summary_parts) + "." if summary_parts else "Your personality shows a balance across traits."

    return summary, insights

def interpret_trait(trait, score):
    if trait == "Openness":
        if score > 0.7:
            return "Highly imaginative, curious, and open to new experiences."
        elif score < 0.4:
            return "Prefers routine and traditional approaches."
        else:
            return "Moderately creative and open-minded."
    elif trait == "Conscientiousness":
        if score > 0.7:
            return "Well-organized, disciplined, and reliable."
        elif score < 0.4:
            return "Spontaneous and may struggle with organization."
        else:
            return "Balanced between planning and flexibility."
    elif trait == "Extraversion":
        if score > 0.7:
            return "Sociable, energetic, and expressive."
        elif score < 0.4:
            return "Reserved and introspective."
        else:
            return "Comfortable in both social and solitary settings."
    elif trait == "Agreeableness":
        if score > 0.7:
            return "Compassionate, cooperative, and friendly."
        elif score < 0.4:
            return "More assertive and competitive."
        else:
            return "Generally warm but maintains independence."
    elif trait == "Neuroticism":
        if score > 0.7:
            return "Emotionally reactive and sensitive to stress."
        elif score < 0.4:
            return "Calm, resilient, and emotionally stable."
        else:
            return "Sometimes reactive, but often composed."

def plot_radar(scores, labels=["O", "C", "E", "A", "N"], show=True):
    values = scores.tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values))
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Predicted Big Five Personality Traits")
    if show:
        plt.show()
    return fig

def generate_pdf(scores, radar_fig, output_path="personality_report.pdf"):
    summary, insights = generate_personality_insights(scores)

    tmp_img_path = tempfile.mktemp(suffix=".png")
    radar_fig.savefig(tmp_img_path)
    plt.close(radar_fig)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Personality Trait Report", ln=True, align='C')
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Summary: {summary}")
    pdf.ln(5)

    for line in insights:
        pdf.multi_cell(0, 10, line)
        pdf.ln(1)

    pdf.image(tmp_img_path, x=30, y=None, w=150)
    pdf.output(output_path)
    return output_path
