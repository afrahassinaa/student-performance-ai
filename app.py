from flask import Flask, request, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# ==============================
# LOAD TRAINED ML MODEL
# ==============================

with open("student_performance_model.pkl", "rb") as file:
    model, encoder = pickle.load(file)


# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("Student_data.csv")


# ==============================
# DASHBOARD STATISTICS
# ==============================

total_students = len(data)

good_count = len(
    data[data["Performance"] == "Good"]
)

average_count = len(
    data[data["Performance"] == "Average"]
)

risk_count = len(
    data[data["Performance"] == "At Risk"]
)


avg_attendance = round(
    data["Attendance"].mean(), 1
)

avg_study = round(
    data["StudyHours"].mean(), 1
)

avg_previous = round(
    data["PreviousMarks"].mean(), 1
)


# ==============================
# HTML + CSS
# ==============================

html = """

<!DOCTYPE html>

<html>

<head>

<title>AI Student Performance Dashboard</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #eef2f7;
    margin: 0;
    padding: 30px;
}

h1 {
    text-align: center;
    color: #222;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}


/* ================= DASHBOARD CARDS ================= */

.dashboard {
    display: flex;
    justify-content: center;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 30px;
}

.card {
    background: white;
    width: 180px;
    padding: 18px;
    text-align: center;
    border-radius: 15px;
    box-shadow: 0 3px 12px #ccc;
}

.card h2 {
    font-size: 30px;
    margin: 8px;
}

.total {
    color: #333;
}

.good {
    color: green;
}

.average {
    color: orange;
}

.risk {
    color: red;
}


/* ================= ANALYTICS ================= */

.analytics-title {
    text-align: center;
    margin-top: 10px;
}

.analytics {
    display: flex;
    justify-content: center;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 30px;
}

.analytics-card {
    background: white;
    width: 210px;
    padding: 18px;
    text-align: center;
    border-radius: 15px;
    box-shadow: 0 3px 12px #ccc;
}

.analytics-card h2 {
    margin: 8px;
}


/* ================= FORM ================= */

.container {
    background: white;
    max-width: 600px;
    margin: auto;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 3px 15px #ccc;
}

.container h2 {
    text-align: center;
}

label {
    display: block;
    margin-top: 14px;
    font-weight: bold;
}

input {
    width: 95%;
    padding: 11px;
    margin-top: 6px;
    border: 1px solid #ccc;
    border-radius: 7px;
    font-size: 15px;
}

button {
    width: 100%;
    padding: 14px;
    margin-top: 22px;
    background: #222;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #444;
}


/* ================= RESULT ================= */

.result {
    margin-top: 28px;
    padding: 22px;
    background: #f5f5f5;
    border-radius: 12px;
}

.result h2 {
    text-align: center;
}

.warning {
    color: red;
    font-weight: bold;
    text-align: center;
}

.recommendation {
    text-align: left;
}

.recommendation h3 {
    margin-top: 20px;
}

hr {
    margin: 35px 0;
    border: none;
    border-top: 1px solid #ddd;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 30px;
    font-size: 13px;
}

</style>

</head>


<body>


<!-- ==============================
     HEADER
================================ -->

<h1>
🎓 AI Student Performance Dashboard
</h1>

<div class="subtitle">

AI-Based Student Performance Prediction
& Early Warning System

</div>


<!-- ==============================
     DASHBOARD
================================ -->

<div class="dashboard">


<div class="card">

<h3>Total Students</h3>

<h2 class="total">

{{ total_students }}

</h2>

</div>


<div class="card">

<h3>🟢 Good</h3>

<h2 class="good">

{{ good_count }}

</h2>

</div>


<div class="card">

<h3>🟡 Average</h3>

<h2 class="average">

{{ average_count }}

</h2>

</div>


<div class="card">

<h3>🔴 At Risk</h3>

<h2 class="risk">

{{ risk_count }}

</h2>

</div>


</div>


<!-- ==============================
     ANALYTICS
================================ -->

<h2 class="analytics-title">

📊 Academic Analytics

</h2>


<div class="analytics">


<div class="analytics-card">

<h3>Average Attendance</h3>

<h2>

{{ avg_attendance }}%

</h2>

</div>


<div class="analytics-card">

<h3>Average Study Hours</h3>

<h2>

{{ avg_study }}

</h2>

<p>hours/day</p>

</div>


<div class="analytics-card">

<h3>Average Previous Marks</h3>

<h2>

{{ avg_previous }}

</h2>

</div>


</div>


<hr>


<!-- ==============================
     INDIVIDUAL PREDICTION
================================ -->

<div class="container">


<h2>

🔍 Predict Individual Student

</h2>


<form method="POST">


<label>
Attendance (%)
</label>

<input
type="number"
name="attendance"
min="0"
max="100"
step="0.1"
placeholder="Example: 85"
required
>


<label>
Study Hours
</label>

<input
type="number"
name="study"
min="0"
max="24"
step="0.1"
placeholder="Example: 4"
required
>


<label>
Previous Marks
</label>

<input
type="number"
name="previous"
min="0"
max="100"
step="0.1"
placeholder="Example: 80"
required
>


<label>
Assignment Score
</label>

<input
type="number"
name="assignment"
min="0"
max="100"
step="0.1"
placeholder="Example: 85"
required
>


<label>
Internal Marks
</label>

<input
type="number"
name="internal"
min="0"
max="100"
step="0.1"
placeholder="Example: 82"
required
>


<button type="submit">

🔮 Predict Performance

</button>


</form>


<!-- ==============================
     RESULT
================================ -->

{% if result %}

<div class="result">


{% if result == "Good" %}


<h2 class="good">

🟢 Good Performance

</h2>


<p style="text-align:center;">

The student is currently performing well.

</p>


<div class="recommendation">

<h3>
💡 Recommendation
</h3>

<ul>

<li>
Maintain the current study routine.
</li>

<li>
Continue regular assignments.
</li>

<li>
Maintain good attendance.
</li>

</ul>

</div>


{% elif result == "Average" %}


<h2 class="average">

🟡 Average Performance

</h2>


<p style="text-align:center;">

The student may need additional academic support.

</p>


<div class="recommendation">

<h3>
💡 Recommendation
</h3>

<ul>

<li>
Increase daily study hours.
</li>

<li>
Improve assignment performance.
</li>

<li>
Focus on weak academic areas.
</li>

<li>
Maintain consistent attendance.
</li>

</ul>

</div>


{% else %}


<h2 class="risk">

🔴 At Risk

</h2>


<p class="warning">

⚠️ EARLY WARNING:
Student may be academically at risk.

</p>


<div class="recommendation">


<h3>
🚨 Warning Signs
</h3>


<ul>


{% if attendance < 75 %}

<li>
Low attendance detected.
</li>

{% endif %}


{% if study < 2 %}

<li>
Low study hours detected.
</li>

{% endif %}


{% if previous < 50 %}

<li>
Previous marks are low.
</li>

{% endif %}


{% if assignment < 50 %}

<li>
Assignment performance is low.
</li>

{% endif %}


{% if internal < 50 %}

<li>
Internal marks are low.
</li>

{% endif %}


</ul>


<h3>
💡 Recommended Actions
</h3>


<ul>

<li>
Increase study time.
</li>

<li>
Improve attendance.
</li>

<li>
Complete assignments regularly.
</li>

<li>
Seek academic guidance from faculty.
</li>

</ul>


</div>


{% endif %}


</div>

{% endif %}


</div>


<div class="footer">

AI-Based Student Performance Prediction & Early Warning System

</div>


</body>

</html>

"""


# ==============================
# FLASK ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    # Default values
    attendance = 0
    study = 0
    previous = 0
    assignment = 0
    internal = 0


    # ==============================
    # WHEN USER CLICKS PREDICT
    # ==============================

    if request.method == "POST":


        attendance = float(
            request.form["attendance"]
        )


        study = float(
            request.form["study"]
        )


        previous = float(
            request.form["previous"]
        )


        assignment = float(
            request.form["assignment"]
        )


        internal = float(
            request.form["internal"]
        )


        # Create input for ML model

        student_data = [[

            attendance,
            study,
            previous,
            assignment,
            internal

        ]]


        # ML prediction

        prediction = model.predict(
            student_data
        )


        # Convert prediction back to text

        result = encoder.inverse_transform(
            prediction
        )[0]


    # ==============================
    # SEND EVERYTHING TO HTML
    # ==============================

    return render_template_string(

        html,

        result=result,

        # IMPORTANT:
        # These fix the UndefinedError

        attendance=attendance,
        study=study,
        previous=previous,
        assignment=assignment,
        internal=internal,

        # Dashboard

        total_students=total_students,
        good_count=good_count,
        average_count=average_count,
        risk_count=risk_count,

        # Analytics

        avg_attendance=avg_attendance,
        avg_study=avg_study,
        avg_previous=avg_previous
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    app.run(debug=True)