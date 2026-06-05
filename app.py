from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

HTML = """
<h2>Customer Purchase Prediction</h2>
<form method="post">
Age: <input name="age"><br><br>
Income: <input name="income"><br><br>
Experience: <input name="experience"><br><br>
<input type="submit" value="Predict">
</form>
{% if prediction is not none %}
<h3>Prediction: {{prediction}}</h3>
{% endif %}
"""

@app.route("/", methods=["GET","POST"])
def home():
    pred = None
    if request.method == "POST":
        data = pd.DataFrame([[
            float(request.form["age"]),
            float(request.form["income"]),
            float(request.form["experience"])
        ]], columns=["age","income","experience"])

        pred = int(model.predict(data)[0])

    return render_template_string(HTML, prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
