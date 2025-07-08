from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model and features
with open("model.pkl", "rb") as f:
    model, selected_features, feature_importance = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            user_input = {feat: float(request.form[feat]) for feat in selected_features}
            df = pd.DataFrame([user_input])
            price = model.predict(df)[0]
            rounded_price = int(round(price, -2))
            suggestions = feature_importance.head(3).index.tolist()
            return render_template("result.html", prediction=rounded_price, suggestions=suggestions)
        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", features=selected_features)

if __name__ == "__main__":
    app.run(debug=True)
