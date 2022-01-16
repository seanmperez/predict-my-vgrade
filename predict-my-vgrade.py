from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__, 
static_url_path='/static')

app.config["SECRET_KEY"] = "mysuperdupersecretkey"

@app.route("/", methods = ["GET", "POST"])
def index():
    """
    Renders the home page climber form.
    Upon submission, redirects to the results page
    """
    #print("\n\ntesting print at start\n\n")

    if request.method == "POST":
        #print("POST triggered")
        #print(print(f"\n\n Form Data---\n{request.form}\n---\n\n"))

        session["data"] = request.form
        
        return redirect(url_for("results"))

    return render_template('index.html')

@app.route("/results", methods=["GET", "POST"])
def results():
    """
    Returns the results
    """
    data = session.get("data")
    #print(f"\nSession Data includes\n\n{data}\n\n")
    return render_template("results.html", data=data)

if __name__ == '__main__':
   app.run()