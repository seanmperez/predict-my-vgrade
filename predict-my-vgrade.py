from flask import Flask, render_template, request, redirect, url_for, session
import utils.transform
import utils.predict

app = Flask(__name__, 
static_url_path='/static')

app.config["SECRET_KEY"] = "mysuperdupersecretkey"

@app.route("/", methods = ["GET", "POST"])
def index():
    """
    Renders the home page climber form.
    Upon submission, redirects to the results page
    """

    if request.method == "POST":
        #print("POST triggered")
        #print(print(f"\n\n Form Data---\n{request.form}\n---\n\n"))
        form_data = request.form

        #Transform data to a numpy array
        transformed_form_data = utils.transform.inputs_to_array(form_data)

        test_prediction = utils.predict.predict(transformed_form_data)
        #print(f"\n\n Predicted Data---\n {test_prediction} \n---\n\n")

        session["data"] = test_prediction
        
        return redirect(url_for("results"))

    return render_template('index.html')

@app.route("/results", methods=["GET", "POST"])
def results():
    """
    Returns the results
    """
    data = {"results": session.get("data")}
    #print(f"\nSession Data includes\n\n{data}\n\n")
    #print(f"\nType of Data is: {type(data)}\n")
    return render_template("results.html", data=data)

if __name__ == '__main__':
   app.run()