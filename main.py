import pickle
import json
import numpy as np
from wsgiref import simple_server
from flask import Flask, request, render_template

"""
           This file is used to run application!!.

           Written By: Mustafa Khan
           Version: 1.0
           Revisions: None

           """
app = Flask(__name__)

def get_predict_profit(r_d_expenses, administration_expenses, marketing_expenses, state):
    """
               This function is used to predict results!!.

               Written By: Mustafa Khan
               Version: 1.0
               Revisions: None
               Parameters
                 r_d_expenses: R&D spent
                 administration_expenses: admin cost
                 marketing_expenses: marketing expenses
                 state: state name

               """
    with open('models/profit_prediction.pkl', 'rb') as f:
        model = pickle.load(f)

    with open("models/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    try:
        state_index = data_columns.index('state_' + str(state).lower())
    except:
        state_index = -1

    x = np.zeros(len(data_columns))
    x[0] = r_d_expenses
    x[1] = administration_expenses
    x[2] = marketing_expenses
    if state_index >= 0:
        x[state_index] = 1

    return round(model.predict([x])[0], 2)

@app.route('/')
def index_page():
    """
               This method is used to display index page!!.

               Written By: Mustafa Khan
               Version: 1.0
               Revisions: None

               """
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    """
               This method is used to predict!!.

               Written By: Mustafa Khan
               Version: 1.0
               Revisions: None

               """

    if request.method == 'POST':
        r_d_expenses = request.form['r_d_expenses']
        administration_expenses = request.form["administration_expenses"]
        marketing_expenses = request.form["marketing_expenses"]
        state = request.form["state"]
        output = get_predict_profit(r_d_expenses, administration_expenses, marketing_expenses, state)
        return render_template('index.html', show_hidden=True,
                               prediction_text='Startup Profit must be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)