import json

from flask import Flask, render_template, request
from analyze_stock import analyze_stock
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# testing potato
@app.route('/analyze/', methods=['GET'])
def hello():
    if request.method == 'GET':
        # print(ticker)
        ticker = request.args.get('ticker')


        trend_chart_output, estimates, high_low_chart_output= analyze_stock(ticker)

        return render_template('chart.html',
                               ticker=ticker,
                               trend_chart_output=trend_chart_output,
                               estimates=estimates,
                               high_low_chart_output=high_low_chart_output
                               )
    else:
        return "Enter ticker"

@app.route('/get_options_date', methods=['GET'])
def get_options_date():
    if request.method == 'GET':
        ticker = request.args.get('ticker')
        return json()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001,debug=True)