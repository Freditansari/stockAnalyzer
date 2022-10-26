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

        chart_result,\
             confidence,\
                 price_chart, \
                    estimates, \
                    price_estimates,\
                         daily_price_chart = analyze_stock(ticker)

        return render_template('chart.html', 
        chart_result = chart_result, 
        confidence=confidence, 
        price_chart= price_chart, 
        daily_price_chart = daily_price_chart,
        estimates=estimates, 
        price_estimates = price_estimates)
    else:
        return "Enter ticker"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)