import json

from flask import Flask, render_template, request
from analyze_stock import analyze_stock
from analyze_options import analyze_options
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# testing potato
@app.route('/analyze/', methods=['GET'])
def hello():
    if request.method == 'GET':
        # print(ticker)
        ticker = request.args.get('ticker')
        options_date_index = request.args.get('options_date_index')
        if options_date_index :
            calls_chart_output, puts_chart_output, options_metrics = analyze_options(ticker, options_date_index)
        else:
            calls_chart_output, puts_chart_output, options_metrics = analyze_options(ticker)



        trend_chart_output, estimates, high_low_chart_output= analyze_stock(ticker)

        return render_template('chart.html',
                               ticker=ticker,
                               trend_chart_output=trend_chart_output,
                               estimates=estimates,
                               high_low_chart_output=high_low_chart_output,
                               calls_chart_output = calls_chart_output,
                               puts_chart_output = puts_chart_output,
                               options_metrics = options_metrics
                               )
    else:
        return "Enter ticker"




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001,debug=True)