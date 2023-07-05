from flask import Flask , render_template , request
import joblib
# loading the model and scaler
model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')

app = Flask(__name__)
# categorical features 
seasons = ['spring','summer','winter']
weather = ['mist','rainy','snowy']
weekday = ['Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday']


@app.route('/' , methods = ['Get'])
def home():
    return render_template ('index.html')

@app.route('/predict' , methods = ['Get'])
def predict():
    inpt_data = [
    request.args.get('is_holiday',0),
    request.args.get('is_working_day',0),
    request.args.get('temp'),
    request.args.get('humidity'),
    request.args.get('windspeed'),
    request.args.get('rented_bikes_count'),
    request.args.get('Month'),
    request.args.get('Hour')
    ]
    season_dummies = [0 for i in range(3)]
    weather_dummies = [0 for i in range(3)]
    weekday_dummies = [0 for i in range(6)]
# passing the index of the categorical
    try:
        season_dummies[seasons.index(request.args.get('seasons'))] = 1
    except:
        pass     
    try:
        weather_dummies[weather.index(request.args.get('weather'))] = 1
    except:
        pass
    try:
        weekday_dummies[weekday.index(request.args.get('weekday'))] = 1
    except:
        pass 
# concatanate numerical with dummies features 
    inpt_data += season_dummies
    inpt_data += weather_dummies
    inpt_data += weekday_dummies

    inpt_data = [int(n) for n in inpt_data]

    profit = model.predict(scaler.transform([inpt_data]))[0]

    return render_template ('index.html',profit = profit)
# make the app acceable to multible users 
if __name__ == '__main__':
    app.run(debug = True , threaded=True)

