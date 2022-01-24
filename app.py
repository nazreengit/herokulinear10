



from flask import Flask,request, jsonify
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])

def predict():
   
    
    if request.method =="POST":
        model = joblib.load("model_file.sav")
        value=request.form["value"]
        pred = model.predict([[value]])

       
        pre=''
        p=pred.tolist()
        for ele in p[0]:
            pre += str(ele)
        return jsonify(rating= pre)



if __name__ == '__main__':
    app.run(debug=True)        

