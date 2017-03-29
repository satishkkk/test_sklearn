
import os
from flask import Flask,request, redirect, Response,jsonify
import requests

from eyedisease_dataanalysis import predict_disease
app = Flask(__name__)

#languages=[{'name':'C'} ,{'name':'JAVA'} , {'name':'PYTHON'} ,{'name':'RUBY'}]

@app.route('/getstr', methods=['POST'])
def getString():

    jsonResponse = {'response': request.json['data']}
    str=request.json['data']
    s=predict_disease(str)
    return jsonify({'prediction' : int(s[0])})


#@app.route('/', methods=['GET'])
#def test():
#    return jsonify({'message' : 'It works!'})

#@app.route('/lang', methods=['GET'])
#def retuenall():
#    return jsonify({'languages' : languages})

#@app.route('/lang', methods=['POST'])
#def getString1():
#   language = {'name': request.json['name']}
#   str=language
#   print (str)
#   return jsonify({'languages' : language})



@app.route('/')
def hello():
    return redirect('https://github.com/ankit96/moodler')


#setmoodler()

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)