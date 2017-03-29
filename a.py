response_string="1,0,0,1,1,0.6,0.2"
from eyedisease_dataanalysis import predict_disease

predict = [int(e) if e.isdigit() else float(e) for e in response_string.split(',')]


s=predict_disease(response_string)
print (int(s[0]))

