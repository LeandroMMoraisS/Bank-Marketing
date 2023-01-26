#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from flask import Flask, request, render_template,jsonify
import pickle
from joblib import dump, load
import os
import pandas as pd


# In[8]:


app = Flask(__name__)
model = load('model.pkl')


# In[11]:


@app.route('/')
def display_gui():
    render_template('template.html')


# @app.route('/verificar', methods=['POST'])
# def verificar():
#     age = request.form['age']
#     job = request.form['job']  
#     marital= request.form['marital']
#     education= request.form['education']
#     default = request.form['default']
#     housing = request.form['housing']
#     contact = request.form['contact']
#     month   = request.form['month']
#     day_of_week = request.form['day_of_week']
#     duration= request.form['duration']
#     campaign= request.form['campaign']
#     pdays= request.form['pdays']
#     previous= request.form['previous']
#     poutcome= request.form['poutcome']
#     emp.var.rate= request.form['emp.var.rate']
#     cons.price.idx= request.form['cons.price.idx']
#     cons.conf.idx=  request.form['cons.conf.idx']
#     euribor3m =  request.form['euribor3m']
#     nr.employed= request.form['nr.employed']
#     teste = np.array([age, job, marital, education, default, housing, loan,
#        contact, month, day_of_week, duration, campaign, pdays,
#        previous, poutcome, emp.var.rate, cons.price.idx,
#        cons.conf.idx, euribor3m, nr.employed])
#     print("::::::: Dados de teste ::::::")
#     print("age : {}".format(age))
#     print("job : {}".format(job))
#     print("marital : {}".format(marital))
#     print("education : {}".format(education))
#     print("default : {}".format(default))
#     print("housing : {}".format(housing))
#     print("contact : {}".format(contact))
#     print("month : {}".format(month))
#     print("day_of_week : {}".format(day_of_week))
#     print("duration : {}".format(duration))
#     print("campaign : {}".format(campaign))
#     print("pdays : {}".format(pdays))
#     print("previous : {}".format(previous))
#     print("poutcome : {}".format(poutcome))
#     print("emp.var.rate : {}".format(emp.var.rate))
#     print("cons.price.idx : {}".format(cons.price.idx))
#     print("euribor3m : {}".format(euribor3m))
#     print("nr.employed : {}".format(nr.employed))
#     
#     classe = model.predict(teste)[0]
#     print("Classe predita: {}".format(str(classe)))
#     
#     return render_template('template.html',classe=str(classe))

# In[21]:


@app.route('/predict', methods=['POST'])
def predict():
  dados = request.get_json(force=True)
  predicao = modelo.predict(np.array([list(dados.values())]))
  resultado = predicao[0]

  resposta = {'DIABETES': int(resultado)}
  return jsonify(resposta)


# In[20]:


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)


# In[ ]:




