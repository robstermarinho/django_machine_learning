from django.shortcuts import render

import io
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   							# Is a python 2D plotting lib
import numpy as np                  						# Package for scientific computing (N-dimensional array, linear algebra, random numbers...)
import pandas as pd                 						# Library for data analysis tools and data structures
#import seaborn as sns               						# Visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#Django
from django.shortcuts import render_to_response
from django.http import JsonResponse
from django.conf import settings

# Global Variables
lm = []

def index(request):

	global lm

	# Getting My Dataset
	diabetes = datasets.load_diabetes()

	# transform to Dataframe
	diabetes_X_df = pd.DataFrame(diabetes.data)
	diabetes_X_df.columns = list(diabetes.feature_names)
	html_diabeses = create_HTML_table(diabetes_X_df.head(50))

	# X,Y
	diabetes_X = diabetes.data
	diabetes_y = diabetes.target

	# Train Test Split
	X_train, X_test, y_train, y_test = train_test_split(
		diabetes_X,
		diabetes_y,
		test_size=0.25,
		random_state=1)

	# Create linear regression object
	lm = linear_model.LinearRegression()
	lm.fit(X_train, y_train)

	# Cross-validate the model
	cv_score = cross_val_score(lm, X_train, y_train, cv=5 )
	cv_score_train_mean_LINEAR = np.mean(cv_score)

	# Make cross-validated predictions
	y_train_pred_cv = cross_val_predict(lm, X_train, y_train, cv=5)
	validation_mse_LINEAR = mean_squared_error(y_train, y_train_pred_cv)

	# Explained variance score: 1 is perfect prediction
	validation_r2_score_LINEAR = r2_score(y_train, y_train_pred_cv)

	# Plotting
	ploted = plt.scatter(y_train, y_train_pred_cv, color = "green")
	fig = ploted.get_figure()
	file = "files/diabeses_graph1.png"
	path = settings.PROJECT_ROOT + "/static/" + file
	fig.savefig(path)

	#  Make predictions using the testing set
	y_test_pred = lm.predict(X_test)
	# The mean squared error
	test_mse_LINEAR = mean_squared_error(y_test, y_test_pred)
	# Explained variance score: 1 is perfect prediction
	test_r2_score_LINEAR = r2_score(y_test,y_test_pred)

	######################## VIEW ####################
	# JS script for this page
	additional_script = '<script type="text/javascript" src="' + settings.STATIC_URL + 'js/scripts_diabeses.js"></script>'
	# Render View
	base_url = request.get_host()
	if(base_url == "127.0.0.1:8000"):
		base_url = "http://127.0.0.1:8000"
	return render_to_response('templates/myapp/diabeses.html', {
		'html_diabeses': html_diabeses,
		'base_url': base_url,
		'cv_score_train_mean_LINEAR': cv_score_train_mean_LINEAR,
		'validation_mse_LINEAR': validation_mse_LINEAR,
		'validation_r2_score_LINEAR' : validation_r2_score_LINEAR,
		'test_mse_LINEAR': test_mse_LINEAR,
		'test_r2_score_LINEAR': test_r2_score_LINEAR,
		'file': file,
		'additional_script': additional_script
	})



# Get Predicted Diabeses

def getPredictedDiabeses(request):

	global lm
	data_get = [
		float(request.GET['age']),
		float(request.GET['sex']),
		float(request.GET['bmi']),
		float(request.GET['bp']),
		float(request.GET['s1']),
		float(request.GET['s2']),
		float(request.GET['s3']),
		float(request.GET['s4']),
		float(request.GET['s5']),
		float(request.GET['s6'])
	]
	X_new = [data_get]
	predicted = 0
	if(not isinstance(lm, list)):
		predicted = lm.predict(X_new)
		return render_to_response('templates/myapp/test.html', {'predicted': predicted})

# Creating a Styler object HTML table from a Dataframe
def create_HTML_table(dataframe):
	html_table_dataframe = (
		dataframe.style
		.set_properties(**{'font-size': '12pt', 'width' : '100%','font-family': 'Calibri'})
		.render()
		)
	return html_table_dataframe