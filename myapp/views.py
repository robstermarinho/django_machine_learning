from django.shortcuts import render

# Create your views here.
import io
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   						# Is a python 2D plotting lib
import numpy as np                  						# Package for scientific computing (N-dimensional array, linear algebra, random numbers...)
import pandas as pd                 						# Library for data analysis tools and data structures
#import seaborn as sns               						# Visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics
from sklearn import linear_model                           	# Tools for data mining and data analysis
from sklearn.model_selection import train_test_split       	# Split arrays or matrices into random train and test subsets
from sklearn.metrics import mean_squared_error, r2_score   	# Mean squared error regression loss; R^2 score - Coefficient of determination
from django.shortcuts import render_to_response
from django.http import JsonResponse
from django.conf import settings

# Global Variables
lr_model = []
life_sat_gdp = []

def gdp(request):

	global lr_model, life_sat_gdp

	# Reading CSV files
	gdp_pc = read_csv_gdp("/myapp/data/gdp_per_capita.csv")
	life_sat = read_csv_life_sat("/myapp/data/oecd_bli_2015.csv")

	# Creating a Styler object from a Dataframe
	html_table_gdp = (
    	gdp_pc.style
	    .set_properties(**{'font-size': '8pt', 'font-family': 'Calibri'})
	    .bar(subset=['2015'], color='lightblue')
	    .render()
	)

	#Pre - Process DATA
	life_sat_condition = (life_sat["INDICATOR"] == "SW_LIFS") & (life_sat["INEQUALITY"] == "TOT")
	columns_of_interest = ["Country", "INDICATOR", "Value"]
	life_sat_by_country = life_sat[life_sat_condition][columns_of_interest]

	columns_of_interest = ["Country", "2015"]
	gdp_by_country = gdp_pc[columns_of_interest]

	life_sat_gdp = pd.merge(gdp_by_country, life_sat_by_country, on = "Country" )
	life_sat_gdp = life_sat_gdp[["Country", "2015", "Value"]]
	life_sat_gdp.columns = ["Country", "GDP_2015", "Life_Satisfaction"]

	# Sorting by Country
	life_sat_gdp = life_sat_gdp.sort_values(by=('Country'), ascending=True)

	# Plotting the dataset using Matplotlib
	ploted = life_sat_gdp.plot(kind = "scatter",
		x = "GDP_2015", y = "Life_Satisfaction",
		title = "Life Satisfaction x GDP", color="#62adea")

	fig = ploted.get_figure()
	file1 = ({
		'path' : "files/graph1.png",
		'title': "Life Satisfaction x GDP",
		'life_sat_gdp_json' : life_sat_gdp.to_json()
	})
	path = settings.PROJECT_ROOT + "/static/" + file1['path']
	fig.savefig(path)

	#Taining the model
	X = np.c_[life_sat_gdp["GDP_2015"]]
	y = np.c_[life_sat_gdp["Life_Satisfaction"]]

	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		train_size=0.8,
		test_size=0.2, 
		random_state=12341)

	lr_model = linear_model.LinearRegression()
	lr_model.fit(X_train, y_train)
	y_train_pred = lr_model.predict(X_train)

	# Plotting the result of the modeling
	plt.scatter(X_train, y_train, color = "orange", alpha = 0.5, edgecolors = 'red')
	ploted2 = plt.plot(X_train, y_train_pred, color = "#8e350b", linewidth = 3, alpha = 0.7)
	plt.title("Fitting a linear model to the training set")
	plt.xlabel("GDP 2015")
	plt.ylabel("Life Satisfaction Index")

	fig2 = ploted.get_figure()
	file2 = ({
		'path' : "files/graph2.png",
		'title': "Fitting a linear model to the training set",
	})
	path = settings.PROJECT_ROOT + "/static/" + file2['path']
	fig2.savefig(path)


	base_url = request.get_host()
	return render_to_response('templates/myapp/gdp.html', {
		'data': life_sat_gdp.to_html(),
		'gdp_pc': html_table_gdp,
		'life_sat': life_sat.to_html(),
		'file1' : file1,
		'file2' : file2,
		'base_url': base_url
	})



def dataframe_to_json_chart_plot(request):
	global life_sat_gdp

	# Preparing labels
	labels = list(life_sat_gdp["Country"])

	# Preparing data
	x_ = np.array(life_sat_gdp["GDP_2015"].values)
	y_ = np.array(life_sat_gdp["Life_Satisfaction"].values)
	data = []
	for a in range(0, len(x_)):
	    data.append({ "x" : x_[a], "y" : y_[a]})

	result = ({'labels' : labels, "data": data})
	final_json_result = json.dumps(result)

	return JsonResponse(final_json_result, safe=False)
	#return HttpResponse(final_json_result, content_type='application/json')
	#return render_to_response('templates/myapp/json_scatter.html', {'labels': labels, 'data',})


# Get Predicted Y
def getPredictedY(request):

	global lr_model

	new_x = float(request.GET['new_x'])
	X_new = [[new_x]]
	predicted = 0
	if(not isinstance(lr_model, list)):
		predicted = lr_model.predict(X_new)
	return render_to_response('templates/myapp/test.html', {'predicted': predicted, 'new_x' : new_x})


## Reading the GDP CSV file
def read_csv_gdp(path):
	gdp_pc = pd.read_csv(os.path.realpath('.') + path,
		delimiter = "\t", thousands = ",",
		encoding = "latin1", na_values = "n/a")
	return gdp_pc

## Reading the LIFE_SAT CSV file
def read_csv_life_sat(path):
	life_sat = pd.read_csv(os.path.realpath('.') + path,
		thousands = ",")
	return life_sat