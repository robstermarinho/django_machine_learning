from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [

	url(r'gdp/$', views.gdp),
	url(r'getPredictedY/$', views.getPredictedY),
	url(r'chartplot/$', views.dataframe_to_json_chart_plot),
]