$(document).ready(function() {

	/* DATATABLES */

	$("#tab_1 table").addClass("table table-hover table-striped");
	$("#tab_1 table").css("width", "100%");
	$("#tab_2 table").addClass("table table-hover table-striped");
	$("#tab_2 table").css("width", "100%");
	$("#tab_3 table").addClass("table table-hover table-striped");
	$("#tab_3 table").css("width", "100%");

	$("#tab_1 table").DataTable({
		"pageLength": 5,
		"columnDefs": [
		{
			"targets": [ 5 ],
			"visible": false,
			"searchable": true
		},
		{
			"targets": [ 1 ],
			"width": "10%"
		},
		{
			"targets": [ 2 ],
			"width": "40%"
		}
		]
	});
	$("#tab_2 table").DataTable({
		"pageLength": 5,
		"columnDefs": [
		{
			"targets": [ 4 ],
			"visible": false,
			"searchable": true
		}]
	});

	$("#tab_3 table").DataTable({
		"pageLength": 5
	});


	/* BOOTSTRAP SLIDER */
	$('#gdp_slider').slider().on('slideStop', predict_gdp_value);

	getPredictedY(20000);
	load_graphs();


	function predict_gdp_value(e){
		$("#gdp_value").html($(this).val());
		getPredictedY($(this).val());
	}

	function getPredictedY(valuex){

		$.ajax({
			url: base_URL_ + "/myapp/getPredictedY/",
			method: "GET",
			async: true,
			data: { new_x : parseFloat(valuex)},
			dataType: "json",
			beforeSend: function() {
			},
			complete: function(){
			},
			success: function(data){
				if(data == 0){
					getPredictedY(valuex);
				}else{
					$("#ls_result").html(data);
				}
			},
			error: function(xhr, status, error) {
				alert("Internal Error");
				console.log(xhr);
				console.log(status);
				console.log(error);
			}
		});
	}

	function HTMLEscape(str) {   //Javascript code, adjust as per your server-side lang
		return String(str)
		.replace(/&/g, '&amp;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;');
	}

	/** GET data from graph 1 */
	data_graph1 = {};
	function load_graphs(){
		$.ajax({
			url: base_URL_ + "/myapp/chartplot/",
			async: true,
			dataType: "json",
			success: function(data){
				data_graph1 = JSON.parse(data)
				build_the_chart(data_graph1);
			},
			error: function(xhr, status, error) {
				alert("Internal Error");
			}
		});
	}


	function build_the_chart(data_graph1){
		var ctx = $("#areaChart");
		var color = Chart.helpers.color;
		var scatterChart = new Chart(ctx, {
			type: 'scatter',
			data: {
				labels: data_graph1.labels,
				datasets: [{
					label: 'Countries',
					borderColor: "red",
					backgroundColor: color("#555").alpha(0.2).rgbString(),
					data: data_graph1.data
				}]
			},
			options: {
				scales: {
					xAxes: [{
						type: 'linear',
						position: 'bottom'
					}]
				},
				tooltips: {
					callbacks: {
						label: function(tooltipItem, data) {
							var label = data.labels[tooltipItem.index];
							return label + ': (' + tooltipItem.xLabel + ', ' + tooltipItem.yLabel + ')';
						}
					}
				},

				legend: {
					boxWidth: 100,
					onHover: function(e) {
						e.target.style.cursor = 'pointer';
					},
					labels: {
						fontColor: 'rgb(255, 99, 132)'
					}
				},
				hover: {
					onHover: function(e) {
						var point = this.getElementAtEvent(e);
						if (point.length) e.target.style.cursor = 'pointer';
						else e.target.style.cursor = 'default';
					}
				},
				layout: {
					padding: {
						left: 10,
						right: 0,
						top: 0,
						bottom: 0
					}
				}
			}
		});
	}


	var color = Chart.helpers.color;
	mse1 = parseFloat($("#mse1").val()).toFixed(3);
	r21 = parseFloat($("#r21").val()).toFixed(3);
	mse2 = parseFloat($("#mse2").val()).toFixed(3);
	r22 = parseFloat($("#r22").val()).toFixed(3);

	var areaChartData = {
		labels  : ['Linear Modelling', 'Polynomial Modelling', 'REFERENCE'],
		datasets: [
		{
			label               : 'MSE',
			backgroundColor: '#00c468',
			data                : [mse1, r21, 0.05, -0.10]
		},
		{
			label               : 'R2',
			backgroundColor: '#00d2f7',
			data                : [mse2, r22, 1.000, 0]
		}
		]
	}



    //-------------
    //- BAR CHART -
    //-------------

    var barChartData                     = areaChartData;
    //barChartData.datasets[1].fillColor   = '#00a65a'
    //barChartData.datasets[1].strokeColor = '#00a65a'
    //barChartData.datasets[1].pointColor  = '#00a65a'
    var barChartOptions                  = {
      //Boolean - Whether the scale should start at zero, or an order of magnitude down from the lowest value
      scaleBeginAtZero        : true,
      //Boolean - Whether grid lines are shown across the chart
      scaleShowGridLines      : true,
      //String - Colour of the grid lines
      scaleGridLineColor      : 'rgba(0,0,0,.05)',
      //Number - Width of the grid lines
      scaleGridLineWidth      : 1,
      //Boolean - Whether to show horizontal lines (except X axis)
      scaleShowHorizontalLines: true,
      //Boolean - Whether to show vertical lines (except Y axis)
      scaleShowVerticalLines  : true,
      //Boolean - If there is a stroke on each bar
      barShowStroke           : true,
      //Number - Pixel width of the bar stroke
      barStrokeWidth          : 2,
      //Number - Spacing between each of the X value sets
      barValueSpacing         : 5,
      //Number - Spacing between data sets within X values
      barDatasetSpacing       : 1,
      //String - A legend template
      legendTemplate          : '<ul class="<%=name.toLowerCase()%>-legend"><% for (var i=0; i<datasets.length; i++){%><li><span style="background-color:<%=datasets[i].fillColor%>"></span><%if(datasets[i].label){%><%=datasets[i].label%><%}%></li><%}%></ul>',
      //Boolean - whether to make the chart responsive
      responsive              : true,
      maintainAspectRatio     : true,
      scales: {
      	xAxes: [{
      		ticks:{
      			beginAtZero:true
      		}
      	}]
      },
  };

  barChartOptions.datasetFill = false

  var barChartCanvas                   = $('#barChart');
  var barChart                         = new Chart(barChartCanvas, {
  	type: 'bar',
  	data: areaChartData,
  	options: barChartOptions
  });




/*
Chart(document.getElementById("chartjs-2"),{
	"type":"horizontalBar",
	"data":{
		"labels":["January","February","March","April","May","June","July"],
		"datasets":[
		{"label":"My First Dataset",
		"data":[65,59,80,81,56,55,40],
		"fill":false,"backgroundColor":[
		"rgba(255, 99, 132, 0.2)",
		"rgba(255, 159, 64, 0.2)",
		"rgba(255, 205, 86, 0.2)",
		"rgba(75, 192, 192, 0.2)","rgba(54, 162, 235, 0.2)",
		"rgba(153, 102, 255, 0.2)","rgba(201, 203, 207, 0.2)"],
		"borderColor":["rgb(255, 99, 132)",
		"rgb(255, 159, 64)","rgb(255, 205, 86)","rgb(75, 192, 192)",
		"rgb(54, 162, 235)","rgb(153, 102, 255)","rgb(201, 203, 207)"],
		"borderWidth":1}]},"options":{
			"scales":{"xAxes":[{"ticks":{"beginAtZero":true}}]}
		}});
		*/



	});
