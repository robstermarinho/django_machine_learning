$(document).ready(function() {

	/* DATATABLES */
	$("#diabese_tab_1 table").addClass("table table-hover table-striped");
	$("#diabese_tab_1 table").css("width", "100%");
	$("#diabese_tab_1 table").DataTable({"pageLength": 5,});



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
});
