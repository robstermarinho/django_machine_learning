$(document).ready(function() {

	$(".dataframe").addClass("table table-hover table-striped");
	$(".dataframe").prop("border", 0);
	$(".dataframe").DataTable();

	$("#tab_1 table").DataTable();
	// With JQuery
	$('#gdp_slider').slider({
		formatter: function(value) {
			return 'Current value: ' + value;
		}
	});


	$("#gdp_slider").change(function(e){

		e.preventDefault();
		$("#gdp_value").html($("#gdp_slider").val());
		$.ajax({
			url: base_URL_ + "/myapp/getPredictedY/",
			method: "GET",
			data: { new_x : $("#gdp_slider").val() },
			dataType: "json",
			beforeSend: function() {
				$("#predict_life_satisfaction").prop("disabled", true);
			},
			complete: function(){
				$("#predict_life_satisfaction").prop("disabled", false);
			},
			success: function(data){
				$("#ls_result").html(data);
			},
			error: function(data){
				alert('error');
			},
		});
	});


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
	$.ajax({
		url: base_URL_ + "/myapp/chartplot/",
		async: true,
		dataType: "json",
		success: function(data){
			var data_graph1 = JSON.parse(data)
			build_the_chart(data_graph1);
		},
		error: function(xhr, status, error) {
			var err = eval("(" + xhr.responseText + ")");
			alert(err.Message);
		}
	});

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
				}
			}
		});
	}
});
