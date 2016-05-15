var cal = 0;
var socket = io();

function render_vals(){
    var skillBar = $('.outer').find('.inner');
    $.each(skillBar, function(){
        var skillVal = $(this).attr("data-progress");
        $(this).animate({
            width: (parseInt(skillVal) * 0.8) + '%'
        }, 1000);
    })	
}

socket.on('cal', function(val){
  if(!$('#data-stats').is(':visible')){
    $('#data-stats').slideDown(500) 
  }
    data = JSON.parse(val)
    for(var i = 0; i < 3; i++){//data.images.length
        var ele = $('.demo_item')[i];
        $(ele).find('img')[0].src = data.images[i]

        for(var j = 0; j < 3; j++){//data.probs[0].length
            var li = $(ele).find('.outer')[j]
            $(li).find('.lable').html(data.labels[i][j])
            $(li).find('.inner').attr("data-progress", data.probs[i][j].toFixed(2) * 100 + "%")
            console.log(data.probs[i][j]+ " --> " + data.probs[i][j].toFixed(2) * 100)
        }
    }
    render_vals()
    appendData(parseFloat(data.iteration))
});

var randomColorFactor = function() {
    return Math.round(Math.random() * 255);
};
var randomColor = function(opacity) {
    return 'rgba(' + randomColorFactor() + ',' + randomColorFactor() + ',' + randomColorFactor() + ',' + (opacity || '.3') + ')';
};

var config = {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: "Cost",
            data: [],
            fill: false,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        title:{
            display:true,
            text:'CNN iterations'
        },
        tooltips: {
            mode: 'label',
            callbacks: {
            }
        },
        hover: {
            mode: 'dataset'
        },
        scales: {
            xAxes: [{
                display: false,
                scaleLabel: {
                    show: true,
                    labelString: 'Iteration'
                },
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    show: true,
                    labelString: 'Cost'
                },
                /*ticks: {
                    suggestedMin: -10,
                    suggestedMax: 250,
                },*/
            }]
        }
    }
};

$.each(config.data.datasets, function(i, dataset) {
    dataset.borderColor = 'rgba(255, 0, 0, 0.4)'
    dataset.backgroundColor = 'rgba(255, 0, 0, 0.4)'
    dataset.pointBorderColor = 'rgba(255, 0, 0, 0.7)'
    dataset.pointBackgroundColor = 'rgba(255, 0, 0, 0.5)'
    dataset.pointBorderWidth = 1;
});

window.onload = function() {
    var ctx = document.getElementById("canvas").getContext("2d");
    window.myLine = new Chart(ctx, config);
};

$('#addData').click(function() {
    appendData(1000)
});

function appendData(data){
  var res = data
    if (config.data.datasets.length > 0) {
        config.data.labels.push(config.data.labels.length + 1);

        $.each(config.data.datasets, function(i, dataset) {
            dataset.data.push(res);
        });

        window.myLine.update();
    }
}

$('#removeData').click(function() {
    config.data.labels.splice(-1, 1);

    config.data.datasets.forEach(function(dataset, datasetIndex) {
        dataset.data.pop();
    });

    window.myLine.update();
});