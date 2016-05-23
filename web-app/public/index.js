var cal = 0;
var socket = io();

//renders bars according to the confidence values
function render_vals(){
    //find all skill bars
    var skillBar = $('.outer').find('.inner');
    $.each(skillBar, function(){
        //foreach skill bar get value of data-progress attribute
        var skillVal = $(this).attr("data-progress");
        $(this).animate({
            //create transition on width, transforming it to 0.6 in to skill value in 1s
            width: (parseInt(skillVal) * 0.6) + '%'
        }, 1000);
    })	
}

//listens to cal event from server node and processes data sent
socket.on('cal', function(val){
  if(!$('#data-stats').is(':visible')){
    //if stats are not being show, start showing, generally happens on first cal event
    $('#data-stats').slideDown(500) 
  }
    //extract info from json string
    data = JSON.parse(val)
    for(var i = 0; i < 3; i++){
        //update demo items with new values
        var ele = $('.demo_item')[i];
        $(ele).find('img')[0].src = data.images[i]

        for(var j = 0; j < 3; j++){
            var li = $(ele).find('.outer')[j]
            $(li).find('.lable').html(data.label_names[data.labels[i][j]])
            $(li).find('.inner').attr("data-progress", data.probs[i][j].toFixed(2) * 100 + "%")
        }
    }
  
    //render bars after updating test images
    render_vals()
    //add data point to graph
    appendData(parseFloat(data.iteration))
});

//creates random color intensity
var randomColorFactor = function() {
    return Math.round(Math.random() * 255);
};

//generates full random rgb color with .3 opacity
var randomColor = function(opacity) {
    return 'rgba(' + randomColorFactor() + ',' + randomColorFactor() + ',' + randomColorFactor() + ',' + (opacity || '.3') + ')';
};

//line graph configurations
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

//setting colors of graph explicitly
$.each(config.data.datasets, function(i, dataset) {
    dataset.borderColor = 'rgba(255, 0, 0, 0.4)'
    dataset.backgroundColor = 'rgba(255, 0, 0, 0.4)'
    dataset.pointBorderColor = 'rgba(255, 0, 0, 0.7)'
    dataset.pointBackgroundColor = 'rgba(255, 0, 0, 0.5)'
    dataset.pointBorderWidth = 1;
});

//load chart when document is fully loaded
window.onload = function() {
    var ctx = document.getElementById("canvas").getContext("2d");
    window.myLine = new Chart(ctx, config);
};

//for testing purposes
$('#addData').click(function() {
    appendData(1000)
});

//append a data point to graph
function appendData(data){
  var res = data
    if (config.data.datasets.length > 0) {
        config.data.labels.push(config.data.labels.length + 1);
        //add data point to each dataset
        $.each(config.data.datasets, function(i, dataset) {
            dataset.data.push(res);
        });

        window.myLine.update();
    }
}

//remove last inserted data point from graph
$('#removeData').click(function() {
    config.data.labels.splice(-1, 1);
    //remove last added data point from each dataset
    config.data.datasets.forEach(function(dataset, datasetIndex) {
        dataset.data.pop();
    });

    window.myLine.update();
});
