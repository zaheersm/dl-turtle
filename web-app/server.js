var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var net = require('net');

app.use(express.static('public'));
var client = new net.Socket();

app.get('/', function(req, res){
  res.sendFile(__dirname + '/model.html')
});

app.get('/demo', function(req, res){
  res.sendFile(__dirname + '/index.html')
});

var connected_socket = 0
var server_connected = false

io.on('connection', function(socket){
  connected_socket = socket
  socket.on('command', function(msg){
    if(server_connected){
      client.write(msg);
      server_connected = false
    }
    else{
      client.connect(4777, '127.0.0.1', function() {
        console.log('Connected');
        client.write(msg);
        server_connected = true
      });
    }
  });
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});

/*setInterval(function(){
  if(connected_socket != 0)
    io.emit('data', 3000 * Math.random());
}, 100)*/

client.on('data', function(data) {
	io.emit('cal', data.toString())
});

client.on('close', function() {
	console.log('Connection closed');
	server_connected = false
	
});
