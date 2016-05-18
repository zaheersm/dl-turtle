/*********************************************************************************************************/
/*********************************************************************************************************/
/******************************************* Configuaration **********************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

//layers: building blocks for building up model
var layers = {
  'convpool': [{color: 'green', text: 'Convpool', x: 20, y: 100, extras: {n_filters: 70, poolsize: [2,2]}}],
  softmax: [{color: 'red', text: 'Softmax', x: 20, y: 200, extras: {units: 10}}],
  fc: [{color: '#DF9300', text: 'Fully Connected', x: 130, y: 200, extras: {units: 500}}],
  input: [{color: '#0078ff', text: 'MNIST', x: 200, y: 100, extras: {batch_size: 4, input_shape:[1,28,28]}},{color: '#0078ff', text: 'CIFAR', x: 20, y: 300, extras: {batch_size: 500, input_shape: [3,32,32]}}],
  output: [{color: 'black', text: 'Output', x: 180, y: 300, extras: {}}]
}

//connection constraints, only key can act as input to listed outputs
var constraints = {
  fc: ['softmax', 'fc'],
  input: ['fc', 'convpool', 'softmax'],
  convpool: ['convpool', 'softmax', 'fc'],
  output: [],
  softmax: ['output']
}

/*********************************************************************************************************/
/*********************************************************************************************************/
/****************************************** Global Variables *********************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

//some necessory variables
var width = window.innerWidth;
var height = window.innerHeight;
var socket = io();

//for making connections
var activeObj = null,
    activeLine = null,
    acting = false,
    active_prop = null,
    props = null,
    prop_count = 0

//for easing effect
var tween = null;

//for uniques ids of objects
var obj_count = 0
var line_count = 0

//initializing Konva library stage
var stage = new Konva.Stage({
    container: 'container',
    width: width,
    height: height
});

//adding two object containers, one for building blocks and other for connections
var layer = new Konva.Layer();
var lineLayer = new Konva.Layer()
var prop_layer = new Konva.Layer();

//space for toolbox at right side
var tool_rect = new Konva.Rect({
  width: 300,
  height: stage.getHeight() - 200,
  fill: 'whitesmoke',
  stroke: 'white',
  strokeWidth: 2,
  cornerRadius: 5
})

//lable for toolbox
var tool_text = new Konva.Text({
  x: 10,
  y: 10,
  text: 'Tools',
  fontSize: 18,
  fontFamily: 'Calibri',
  fill: '#999'
});

var tool_line = new Konva.Line({
  points: [2, 30, 298, 30],
  stroke: '#ddd',
  strokeWidth: 1,
  lineCap: 'round',
  lineJoin: 'round'
});


var tool_group = new Konva.Group({
  x: 10,
  y: 10,
  draggable: true,
  id: 'tool'
})

var prop_group = new Konva.Group({
  x: stage.getWidth() - 310,
  y: 10,
  draggable: false,
  id: 'prop_window'
})

//** adding property window

var propery_text = new Konva.Text({
  x: 10,
  y: 10,
  text: 'Properties',
  fontSize: 18,
  fontFamily: 'Calibri',
  fill: '#999'
});

var prop_rect = new Konva.Rect({
  width: 300,
  height: 200,
  fill: 'whitesmoke',
  stroke: 'white',
  strokeWidth: 2,
  cornerRadius: 5
})

var prop_line = new Konva.Line({
  points: [2, 30, 298, 30],
  stroke: '#ddd',
  strokeWidth: 1,
  lineCap: 'round',
  lineJoin: 'round'
});

prop_group.add(prop_rect)
prop_group.add(prop_line)
prop_group.add(propery_text)

tool_group.add(tool_rect)
tool_group.add(tool_line)
tool_group.add(tool_text)
//tool_group.add(propery_text)

/*********************************************************************************************************/
/*********************************************************************************************************/
/******************************************* Firing up tools *********************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

//adding building blocks
add_layers(layers)

//adding button controls
add_button()


//adding toolbox to object container
layer.add(tool_group)
prop_layer.add(prop_group)

//adding object containers to stage
stage.add(layer);
stage.add(lineLayer)
stage.add(prop_layer)

//event handlers
layer.on('beforeDraw', function() {
   handle_connections()
});
stage.on('mousedown', function(){
  if(acting){
    acting = false
    activeObj.setAttrs({
      scale:{
        x: activeObj.getAttr('startScale'),
        y: activeObj.getAttr('startScale')
      }
    })

    var item = connected_successfully();
    if(item && item.getAttr('id') != activeObj.getAttr('id') && satisfy_constraints(activeObj, item) && item.getAttr('in') == null){
        activeLine.setPoints([activeLine.getAttr('points')[0], activeLine.getAttr('points')[1], item.getAttr('x') + item.getWidth()/2, item.getAttr('y')])
        activeLine.setAttrs({output: item.getAttr('id')})
        item = layer.find('#' + item.getAttr('id'))[0]
        item.setAttrs({in: activeLine.getAttr('id')})
        activeObj = layer.find('#' + activeObj.getAttr('id'))[0]
        activeObj.setAttrs({out: activeLine.getAttr('id')})
        handle_connections()
      }
    else
      activeLine.destroy()
    activeLine = null
    lineLayer.draw()
  }
  stage.off('mousemove')
})

/*********************************************************************************************************/
/*********************************************************************************************************/
/********************************* utility functions starts from here ************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

//extractor function extracts layers information form block diagram, returns JSON
function extractor(){
  var olayers = []
  var cur = layer.find('.input')[0]
  while(cur != null){
    var obj = {type :cur.getAttr('name')};
    if(olayers.length == 0)
      obj.type = cur.getAttr('val')
    var extras = cur.getAttr('extras')
    var keys = Object.keys(extras)
    for(var i = 0; i < keys.length; i++){
      obj[keys[i]] = extras[keys[i]]
      if(typeof(extras[keys[i]]) !== typeof(Array())){
        var ans = extras[keys[i]].split(',').map(function(val){
          return parseInt(val)
        })
        if(ans.length == 1)
          ans = ans[0]
        obj[keys[i]] = ans
      }
    }
    olayers.push(obj)
    cur = lineLayer.find('#' + cur.getAttr('out'))
    if(cur.length> 0){
      cur = layer.find('#' + cur[0].getAttr('output'))[0]
    }
    else{
      break
    }
  }
  var layermap = {meta : {'dataset' : olayers[0].type, batch_size: olayers[0].batch_size, input_shape: olayers[0].input_shape}, layers: []}

  for(var i = 1; i < olayers.length -1; i++){
    layermap.layers.push(olayers[i])
  }

  return layermap
}

//adds button controls, uses AddItem
function add_button(){
  addItem(layer, stage, {text: 'Generate Model', background_color: 'seagreen', color: 'white', x : 80, y : 450, isDraggable: false, isFancy : true, name: 'button', extras: {}}, button_handler);
}

//handles buttons events
function button_handler(item){
  item.on('mouseenter', function(){
    document.body.style.cursor = 'pointer';
    item.setAttrs({
      opacity: 1
    })
    layer.draw()
  })

  item.on('mouseleave', function(){
    document.body.style.cursor = 'default';
    item.setAttrs({
      opacity: 0.75
    })
    layer.draw()
  })

  item.on('mouseup', function(){
    socket.emit('command', JSON.stringify(extractor()))
    window.open('/demo', '_SELF')
  })
}

//add a building block to object container
function addItem(layer, stage, prop, handler = null) {

    var item = new Konva.Label({
        x: prop.x,
        y: prop.y,
        opacity: 1,//0.75,
        draggable: prop.isDraggable,
        startScale: 1,
        startX: prop.x,
        startY: prop.y,
        isFixed: true,
        child_name: prop.name,
        val: prop.text,
        extras: prop.extras
    });

    item.add(new Konva.Tag({
        fill: /*"#eef3f3",*/prop.background_color,
        cornerRadius: 5,
        stroke: "#ffffff",
        strokeWidth: 2
    }));

    item.add(new Konva.Text({
        text: prop.text,
        fontFamily: 'Calibri',
        fontSize: 18,
        padding: 20,
        fill: /*'#555555'*/prop.color
    }));

    if( handler == null)
      handler = fancy_dnd

    if(prop.isFancy)
        handler(item)
    tool_group.add(item)
}

//add all building blocks to object container, uses addItem
function add_layers(layers){
  var keys = Object.keys(layers)
    for(var i = 0; i < keys.length; i++){
      var list = layers[keys[i]]
      for(var j = 0; j < list.length; j++){
        addItem(layer, stage, {text: list[j].text, background_color: list[j].color/*'#eef3f3'*/, color: 'white'/*'#555555'*/, x : list[j].x, y : list[j].y, isDraggable: true, isFancy : true, name: keys[i], extras: list[j].extras}, tool_handler);
      }
    }
}

//checks if connection constructed satifies constrains
function satisfy_constraints(input, output){
  return constraints[input.getAttr('name')].indexOf(output.getAttr('name')) > -1
}

//arranges arrow properly on draggin objects
function handle_connections(){
  var items = lineLayer.find('.arrow')
  for(var i = 0; i < items.length; i++){
    var item = items[i]
    var input = layer.find('#' + item.getAttr('input'))[0];
    var output = layer.find('#' + item.getAttr('output'))[0];
    if(input && output){
      if(output.getAttr('y') > input.getAttr('y') + input.getHeight()){
        lineLayer.find('#' + item.getAttr('id'))[0].setPoints([input.getAttr('x') + input.getWidth()/2, input.getAttr('y') + input.getHeight(), output.getAttr('x') + output.getWidth()/2, output.getAttr('y')])
      }
      else if(input.getAttr('y') > output.getAttr('y') + output.getHeight()){
        lineLayer.find('#' + item.getAttr('id'))[0].setPoints([input.getAttr('x') + input.getWidth()/2, input.getAttr('y'), output.getAttr('x') + output.getWidth()/2, output.getAttr('y') + output.getHeight()])
      }
      else if(input.getAttr('x') > output.getAttr('x')){
        lineLayer.find('#' + item.getAttr('id'))[0].setPoints([input.getAttr('x'), input.getAttr('y') + input.getHeight()/2, output.getAttr('x') + output.getWidth(), output.getAttr('y') + output.getHeight()/2])
      }
      else{
        lineLayer.find('#' + item.getAttr('id'))[0].setPoints([input.getAttr('x') + input.getWidth(), input.getAttr('y') + input.getHeight()/2, output.getAttr('x'), output.getAttr('y') + output.getHeight()/2])
      }
    }
  }
  lineLayer.draw()
}

//checks if arrow initiated landed at an object or not
function connected_successfully(){
  var items = layer.find(Object.keys(layers).map(function(val){return "." + val + ' '}).join(','));
  var cursor = stage.getPointerPosition()
  for(var i = 0; i < items.length; i++){
    var item = items[i]
    var pos = item.getAbsolutePosition()
    if(cursor.x > pos.x && cursor.y > pos.y && cursor.x < pos.x + item.getWidth() && cursor.y < pos.y + item.getHeight()){
      return item
    }
  }
  return null
}

//event handlers for building blocks
function tool_handler(item){
  item.on('dragstart', function(evt) {
        var shape = item;

        item.stopDrag()
        var clone = item.clone({isFixed: false, name: item.getAttr('child_name'), id: 'node' + (layer.children.length + ++obj_count), in: null, out: null, x: item.getAbsolutePosition().x, y: item.getAbsolutePosition().y})
        clone.off('dragstart')
        clone.on('dragend', function(evt){
          var shape = item;
          if(object_in_tools()){
            var _in = lineLayer.find('#' + evt.target.getAttr('in'))
            var _out = lineLayer.find('#' + evt.target.getAttr('out'))
            if(_in.length > 0){
              var input = layer.find('#' + _in[0].getAttr('input'));
              //var output = layer.find('#' + _in.getAttr('output'));
              if(input.length > 0){
                input[0].setAttrs({out: null})
              }
              _in[0].destroy()
            }
            if(_out.length > 0){
              //var input = layer.find('#' + _out.getAttr('input'));
              var output = layer.find('#' + _out[0].getAttr('output'));
              if(output.length > 0){
                output[0].setAttrs({in: null})
              }
              _out[0].destroy()
            }
            //prop_layer.clear()
            layer.draw()
            setTimeout(function(){evt.target.destroy()}, 100)
          }
        })
        
        clone.on('click',function(){
          active_prop = clone
          render_property_window()
        })
        
        clone.on('dblclick', function(evt){
          if(clone.getAttr('out') != null)
            return
          stage.on('mousemove', mouse_move_handler)
          acting = true
          activeObj = clone

          activeLine = new Konva.Arrow({
            x: 0,
            y: 0,
            points: [clone.getAbsolutePosition().x + clone.getWidth()/2, clone.getAbsolutePosition().y + clone.getHeight(), stage.getPointerPosition().x + 100, stage.getPointerPosition().y + 100],
            pointerLength: 5,
            pointerWidth : 5,
            fill: '#555555',
            stroke: 'black',
            strokeWidth: 1,
            opacity: 0.8,
            name: 'arrow',
            input: clone.getAttr('id'),
            output: null,
            id: 'line' + (lineLayer.children.length + ++line_count)
          });

          lineLayer.add(activeLine)
        })

        fancy_dnd(clone)

        layer.add(clone)
        clone.startDrag()

        shape.setAttrs({
            x: shape.getAttr('startX'),
            y: shape.getAttr('startY'),
            scale: {
                x: shape.getAttr('startScale'),
                y: shape.getAttr('startScale'),
            }
        });

    });
}

//checks if object is in toolbox
function object_in_tools(){
  var item = stage.getPointerPosition();
  var tool = layer.find('#tool')[0]
  var tool_pos = tool.getAbsolutePosition()
  
  return (item.x > tool_pos.x && item.y > tool_pos.y && item.x < tool_pos.x + tool.children[0].getWidth() && item.y < tool_pos.y + tool.children[0].getHeight())
}

//adds tweens to drag and drop
function fancy_dnd(item){
    item.on('dragstart', function(evt) {
        var shape = item;
        if (tween) {
            tween.pause();
        }
        shape.setAttrs({
            shadowOffset: {
                x: 15,
                y: 15
            },
            scale: {
                x: shape.getAttr('startScale') * 1.2,
                y: shape.getAttr('startScale') * 1.2
            }
        });
    });

    item.on('dragend', function(evt) {
        var shape = item;

        tween = new Konva.Tween({
            node: shape,
            duration: 0.5,
            easing: Konva.Easings.ElasticEaseOut,
            scaleX: shape.getAttr('startScale'),
            scaleY: shape.getAttr('startScale'),
            shadowOffsetX: 5,
            shadowOffsetY: 5
        });

        tween.play();
        
    });
}

//handles arrow being drawn
function mouse_move_handler(){
  if(acting){
    activeLine.setPoints([activeLine.getAttr('points')[0], activeLine.getAttr('points')[1], stage.getPointerPosition().x, stage.getPointerPosition().y])
  }

  handle_connections()
}

//renders properties of active_prop object
function render_property_window(){
  if(active_prop == null)
    return
  
  //prop_layer.clear()
  if(props != null && props.length > 0){
    for(var i = 0; i < props.length; i++){
      props[i].lable.destroy()
      props[i].field.destroy()
    }
    props = []
  }
    
  var prop = layer.find('#' + active_prop.getAttr('id'))[0].getAttr('extras')
  var render_offset = 30
  
  var keys = Object.keys(prop)
  
  props = []
  
  for(var i = 0; i < keys.length; i++){
    render_prop(render_offset, keys[i], prop[keys[i]])
    render_offset += 40
  }
  
  prop_layer.draw()
  
  for(var i = 0; i < props.length; i++){
    props[i].field.render()
  }
  
}

//renders a single property at specified location
function render_prop(y, key, val){
  var txt = new Konva.Text({
    x: 0,
    y: y,
    text: to_title_case(key.replace('_', ' ')),
    fontFamily: 'Calibri',
    fontSize: 18,
    padding: 20,
    fill: '#555555',
    id: 'prop' + prop_count + 1
  });
  
  prop_group.add(txt)
  
  var prop_win_pos = prop_group.getAbsolutePosition()
  
  var input = new CanvasInput({
    canvas: $('#container canvas')[2],
    fontSize: 12,
    x: stage.getWidth() - 300 + 150,
    y: y + 25,
    fontFamily: 'Calibri',
    fontColor: '#555555',
    //fontWeight: 'bold',
    width: 100,
    padding: 3,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 1,
    //boxShadow: '1px 1px 0px #fff',
    innerShadow: '0px 0px 5px rgba(0, 0, 0, 0.1)',
    placeHolder: '',
    value: val,
    onkeydown: function(e){
      if(e.keyCode == 13){
        input.blur()
      }
    },
    onblur: function(){
      var ext = active_prop.getAttr('extras')
      if(ext[key] !== undefined)
        ext[key] = input.value()
      active_prop.setAttrs({
        extras: ext
      })
    }
  });
  
  props.push({lable: txt, field: input})
}


//normal text to title case
function to_title_case(text){
  return text.replace(/\w\S*/g, function(word){
    return word.charAt(0).toUpperCase() + word.substr(1).toLowerCase()
  })
}
/*********************************************************************************************************/
/*********************************************************************************************************/
/******************************************* Code ends here **********************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/
