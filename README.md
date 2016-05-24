Deep Learning Model Builder and Analyser [dl-turtle]
====================================================

dl-turtle is a wrapper library around Theano which allows user to build 
deep learning models, set parameters, start and analyse training. dl-turtle 
comes with an intuitive UI enabling user to build model using drag and drop.

Theano is a scientific library for defining symbolic mathematical expressions. 
Theano simplifies deep learning because of its automatic differentiation 
feature. However, Theano is an over-kill for traditional applications of 
deep learning since it is geared towards researchers. Thus, building an 
end-to-end deep learning system in Theano is still a lot of work. dl-turtle 
simplifies model building by its canvas based intuitive web UI. It also allows 
user to view how the training is progressing using rich visualizations.

Modules
-------
1. Canvas based HTML5 UI 
	Web frontend for building the model graph and sending equivalent JSON 
	representation to server

2. Model Builder
	Parses JSON representation and builds equivalent model in Theano

3. Data Handler
	Handles standard datasets (currently supported datasets: MNIST, CIFAR)

4. Optimizer
	Takes the model and minimizes it using 'Gradient Descent' algorithms

5. Layers
	Basic building blocks of the deep net 
	(currently supported layers: convolution, pool, FC, softmax)
6. Sampler
	Samples 3 random images from test set and sends back the top-three 
	guesses for those images to the front-end

Work flow
---------
1. Model is represented by a graph using HTML5 UI
2. Graphical model is then transformed into a JSON representation
3. JSON representation of model is then sent to the server. 
4. Server re-directs the string to Model Builder which parses the string and 
   creates an equivalent model in Theano. 
5. User is then redirected to Analyser window where training progression is seen
6. Server sends cost along with sample test images with top three guesses back 
   to the user. Aalyser window incorporates this information as it arrives. 
  
Novelty
-------
dl-turtle could be used by students who could define model with ease, tune 
parameters, apply them to standard datasets and see how training progresses by 
plugging different hyper-parameters. Most of the heavy-lifting is then done by 
dl-turtle at server end. 
Once we integrate feature to upload dataset, persist and load model, dl-turtle 
could be used a general purpose tool for solving arbitrary deep learning 
problems.
