## Selected Results from the Paper
Selected results for the paper can be found in more detailed at [LCAV's Reproducibility Research platform](http://rr.epfl.ch/paper/KSV2015). Here, we briefly report some selected results for the connections in a recurrent network with *60* neurons, *50* excitatory and *10* inhibitory. The network was generated artificially (using the [Brian simulator](http://briansimulator.org/)) and the underlying graph was a *directed*
random graph (i.e., an Erdos-Renyi model), with a connection probability of *0.2*. A short section of the graph (10 by 10) is shown below. The raster plot of generated data is shown below as well.


<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Actural_Graph.png" style="margin-bottom:5px;margin-top:5px;" >
<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Raster_Plot.png" style="margin-bottom:0px;">


#### Examples of Inferred Graphs
Below we find an example of the graph inferred by the algorithm: the left part illustrates the adjacency matrix of the actual graph (ground truth), where red, green and blue pixels represent excitatory, "non-existent" and inhibitory connections. The middle part shows the inferred *association matrix* and the right figure illustrates the ternary *adjacency matrix*, where "ternarification" has been done by picking roughly *p  n* connections in each column.

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Actual_Matrix.png" style="width:32%;float:left;" >

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Association_Matrix.png" style="width:32%;float:left;margin-left:10px;" >

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Adjacency_Matrix.png" style="width:32%;float:left;margin-left:10px;" >

<div style="clear:both;">
</div>



#### Precision and recall
We can also transform the analog association matrix to the *ternary adjacency matrix*, where the sign of entry (*i*,*j*) illustrates the inferred nature of the connection from neuron *i* to neuron *j*. A value of *+1* indicates an excitatory connection, *-1* indicates inhibitory and *0* a non-existent connection. We use the [K-Means algorithm](https://en.wikipedia.org/wiki/K-means_clustering) to categorize the weights of incoming connections for each neuron *i* (from the association matrix) to the above three classes. 

We can then evaluate the performance of STOCHASTIC NEUINF based on [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall), namely, how well the algorithm finds all different connection types without producing too many false positives or negatives. The following figure shows the results. In the paper, figure 6 illustrates the same results with more details and also in comparison with other similar algorithms.
<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Recurrent_Prec_Reca.png" style="height:400px;margin-left:7%;">



## How to (re)use the code
To reproduce the figures in the paper, check the ReadMe file in the code folder [here](https://github.com/saloot/NeuralNetworkTomography/blob/master/Codes%20Used%20in%20Papers/InverseNeural/Simulation%20Code/README.md).
To use the algorithm for your own database, please check the instructions given [here](https://github.com/saloot/NeuralNetworkTomography/tree/master/Network%20Tomography%20Toolbox).

### License
Copyright (C) 2015 Laboratory of Audiovisual Communications (LCAV),
Ecole Polytechnique Federale de Lausanne (EPFL),
CH-1015 Lausanne, Switzerland.
<a rel="license" href="https://en.wikipedia.org/wiki/GNU_General_Public_License"><img alt="GNU General Public License" style="border-width:0" src="http://rr.epfl.ch/img/GNU.png" /></a><br />


### Authors
Amin Karbasi, Amir Hesam Salvati and Martin Vetterli
Laboratory for Audiovisual Communications ([LCAV](http://lcav.epfl.ch)) at 
[EPFL](http://www.epfl.ch).
<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">


#### Contact
[Amir Hesam Salavati](mailto: saloot@gmail.com) <br>

