Introduction
===================
This repository contains the code to reproduce the figures for the paper [Inverse Neural](http://rr.epfl.ch/paper/KSV2015). In what follows, we describe how one can generate the figures for the paper and the ones in the supplementary materials.

Reproducing the Figures
===================
The codes are written in *Python*. The required libraries and dependencies are listed further down. Here, we discuss how to run the code. In principle, there are several to get the results shown in the paper, one of which is running the algorithm itself and the others involve generating data and evaluating the performance of the algorithm. The steps are listed below:

1. Running the inference algorithm (Cross Correlogram, GLM and NeuInf)
2. Transforming the resulting *analog* association matrix to *ternary* adjaceny matrix
3. Plotting the figures


### Step 1: Running the Inference Algorithms
#### Feed-forard networks
For the feed-forward case, open up a Terminal (in Mac OS/Linux) or Command Prompt (in Windows), navigate to the `Codes/` folder and type the following commands.

To run *Stochastic NeuInf* type

    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3

To run *Cross Correlogram* type

    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 4
    
    
For GLM, we have to switch to *MATLAB*, and use the code kindly provided by Prof. Jonathan Pillow [here](http://pillowlab.cps.utexas.edu/code_GLM.html).
We have modified the code to adapt it for our own needs, which is in the `GLM/` folder. To run the code, please open MALTAB, navigate to the `GLM\` folder
and perform the following commands

    >> run ./tools_mexcode/initialize_mexcode
    >> run setpaths
    >> T = Infer_GLM(2,[60,12],[15,3],0.2,0.3,'N')
    >> compare_results_GLM(2,[60,12],[15,3],0.2,0.3,'N',T)


#### Recurrent networks
For the recurrent case, open up a Terminal (in Mac OS/Linux) or Command Prompt (in Windows), navigate to the `Codes/` folder and type the following commands.

To run *Stochastic NeuInf* type

    $ python Inference_Tomography.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -M 3

To run *Cross Correlogram* type

    $ python Inference_Tomography.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -M 4
    
For GLM, please open MALTAB, navigate to the `GLM\` folder and perform the following commands

    >> run ./tools_mexcode/initialize_mexcode
    >> run setpaths
    >> T = Infer_GLM(2,[50],[11],0.4,0.1,'N')
    >> compare_results_GLM(2,[50],[11],0.4,0.1,'N',T)


### Step 2: Getteing the Ternary Adjacency Matrix
To transform the analog *association matrices* returned by the previous step, we can simply execute
    
**Feed-forward Networks**

    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -B 4 -M 3 -T 13499
    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -B 4 -M 4 -T 13499
    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -B 4 -M 8

**Recurrent Networks**

    $ python Transform_to_Ternary.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 3
    $ python Transform_to_Ternary.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 4
    $ python Transform_to_Ternary.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 8


### Step 3: Plotting the results
After performing the above steps, we can finally reproduce the plots.

For **Figure 2**, simply type

    $ python Figure2.py

For **Figure 3a**, execute the following command

    $ python Figure3a.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -K N -O "13494,13500,13500,13495"

To reproduce **Figures 4a**

    $ python Figure4a.py

For **Figure 4b**, we have to first run

    $ python Plot_Results.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 3 -O "1041,2082,3123,4164,5205,6246"
    $ python Plot_Results.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 4 -O "1041,2082,3123,4164,5205,6246"
    $ python Plot_Results.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 8 -O "100,1330,2560,3790,5020,6250" 

and then

    $ python Figure4b.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -G R -Q 0.1 -K N -O "6246,6250,6246"
    
For **Figure 5**

    $ python Figure5.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -G R -Q 0.1 -K N -T 6246
    
For **Figure 6**

    $ python Figure6.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -G R -Q 0.1 -K N -O "6246,6250,6246"

For **Figure 7a**, first execute the following commands in the `Codes/` folder (it will take quite long though!)

    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.3;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3
    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.3;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.4 -M 3
    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.3;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.5 -M 3
    
    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3 -B 4 -O "2249,4498,6747"
    
    $ python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3 -B 4 -O "2249,4498,6747" -f "B"
    $ python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.3;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3 -B 4 -O "2249,4498,6747" -f "B"
    $ python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.3;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.4 -M 3 -B 4 -O "2249,4498,6747" -f "B"
    $ python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.3;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.5 -M 3 -B 4 -O "2249,4498,6747" -f "B"
    
    
and then
    
    $ cd ReproduceFigures
    $ python Figure7a.py -E "60,12" -I "15,3" -L 2 -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -K N -p "0.2,0.3,0.3,0.3,0.3" -q "0.3,0.3,0.3,0.4,0.5" -B 4 -M 3 -O "6747,6747,6747,6747,6747" -c "b,orange,b,g,orange" -l "2,3"


For **Figure 7b**, first execute the following commands in the `Codes/` folder (it will take quite long though!)

    $ python Inference_Tomography.py -E "40" -I "10" -L 1 -P "0.25" -D "10" -Q 0.4 -M 3    
    $ python Inference_Tomography.py -E "40" -I "10" -L 1 -P "0.4" -D "10" -Q 0.25 -M 3
    $ python Inference_Tomography.py -E "40" -I "10" -L 1 -P "0.45" -D "10" -Q 0.3 -M 3    
    $ python Inference_Tomography.py -E "40" -I "10" -L 1 -P "0.3" -D "10" -Q 0.5 -M 3
    $ python Inference_Tomography.py -E "40" -I "10" -L 1 -P "0.3" -D "10" -Q 0.6 -M 3

    $ python Plot_Results.py -E "40" -I "10" -L 1 -P "0.25" -D "10" -Q 0.4 -B 4 -M 3 -f "B" -O "41,82,123,164,205,246"
    $ python Plot_Results.py -E "40" -I "10" -L 1 -P "0.4" -D "10" -Q 0.25 -B 4 -M 3 -f "B" -O "41,82,123,164,205,246"
    $ python Plot_Results.py -E "40" -I "10" -L 1 -P "0.45" -D "10" -Q 0.3 -B 4 -M 3 -f "B" -O "41,82,123,164,205,246"  
    $ python Plot_Results.py -E "40" -I "10" -L 1 -P "0.3" -D "10" -Q 0.5 -B 4 -M 3 -f "B" -O "41,82,123,164,205,246"
    $ python Plot_Results.py -E "40" -I "10" -L 1 -P "0.3" -D "10" -Q 0.6 -B 4 -M 3 -f "B" -O "41,82,123,164,205,246"
    
    $ cd ReproduceFigures
    $ python Figure7b.py -E "40" -I "10" -L 1 -D "10" -p "0.25,0.3,0.4,0.45,0.3,0.3" -q "0.4,0.5,0.25,0.3,0.5,0.6" -B 4 -M 3 -O "246,246,246,246,246,246" -c "b,g,orange,r,b,orange" -l "4,2"

For **Figure 8**

    $ python Figure8.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -K N -O "13500,13500,13495"
    
For **Figure 9**

    $ python Figure9.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -G R -Q 0.1 -K N -O "6246,6250"
    
For **Figure 10**

    $ python Figure10.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -K N -O "13500,13495"

Finally, for **Figure 11**

    $ python Figure11.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -K N -O "13500,13500"
    
### Dependencies
* A working distribution of [Python 2.7](https://www.python.org/downloads/).
* The code relies heavily on [Numpy](http://www.numpy.org/),
  [Scipy](http://www.scipy.org/), and [matplotlib](http://matplotlib.org).
* The [Brian simulator](http://briansimulator.org/).


### Some Possible Issues
* In Microsoft Windows, you might get *IO Error* due to long file names. To resolve this issue, make sure the `Simulation Code/`
is copied in a folder not far from the main dirve (e.g. `C:\Simulation Code` sounds like a very good option ;) ).


Credits and Contacts
===================
Amin Karbasi, Amir Hesam Salvati and Martin Vetterli
Laboratory for Audiovisual Communications ([LCAV](http://lcav.epfl.ch)) at 
[EPFL](http://www.epfl.ch).
<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">

#### Contacts and queries
[Amir Hesam Salavati](mailto: saloot[at]gmail[dot]com) <br>







