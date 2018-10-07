file_ending="LIF_Spike_Times_I_1_S_1.0_C_8_B_500000_K_E_H_0.0_ii_2_0_f"

# Get the inferred files
#scp -r salavati@deneb1.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/W_Pll_${file_ending}*" ../Results/Inferred_Graphs/

# Get the structural or hidden neurons informations
#scp -r salavati@deneb1.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/Hidden_or_Structured_Neurons_${file_ending}*" ../Results/Inferred_Graphs/

for TT in 400000 5000000 6000000 7000000; do
    for ff in 1 50 100 150 200; do
 
        echo "Running for no structural neurons ${ff} and recording size of ${TT}"

        # Transforming to ternary   
        python Transform_to_Ternary.py -B 4 -o "0,1" -N 1000 -f ${ff} -F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Pll_${file_ending}_${ff}_T_${TT}"

        # Calculating accuracy
        python Calculate_Accuracy.py -B 4 -o "0,1" -N 1000 -f ${ff} -n 0 -F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Binary_W_Pll_${file_ending}_${ff}_T_${TT}"
    done   
done

# Plot precision as a function of number of recorded samples
python Plot_Results.py -o "0,1" -N 1000 -n 0 -f 50 -F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Binary__${file_ending}_${ff}_" -U "P" -O "4000000,5000000,6000000,7000000"