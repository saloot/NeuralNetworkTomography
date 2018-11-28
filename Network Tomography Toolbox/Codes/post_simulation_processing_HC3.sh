#file_ending="LIF_Spike_Times_I_1_S_1.0_C_8_B_400000_K_E_H_0.0_ii_2_0_F"

# TODO: 
# Make the function identify the command line arguments from the file

#file_ending="LIF_Spike_Times_I_1_S_1.0_C_8_B_400000_K_E_H_0.0_ii_2_0_F_150_f"

plot_var='f'
#plot_var='H'
no_structural_neurons=28
no_itr_over_dataset=4

for n_ind in {0..94}; do    

    for TT in 1200000; do
        for ff in 28; do

            echo "Running for neuron ${n_ind} and no structural neurons ${ff} and recording size of ${TT}"
            
            file_ending="HC3_ec013_198_processed_I_1_S_1.0_C_8_B_300000_K_E_H_0.0_ii_${no_itr_over_dataset}_${n_ind}"
            
            if [ ${ff} -gt 0 ]
            then
                file_ending2="${file_ending}_${plot_var}_${ff}"
            else
                file_ending2="${file_ending}"
            fi

            echo "${file_ending}"
            # Get the inferred files
            scp -r salavati@deneb2.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/W_Pll_${file_ending2}_T_${TT}*" ../Results/Inferred_Graphs/

            # Get the structural or hidden neurons informations
            scp -r salavati@deneb2.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/Hidden_or_Structured_Neurons_${file_ending2}*" ../Results/Inferred_Graphs/

            # Transforming to ternary   
            echo "${file_ending2}_T_${TT}"
            python Transform_to_Ternary.py -B 4 -N 94 -${plot_var} ${ff} -A "W_Pll_${file_ending2}_T_${TT}"

            # Calculating accuracy
            # python Calculate_Accuracy.py -B 4 -N 1000 -H ${no_hidden_neurons} -${plot_var} ${ff} -n ${n_ind} -F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Binary_${file_ending2}_T_${TT}"
        done   
    done
done

# Plot precision as a function of number of recorded samples
python Plot_Results.py  -N 1000 -n ${n_ind}  -H ${no_hidden_neurons}-F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Binary_${file_ending2}_T_***" -U R -O "1000000,2000000,3000000,7000000" -V T
#python Plot_Results.py  -N 1000 -n ${n_ind}  -F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Binary_${file_ending2}_" -U P -O "4000000,5000000,6000000,7000000"


# Plot precision as a function of number of hidden/structural information
python Plot_Results.py -o "0,1" -N 1000 -n ${n_ind} -H ${no_hidden_neurons} -F "../Data/Graphs/LIF_Actual_Connectivity.txt" -A "W_Binary_W_Pll_${file_ending}_${ff}_" -U "P" -O "50,100,150,200" -V ${plot_var}