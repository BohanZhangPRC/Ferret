# wave_clus_python
Run waveclus with pyhton


input : 
 - neural_data.npy is a file that contains a matrix containing the raw neural signal (channels x (times x sr))
 - fs: sampling rate
 - base_data_path: directory of the input data
 - save_base_path : directory of saved spike sorted data.


output : 
Creates a folder 'spike_sorting', in whihch you will find:
 - plot of the spike sorting (.png)
 - ss_spike_times: a vector containing spikes times.
 - ss_spike_clusters:a vector containing the cluster from which the spikes times is from. 

