# simpl5pn_temperature_modulation

* a 4 compartment (soma-AIS, basal, proximal apical, distal apical) simplified l5 pn biophysical model with temperature sensitivity in ion channels

### Python environment Requirement ###
* python 3.7
* numpy
* scipy
* matplotlib

### Files ###
* \func: functions
* \post_process_matlab <br />
\stat: curated data for the corresponding figures in the manuscript <br />
Figures.m: script that plots the figures <br />
temp.m: script that pool the raw data generated from temperature_modulation_test.py <br />
* Na_channel_test.py: plot channel gating as a function of temperature of long-term-inactivation Na channels
* single_neuron_test.py: sanity check file, create the biophysical simulation and generate a 300ms spontaneous spike train
* temperature_modulation_test.py: create neurons with different parameter and test the temperature modulation

### ionic current ###
* [0] na
* [1] kv
* [2] Im
* [3] ca
* [4] kca
* [5] nad

### ion channels gates ###
* na <br />
[0] na_m <br />
[1] na_h <br />
* kv <br />
[2] kv_m<br />
* Im <br />
[3] Im_m<br />
* ca <br />
[4] ca_m <br />
[5] ca_h <br />
* kca <br />
[6] kca_m <br />
* nad <br />
[7] nad_O1<br />
[8] nad_C1<br />
[9] nad_I1<br />
[10] nad_I2<br />
* ca dynamic <br />
[11] cai<br />
 

### contact ###
For any questions, please contact kjayant@purdue.edu.