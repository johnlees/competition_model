# Competition model
Code for resident vs. challenger LV competition model for *S. pneumoniae*

### Requirements
As well as `python3`, the following python modules are needed to run this code (shown are the versions we used):
* dill==0.2.7.1
* numpy==1.13.3
* matplotlib==2.1.0
* lmfit==0.9.7
* statsmodels==0.8.0
* scipy=0.19.1
* numba==0.36.2
* sdeint=0.2.1
* elfi==0.7
* networkx==1.11 (note, must be < v2)

## Logistic fit to *in vivo* growth
See `growth_fit.ipynb` or run:
```
python growth_fit.py
```

## Running the model in each mode
Some example solutions and plots can be run with:
```
python ode_int.py --mode ode --t_com 3.8 --t_chal 24 --C_size 10
python ode_int.py --mode ctmc --t_chal 10 --C_size 20000 --beta 0.1
python ode_int.py --mode sde --t_com 3.8 --t_chal 6 --C_size 10 --beta 1.48 --R_size 10 --t_end 36 --g-RC 0.01 --g-CR 0.5 --resolution 2000
```
Will run the ODE, CTMC and SDE forumulations respectively, exhibiting different parameter values in each case.

Running the above, but replacing `ode_int.py` with `time_methods.py` will print the time taken to do 100 integrals to STDOUT. For example:
```
python time_methods.py --mode ctmc --t_chal 10 --C_size 20000 --beta 0.1
4.465044894997845
```

## Estimating model parameters with BOLFI
First of all use BOLFI to fit parameters based on the SDE model (takes around 6hrs):
```
python elfi_simulator.py --experiments experimental_data.txt
```

Once BOLFI has approximated the posterior, statistics about the fit, and samples from the posterior can be taken by running:
```
python elfi_post_sample.py --bolfi bolfi.pkl
```

## Generating plots over a range of parameter values
First, use `grid_model.py` to run the model over a range of:
1) lag in arrival time and challenger inoculum size for isogenic challengers
1) competition values for intergenic challengers

```
# isogenic
python grid_model.py --mode ctmc --repeats 20 --output ctmc --isogenic --grid-resolution 100

# intergenic
python grid_model.py --mode ctmc --repeats 20 --threshold 10 --output ctmc.t_chal1 --intergenic --t-chal 1 --grid-resolution 100
python grid_model.py --mode ctmc --repeats 20 --threshold 10 --output ctmc.t_chal2 --intergenic --t-chal 2 --grid-resolution 100
python grid_model.py --mode ctmc --repeats 20 --threshold 10 --output ctmc.t_chal4 --intergenic --t-chal 4 --grid-resolution 100
python grid_model.py --mode ctmc --repeats 20 --threshold 10 --output ctmc.t_chal6 --intergenic --t-chal 6 --grid-resolution 100
python grid_model.py --mode ctmc --repeats 20 --threshold 10 --output ctmc.t_chal24 --intergenic --t-chal 24 --grid-resolution 100
```

Then use these outputs to create contour plots. First for the 'resident wins' boundary in the isogenic case:
```
python isogenic_contour_plot.py --runs ctmc.isogenic_runs.txt --output isogenic_ode --boundary 1 --smooth
```
The domains up to 24hrs separately for each intergenic case:
```
for i in 1 2 4 6 24; 
  do python intergenic_contour_plot.py --runs ctmc.t_chal$i.intergenic_runs.txt \\
  --output intergenic_tchal$i --smooth --title "Intergenic challenger (t_chal = $i hrs)"  \\
  --stochastic; 
done
```
The domains for 1-6hrs on one plot:
```
cat ctmc.t_chal*.intergenic_runs.txt > ctmc.all.intergenic_runs.txt
python intergenic_multiple_contour_plot.py --runs ctmc.all.intergenic_runs.txt --smooth
```
