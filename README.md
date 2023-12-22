## Mutual Stabilization in Chaotic Hindmarsh–Rose Neurons (Parker and Short, 2023)
Repository for the paper titled <i>Mutual Stabilization in Chaotic Hindmarsh–Rose Neurons</i> by John E. Parker and Kevin M. Short, published in <i>Dynamics</i>, 2023. The open-access paper can be found here: https://doi.org/10.3390/dynamics3020017.

Please cite this work appropriately if used as a reference.

The results in this work and the code provided here are an extension of a previous publication from the authors. As such, the details regarding the previous work can be found at the reference listed below:
- John E. Parker and Kevin M. Short, “Cupolets in a chaotic neuron model”, Chaos 32, 113104 (2022) https://doi.org/10.1063/5.0101667



<details> 
<summary>Dependencies</summary><blockquote>

All code was built using Python 3.11. We strongly advise using a virtual environment when running this code. Please see [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) on how to set up and activate virtual environment on your machine.

Once set up, to install the necessary Python modules please run:

`$ pip install -r requirements.txt`

You are now ready to run the code!
</blockquote>
</details>

<details> 
<summary>Repository Overview</summary><blockquote>
This repository has has two subdirectories that are required for these scripts <br>

</br>

The first is `paper_data` which holds all the previous published data (2022 paper) and where data from simulations will be stored.

The second is `paper_figures` which conatins all figures generated `generate_graphics.py` (see Figure Generation section).

All other files are for generation of the figures or analysis of the simulations. The vast majority of code within the scripts are commented. Questions can be directed to the corresponding author.

</blockquote>
</details>


<details> 
<summary>Figure Generation</summary><blockquote>
This section states how to recreate each of the figures in the manuscript. Figure 1, Figure 2, and Figure 3 are all reprouced from the 2022 publication listed above. However, Figure 3 can be reprodcued in this repository. 

Figures are generated with the `generate_graphics.py` script called from the command line using the flag `-f` followed by a list of desired figures (3,4,5.., etc).

For example, to generate figures 4, 6, and 7 run the following one the command line:

```
$ python generate_graphics.py -f 4 6 7
```

</blockquote>
</details>

<br></br>
Please contact the corresponding author regarding questions involving this work. The contact information can be found in the manuscript.

