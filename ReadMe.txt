1. Introduction:

Here is a  python code to calculate IBS growth time in a storage ring,  without or with betatron coupling. The Lebedev-Nagaitsev’s formulas  are used. For details, please check their original article:  "Multiple intrabeam scattering in X-Y coupled focusing systems", available at : arXiv:1812.09275  ( to be submitted to PRAB )

2. To run this code:
>python3  ./cal_ibs_coupling.py

3.  The input files:
1)	beam_parameters.txt:  define ion bunch parameters,  calculation control parameters  if you want to calculate evolution of eigen-emittances, and the file name holding linear optics parameters
2)	ibs_twiss_input_case2.txt: an example optics file

4. The output file:
The output will be printed on screen and also in a file “output.dat”.  Each column lists:
time(h) emit1(m)  emit2(m)  sigmas(m)  sigmap  tao1(min)  tao2(min)  tao3(min)
