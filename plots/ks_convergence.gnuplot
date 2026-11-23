# Marchenko-Pastur is an ASYMPTOTIC law, so the only honest validation is that
# the Kolmogorov-Smirnov distance between the empirical and theoretical CDFs
# shrinks as N grows at fixed q. A single N proves nothing.
#
#   gnuplot plots/ks_convergence.gnuplot

set datafile separator ","
set key autotitle columnhead

if (!exists("infile"))  infile  = "results/mp_ks_sweep.csv"
if (!exists("outfile")) outfile = "results/ks_convergence.png"

set terminal pngcairo size 900,620 enhanced font "Helvetica,12"
set output outfile

set title "KS distance to the MP law vanishes as N grows (q fixed)" font "Helvetica,14"
set xlabel "N (assets)"
set ylabel "KS distance"
set logscale xy
set grid lc rgb "#dddddd"
set key top right

# Reference slopes, to read off the observed convergence rate by eye.
c1 = 0.3
plot infile using 1:4 with linespoints pt 7 ps 1.4 lw 2 lc rgb "#4878a8" title "measured KS", \
     c1 / x       with lines dt 2 lc rgb "#cc0000" title "~ 1/N", \
     c1 / sqrt(x) with lines dt 3 lc rgb "#888888" title "~ 1/sqrt(N)"
