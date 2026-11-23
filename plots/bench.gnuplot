# Randomized vs dense: O(N^2 k) against O(N^3).
#
# On log-log axes the claim is a slope difference -- dense should climb at
# about 3, the randomized routes at about 2 -- so the gap widens without
# bound rather than settling at a constant factor.
#
#   gnuplot plots/bench.gnuplot

set datafile separator ","
set key autotitle columnhead

if (!exists("infile"))  infile  = "results/bench.csv"
if (!exists("thrfile")) thrfile = "results/bench_threads.csv"
if (!exists("outfile")) outfile = "results/bench.png"

set terminal pngcairo size 1150,520 enhanced font "Helvetica,12"
set output outfile
set multiplot layout 1,2

# --- left: time vs N -----------------------------------------------------
set title "Top-k eigenpairs: randomized vs dense" font "Helvetica,13"
set xlabel "N (assets)"
set ylabel "wall time (ms, median)"
set logscale xy
set grid lc rgb "#dddddd"
set key top left

# Reference slopes. Anchored low so they sit under the data.
c3 = 2.0e-7
c2 = 1.0e-5
plot infile using 1:($6 > 0 ? $6 : 1/0) with linespoints pt 7 ps 1.3 lw 2 lc rgb "#cc0000" \
         title "dense (eigenvalues only)", \
     infile using 1:3 with linespoints pt 9 ps 1.3 lw 2 lc rgb "#888888" title "form C", \
     infile using 1:4 with linespoints pt 5 ps 1.3 lw 2 lc rgb "#4878a8" title "RSVD route A", \
     infile using 1:5 with linespoints pt 11 ps 1.3 lw 2 lc rgb "#3a9a5c" title "RSVD route B", \
     c3*x**3 with lines dt 2 lc rgb "#cc0000" title "~ N^3", \
     c2*x**2 with lines dt 3 lc rgb "#4878a8" title "~ N^2"

# --- right: thread scaling -----------------------------------------------
unset logscale y
set logscale x 2
set title "Thread scaling of C = Z^T Z / T" font "Helvetica,13"
set xlabel "threads"
set ylabel "speedup over one thread"
set key top left
set yrange [0:*]

plot thrfile using 1:4 with linespoints pt 7 ps 1.4 lw 2 lc rgb "#4878a8" title "measured", \
     x with lines dt 2 lc rgb "#888888" title "ideal"

unset multiplot
