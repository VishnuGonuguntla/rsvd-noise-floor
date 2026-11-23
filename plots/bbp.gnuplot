# The Baik-Ben Arous-Peche phase transition.
#
# Walk a planted factor's strength down toward 1 + sqrt(q) and it vanishes:
# the eigenvalue sinks onto the bulk edge and the eigenvector decorrelates
# from the truth. Below the threshold no algorithm recovers the factor, which
# is what makes the lambda_plus cut the right place to stop rather than
# merely a convenient one.
#
#   gnuplot -e "scrit=1.5; lplus=2.25" plots/bbp.gnuplot

set datafile separator ","
set key autotitle columnhead

if (!exists("infile"))  infile  = "results/bbp.csv"
if (!exists("outfile")) outfile = "results/bbp.png"
if (!exists("scrit"))   scrit   = 1.5
if (!exists("lplus"))   lplus   = 2.25

set terminal pngcairo size 1050,760 enhanced font "Helvetica,12"
set output outfile
set multiplot layout 2,1 title "BBP phase transition: when does a planted factor become visible?" \
    font "Helvetica,15"

set grid lc rgb "#dddddd"
set key bottom right
set xlabel ""

# --- top: the eigenvalue -------------------------------------------------
set ylabel "leading eigenvalue"
set arrow 1 from scrit, graph 0 to scrit, graph 1 nohead dt 2 lw 2 lc rgb "#cc0000"
set label 1 sprintf("BBP threshold\n1+{/Symbol \326}q = %.2f", scrit) at scrit, graph 0.82 \
    offset 1,0 tc rgb "#cc0000"
plot infile using 1:2 with lines lw 2.5 lc rgb "#cc0000" title "theory  s + qs/(s-1)", \
     infile using 1:3 with points pt 7 ps 1.2 lc rgb "#4878a8" title "measured (converged RSVD)", \
     lplus with lines dt 3 lw 2 lc rgb "#888888" title "bulk edge {/Symbol l}_+"

# --- bottom: the eigenvector ---------------------------------------------
set ylabel "|<u, v>|^2   (overlap with the true direction)"
set xlabel "planted spike strength s"
set yrange [-0.03:1.0]
set key top left
unset label 1
plot infile using 1:4 with lines lw 2.5 lc rgb "#cc0000" title "theory (Paul 2007)", \
     infile using 1:5 with points pt 7 ps 1.2 lc rgb "#4878a8" title "converged RSVD", \
     infile using 1:6 with points pt 6 ps 1.2 lc rgb "#e8a33d" title "under-iterated RSVD"

unset multiplot
