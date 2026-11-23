# Empirical spectrum of a pure-noise correlation matrix, against the
# Marchenko-Pastur density. The whole project rests on these agreeing.
#
#   gnuplot plots/mp_overlay.gnuplot
#   gnuplot -e "lplus=2.25; lminus=0.25" plots/mp_overlay.gnuplot

set datafile separator ","
set key autotitle columnhead

if (!exists("infile"))  infile  = "results/mp_null.csv"
if (!exists("outfile")) outfile = "results/mp_overlay.png"
if (!exists("lminus"))  lminus  = 0.25
if (!exists("lplus"))   lplus   = 2.25

set terminal pngcairo size 1000,620 enhanced font "Helvetica,12"
set output outfile

set title "Sample correlation spectrum of pure noise vs Marchenko-Pastur" font "Helvetica,14"
set xlabel "eigenvalue {/Symbol l}"
set ylabel "density"
set grid lc rgb "#dddddd"
set style fill solid 0.35 noborder
set key top right

# The support edges: everything between them is indistinguishable from noise.
set arrow from lminus, graph 0 to lminus, graph 1 nohead dt 2 lc rgb "#888888"
set arrow from lplus,  graph 0 to lplus,  graph 1 nohead dt 2 lc rgb "#cc0000" lw 2
set label sprintf("{/Symbol l}_+ = %.3f\n(noise floor)", lplus) at lplus, graph 0.88 \
    offset 1,0 tc rgb "#cc0000"
set label sprintf("{/Symbol l}_- = %.3f", lminus) at lminus, graph 0.55 \
    offset 1,0 tc rgb "#666666"

plot infile using 1:2 with boxes lc rgb "#4878a8" title "empirical", \
     infile using 1:3 with lines lw 2.5 lc rgb "#cc0000" title "Marchenko-Pastur"
