freeMem=`free | grep Mem | awk '{printf $4}'`
cpuFree=`top -b -n2 -p 1 | fgrep "Cpu(s)" | tail -1 | awk -F'id,' '{ split($1, vs, ","); v=vs[length(vs)]; sub("%", "", v); printf "%.1f", v }'`
cpuCores=`cat /proc/cpuinfo |grep processor|wc -l`
echo $freeMem,$cpuFree,$cpuCores