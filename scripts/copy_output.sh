# Usage: ./scripts/copy_output.sh <job_name>
# Copies the output of all job output files with job_name to out.txt for plotting
> out.txt
for filename in ./$1.o*
do
	cat $filename >> out.txt
done
