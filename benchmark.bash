if [ $# -eq 0 ] ; then
	file="results.txt"
else
	file=$1
fi

echo "Implementation, N (M = N + 2), Time (ns), IO Time (ns) if applicable" > $file

# Too slow to bother doing them all
for implementation in naive numpy scipy opencv; do
	for N in 16 32 64 128 256; do
		output=$(python diffusion.py -D 0.1 -L 2.0 -N $N -T 0.5 --implementation $implementation)
		time=$(echo "$output" | grep -oE '[0-9]+')
		echo "python_$implementation, $N, $time" >> $file
		echo "python_$implementation, $N, $time ns"
	done
done

for implementation in cupy torch; do
	for N in 16 32 64 128 256 512 1024; do
		output=$(python diffusion.py -D 0.1 -L 2.0 -N $N -T 0.5 --implementation $implementation)
		time=$(echo "$output" | grep -oE '[0-9]+')
		echo "python_$implementation, $N, $time" >> $file
		echo "python_$implementation, $N, $time ns"
	done
done

for implementation in naive openmp cuda; do
	for N in 16 32 64 128 256 512 1024; do
		output=$(./diffusion 0.1 2.0 $N 0.5 0 $implementation)
		if [[ "$implementation" == "cuda" ]]; then
			times=($(echo "$output" | grep -oE '[0-9]+'))
			time=${times[1]}
			echo "$implementation, $N, ${times[1]}, ${times[0]}" >> $file
			echo "cuda IO time is ${times[0]} ns"
		else
			time=$(echo "$output" | grep -oE '[0-9]+')
			echo "cpp_$implementation, $N, $time" >> $file
		fi
		echo "cpp_$implementation, $N, $time ns"
	done
done
