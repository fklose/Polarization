from subprocess import run
import numpy as np

def findBounds(data):
	# Finds start and end points of ROOT array by checking for lines filled with stars
	start, end = 0, 0

	for i in range(0, len(data)):
		if data[i].strip("*").strip() == "":
			start = i
			break

	for i in range(0, len(data)):
		if data[len(data) - 1 - i].strip("*").strip() == "":
			end = len(data) - 1 - i
			break
	return start, end

def load(fname):
    
    completedProcess = run(["python", "./routines/extract.py", fname], capture_output=True)
    
    out = str(completedProcess.stdout)
    err = str(completedProcess.stderr)

    out = out.strip()
    out = out.split("\\n")
    
    titles = out[1]

	# clean up titles to label output
    titles = titles[1:-1]
    titles = titles.split("*")
    titles = [title.strip() for title in titles]

    start, stop = findBounds(out)
    data = out[start + 1:stop]

	# Remove stars from data and convert entries to numbers
    for n, line in enumerate(data):
        line = line.strip()
        line = line.split("*")

        cleanLine = []

        line = line[1:-1]

		# print(line)

        for item in line:
            item = item.strip()
            if item == "":
                cleanLine.append(float(0))
            else:
                cleanLine.append(float(item))

        data[n] = cleanLine

    data = np.array(data)

	# Save data to .csv
    # out_path = "/home/felix/fklose/Data/converted_ROOT_files/output" + fname[-10:-5] + ".csv"
    # np.savetxt(out_path, data, delimiter=",", header=",".join(titles)[:-1], comments="#", fmt="%.4f")

    return data