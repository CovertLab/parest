
from subprocess import check_call
from os.path import join

from problems import DEFINITIONS

N_REPLICATES = 300 # the number of seeds
N_JOBS_PER_PROBLEM = 20 # total jobs to submit

(REPLICATES_PER_JOB, remainder) = divmod(N_REPLICATES, N_JOBS_PER_PROBLEM)
assert (remainder == 0)

for name in DEFINITIONS.viewkeys():
	seed = '$SLURM_ARRAY_TASK_ID'
	outdir = join(
		'out',
		name,
		'seed-{}'.format(seed)
		)

	check_call([
		'sbatch',
		'--array=0-{}'.format(
			N_JOBS_PER_PROBLEM-1
			),
		# '--dependency=singleton',
		'main.sbatch',
		name,
		str(REPLICATES_PER_JOB)
		])
