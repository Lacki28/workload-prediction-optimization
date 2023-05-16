import subprocess


# ARIMA: n=2, sequence_length=12, target="mean_CPU_usage
def run_python_script(script_path, *args):
    try:
        cmd = ['python', script_path] + list(args)
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")
    except FileNotFoundError:
        print("Python is not installed or not found on your system.")


def main():
    for t in (1, 2, 3, 12):
        for history in (1, 12, 288):
            if t == 12 and history == 1:
                parameters = [str(t), str(24), 'mean_CPU_usage']
            else:
                parameters = [str(t), str(history), 'mean_CPU_usage']

            script_path = './arima/arima_one_job_one_output.py'
            run_python_script(script_path, *parameters)


if __name__ == "__main__":
    main()
