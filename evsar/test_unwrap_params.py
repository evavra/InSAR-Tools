#!/home/class239/anaconda3/bin/python3
import os
import sys

def main():
    """
    Test a variety of unwrapping parameters in parallel.
    
    Usage: test_unwrap_params.csh params.txt
    Run in interferogram directory (YYYYMMDD_YYYYMMDD)
    
    params.txt - file with rows of Snaphu parameters:
                 correlation_threshold maximum_discontinuity [<rng0>/<rngf>/<azi0>/<azif>]
    
                 Example:
                 0.1 10 1000/20000/30000/40000
                 0.1 20 1000/20000/30000/40000
                 0.2 20 1000/20000/30000/40000
    
    """

    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit()

    param_file = sys.argv[1]

    # Get current working directory
    cwd = os.getcwd()

    # Open parameter file
    # commands = []
    
    with open(param_file, 'r') as f:
        for i, line in enumerate(f):
            
            # Unpack parameters
            args = line[:-1].split()

            # Make new directory for test and move in
            test_dir = f'{i}_unwrap_test'

            if os.path.isdir(test_dir) == False:
                os.mkdir(test_dir)

            os.chdir(test_dir)

            # Save parameters
            with open('params.txt', 'w') as g:
                # g.write(str{i})
                g.write(' '.join(args) + '\n')

            cmd = 'snaphu_interp.csh ' + ' '.join(args) + ' >& log.tx & bg'
            print(i, cmd)
            os.system(cmd)

            os.chdir(cwd)
    return


if __name__ == '__main__':
    main()

