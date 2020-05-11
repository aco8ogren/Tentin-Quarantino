Yo yo, to set up an environment, simply run the GenEnv script in the folder ABOVE Tentin:
    Add execute permissions to GenEnv:
        chmod +x Tentin-Quarantino/SEIIRQ\ environment/GenEnv
    Run the file:
        ./Tentin-Quarantino/SEIIRQ\ environment/GenEnv

That should create an environment that is fully capable of running and debugging the SEIIRQ scripts, as well as using IPython (Jupyter).
If in the print statements that result while pip is intalling packages there is any red text, something might be messed up, so gimme a shout.

To activate the environment:
    source $PATH$/seiirqEnv/bin/activate
        where you should replace $PATH$ with the path to the directory in which GenEnv was run from (again should be just above Tentin)
    
    Now all python and pip command are implemented using the environments:
        To test, run which python or which pip
            should give you $PATH$/seiirqEnv/bin/python or $PATH$/seiirqEnv/bin/pip

        NOTE:   python command is automatically python3.7.0 (python3, not python2), as this environment was created using python3.7.0
                python2 will NOT be in this environment

To run in slurm scripts:
    instead of calling scripts with python3 path_to_script, run:
        $PATH$/seiirqEnv/bin/python path_to_script


Love, 
Josh