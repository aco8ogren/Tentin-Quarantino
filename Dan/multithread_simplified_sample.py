
import multiprocessing
import numpy as np
import time

def par_fun(data, coreInd, const, EFlag):
    # I know there's MUCH better ways to do this operation BUT
        # for the sake of this example, we'll do it in a for
        # loop to keep it costly
    for i in range(len(data)):
        # Loop through each element of "data" and perform operation

        # Wrap bulk of code (where errors may occur) in a try-except
            # block so that we can signal all other workers to quit
            # whenever one of them has an issue
        try:

            # NOTE (TLDR): If each iteration of your for-loop is fast, call is_set() less often.
                # if each iteration is slow, call it more often.
            # NOTE: The is_set() call below is costly. Thus, you want to balance how often 
                #    you call it with how long you mind waiting for the other cores to stop.
                # If you call it on every iteration, all cores will stop very soon after an 
                #   error occurs. 
                # This is what I do in the Epdi model since each iteration of the for-loop 
                #   takes up to 100s to finish so if we do multiple iterations without checking
                #   the flag, we may have to wait a long time.
                # However, in this example, each iteration takes milliseconds to run. As such,
                #   calling is_set() becomes a large fraction of our runtime. We can fix this
                #   by only calling it every N number of iterations. This means at most N 
                #   iterations will occur on each core after an error BUT when each iteration
                #   takes a few ms, this results in a very small amount of downtime while also
                #   not wasting time checking the flag too often.

            # Check if EFlag is valuable (ie if multiprocessing) THEN check if it's set
                # "and" is short circuited so ErrFlag.is_set() only gets called when needed
            if (EFlag is not None) and (  not i%(round(len(data)/10)) ):
                # In this case, I'm checking the flag after each core finishes 10% of it's run
                    # Look at Epid model for an example where we check every time.

                if EFlag.is_set():
                    # When EFlag is set, another core has reported an error so this
                        # core should stop operation 
                    print('Detected EFlag; exiting loop')
                    break
            
            # ===== THIS IS WHERE YOU DO YOUR INDIVIDUAL OPERATIONS =====

            # Perform the desired operation on this element of the data
                # In this case, I'll put the result back into data just
                # cuz it's simple and minimizes memory usage.
            # You can of course do this another way. Look at the Epid model
                # for a sample of pre-allocating the output variable
                # and populating it iteratively here instead.
            # Notice how I use the "constants"
            data[i] = (const['mult']*data[i])**2 + const['offset']

            # I'll print some updates just so the user knows what the code's up to
                # In this case I'll print an update whenver we finish another 25%
                # of the for-loop. You can do your updates however you want
            # Look at the Epid model for some samples of useful updates
            if not i%(round(len(data)/4)):
                print('Core %d is on itr %d of %d'%(coreInd, i, len(data)))

        except:
            # When errors occur, set the EFlag so other cores will know to stop
            print('\n\nError occurred on itr %d in core %d\n'%(i, coreInd) + \
                  '    Setting error flag for other workers\n\n')

            # Obviously if we're not multiprocessing, then there's only one core 
                # so no need to "set" the flag. EFlag will be None then anyway
            if EFlag is not None:
                 EFlag.set()
            
            # Use "raise" to rethrow the error now that other cores have been signaled
                # Like this we still know what went wrong and can debug
            raise
    
    print('Core %d finished'%coreInd)
    
    # Return data and coreInd in this case (see apply_by_mp for explanation)
    return data, coreInd

def apply_by_mp(func, workers, args):
    # Create the pool of workers 
        # (use "with" context manager to deal with cleanup)
    with multiprocessing.Pool(processes=workers) as pool:
        # starmap will unpack the list of tuples in args
        # and will pass each tuple to the function (func)
        res = pool.starmap(func, args)  

    # Result is a list of the output of par_fun
        # Since par_fun in our case returns 2 things, we get a list
        # of tuples where element 0 of each tuple is the data and 
        # element 1 of each tuple is the coreInd 

    # The results will come back in whatever order the cores finished
        # So, if order matters, sort them back using the returned 
        # "coreInd" variable which denoted each worker
    # If order doesn't matter, you can just append/stack the results
        # see Epid model for an example where order doesn't matter

    # Pretending order matters, sort the results to reorder
    res=sorted(res,key=lambda x:x[1])
    # Return the recombined results
    return np.concatenate([i[0] for i in res])


# Whatever starts the multiprocessing NEEDS to be shielded so that it 
    # DOES NOT execute on every core. 
# Any functions and anything else that are needed by every core need
    # to either be outside the shielded region OR need to be provided
    # as arguments in the par_fun function call
# Might as well put anything that doesn't need to be done every time
    # also in the shielded region to minimize overhead
if __name__ == '__main__':
    #-- Create some artificial data to manipulate
    # Data (in this case, vector to be squared)
    data = np.arange(10000000)
    # Constant by which to multiply all values before squaring (sample of constant value)
    mult = 2
    # Constant to add to all values after squaring (sample of constant value)
    offset = 7

    #-- Set control parameters
    # coreIndber of cores to use
    workers = 8
    # Choose to multiprocess or not
    isMultiProc = False


    #-- Deal with multiprocessing/serialized option
    if isMultiProc:
        # Create error-handling flag when multiprocessing
            # provides a way for other cores to know that errors occurred
        EFlag = multiprocessing.Manager().Event()
    else:
        # Set Error handling flag to something we can discard later
        EFlag = None
        # Set workers to 1 so that we don't split the data
        workers = 1


    #-- Process data to prepare it for multiprocessing
    # Split the data into the coreIndber of cores that will be used
        # (not needed but drastically reduces memory usage on big data)
    data = np.array_split(data,workers)
        # If other data were needed by the cores, you'd want to split that
        # accordingly as well

    # If there is any data that is needed by all the cores, consider
        # putting it into a single dict to reduce coreIndber of arguments
    const = dict()
    const['mult'] = mult
    const['offset'] = offset


    #-- Create argument list
    # Create list of tuples that will be provided to each core
        # Each tuple corresponds to each core
        # The elements of the tuple should be the arguments to the function
    args = [(data[i], i, const, EFlag) for i in range(workers)]


    print('\nUsing %d workers\n'%workers)

    #-- Perform computation
    tic = time.time()
    if isMultiProc:
        # Run in parallel
        res = apply_by_mp(par_fun, workers, args)
    else:
        # Run seriallized
        res = par_fun(*(args[0]))
        
        # Don't need coreInd when running serialized so drop this
        res = res[0]


    toc = time.time()

    print('\nIterated output was:')
    print(res)
    print('\n\n')

    #-- For funsies: 
        # Let's just see how long this would take if we did the same 
        # calculation the smart way (vectorized). This just shows that
        # parallelization is not always the solution. In this case,
        # we obviously iterated over each element unnecessarily which
        # drastically increases the runtime. We can fix this by just
        # doing the math in one line as vector algebra. :)
    
    # Recombine data to get back to original array
    data = np.concatenate(data)

    tic1 = time.time()
    # Perform vectorized calculation
    resVec = (const['mult']*data)**2 + const['offset']
    toc1 = time.time()

    print('Vectorized output was:')
    print(res)
    print('\n\n')

    # Report runtime of each
    print('Time to execute with %d workers: %f' %(workers,toc-tic))
    print('Time to execute as vector: %f'%(toc1-tic1))

    # Check that the two methods provide the same result (including order!)
    if np.array_equal(resVec,res) :
        print('\nThe vector result was the same as the iterated result')
    else:
        print('\nThe vector result and iterated result were different')
