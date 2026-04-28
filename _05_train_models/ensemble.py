def run_ensemble(config, train_fcn):
    """
    
    """

    # Loop through the numer of ensemble members defined in split configuration
    for member in range(config.split_config.n_members):

        print(f'Running ensemble member {member:02d}')

        # Update runtime state so data loaders and path builders use current member
        config.runtime.member = member

        # Call the model traing function
        train_fcn(config)