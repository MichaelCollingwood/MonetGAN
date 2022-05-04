class GeneratorWrapper:
    
    
    

    # Create the generator
    netG = Generator(ngpu).to(device)
    netG = nn.DataParallel(netG, list(range(ngpu)))

    # Print the model
    print(netG)

    # Choose a checkpoint to load from
    checkpoint_choice = False
    if checkpoint_choice:
        ckpt = torch.load(f'checkpoints/{checkpoint_choice}')
        netG.load_state_dict(ckpt['model_state_dict'])
        epoch_offsetG = ckpt['epoch']
    else:
        netG.apply(weights_init)
        epoch_offsetG = 0