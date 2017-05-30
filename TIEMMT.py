def TIEMMT(dI,px,hannScale,w0):
    # Estimate a solution to the transport of intensity equation and return
    # the resulting phase maps.
    # Inputs:
    # dI - First derivative of intensity with respect to defocus. Assuming this incorporates k0 and dz = (k0*dI)/dz
    # px - pixel size in microns (physical pixel size divided by magnification)
    # hannScale - scaling factor (amplitude) for Hanning bump
    # w0 - width of the Hanning bump
    # 
    # Returns:
    # Phase map and Fourier Domain TIE solvers with Hanning bump and global offset 
    # myTIEH, myTIEGO, phiHann, phiGo
    # Author: Q. Tyrell Davis 2017
    
    
    # Image dimensions
    Nx = len(dI[0,:])
    Ny = len(dI[:,0])
    
    X1 = np.arange(-Nx,Nx,1)
    Y1 = np.arange(-Ny,Ny,1)
    
    # set up the coordinates for the Fourier Domain
    myPadFactor = 2
    freqStepX = 2*np.pi / (Nx*myPadFactor)
    freqStepY = 2*np.pi / (Ny*myPadFactor)
    if (0):
        wX = np.arange(-np.pi,np.pi,freqStepX)
        wY = np.arange(-np.pi,np.pi,freqStepY)
    elif(0):
        wX = np.arange(-np.pi-freqStepX,np.pi-freqStepX,freqStepX)
        wY = np.arange(-np.pi-freqStepY,np.pi-freqStepY,freqStepY)
    elif(1):
        wX = np.arange(-np.pi+freqStepX,np.pi+freqStepX,freqStepX)
        wY = np.arange(-np.pi+freqStepY,np.pi+freqStepY,freqStepY)
    WX,WY = np.meshgrid(wX,wY) 
    wR = np.sqrt(WX**2+WY**2)
    
    ww = wR
    wR[wR >= np.pi/w0] = np.pi/(w0)
    # wR is used to define the Hanning Bump (set to make hBump value zero outside w0)
    # ww is used to calculate the TIE 
    

    if(0):
        freqStepX2 = 2*np.pi / (Nx)
        freqStepY2 = 2*np.pi / (Ny)
        wX2 = np.arange(-np.pi,np.pi,freqStepX2)
        wY2 = np.arange(-np.pi,np.pi,freqStepY2)
        WX2,WY2 = np.meshgrid(wX2,wY2)
        wR2 = np.sqrt(WX2**2+WY2**2)
        wR2[wR2 >= np.pi] = np.pi
        aPod = (1+np.cos(.5*wR2))
        if(1):
            hX = np.hamming(Nx)
            hY = np.hamming(Ny)
            aPod = np.sqrt(np.outer(hX,hX))
        if(1):
            plt.figure()
            plt.imshow(aPod,cmap=cmaps.viridis)
            plt.show()
    
    # define the Hanning window, used to stabilize the FD TIE
    # w0 defines how wide, hannScale defines the amplitude
    hann= (hannScale)*(1+np.cos(wR*np.pi/w0))
    dI = dI
    # Transform dI/dz to the Fourier Domain, use zero padding to 2X dimensions
    DI = np.fft.fftshift(np.fft.fft2(dI,[Ny*myPadFactor, Nx*myPadFactor]))

    # Compute the phase
    # Define the Laplacian FD transform pair w/ Hanning bump or global offset
    myTIEGO = (1/ ((4*np.pi**2)*(ww**2+hannScale)))
    myTIEH = (1/ ((4*np.pi**2)*(ww**2+hann)))
    
    # compute phase maps (all caps are FD)
    PHI = - DI * myTIEH
    PHINH = - DI * myTIEGO
    phiHann = np.real(np.fft.ifft2(np.fft.ifftshift(PHI)))
    phiGo = np.real(np.fft.ifft2(np.fft.ifftshift(PHINH)))
    
    # crop the phase image ()
    phiHann = phiHann[0:Ny,0:Nx]
    phiGo = phiGo[0:Ny,0:Nx]
    
    # adjust phase map result so that all values are positive
    if (1):
        phiHann = phiHann - np.min(phiHann)
        phiGo = phiGo -np.min(phiGo)
        
    return phiHann, phiGo, myTIEH, myTIEGO 
