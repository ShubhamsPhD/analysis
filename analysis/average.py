import numpy as np

def average_all_frames(results):
    # We are initializing totals
    tilttotal = np.zeros(results['apt'].shape[1])  # Initialize for multiple layers
    tiltcount = np.zeros(results['apt'].shape[1])  # Count for each layer
    s2total = np.zeros(results['s2'].shape[1])
    apltotal = 0
    apttotal = np.zeros(results['apt'].shape[1])
    heighttotal = np.zeros(results['height'].shape[1])
    heightcount = 0

    # Iterate over all frames
    for i in range(results['apl'].shape[0]):
        # Filter tilt values for valid entries
        for layer in range(results['apt'].shape[1]):
            validtilt = results['tilt'][i][layer][results['tilt'][i][layer] <= 60]
            tilttotal[layer] += np.sum(validtilt)
            tiltcount[layer] += len(validtilt)

        # Sum s2, apl, apt, and height values across frames
        s2total += np.real(results['s2'][i])
        apltotal += results['apl'][i]
        apttotal += results['apt'][i]
        validheight = results['height'][i][results['height'][i] != 0]
        heighttotal += np.sum(validheight)
        heightcount += len(validheight)

    # Averages
    avgtilt = [tilttotal[layer] / tiltcount[layer] if tiltcount[layer] > 0 else 0 for layer in range(len(tiltcount))]
    avgs2 = s2total / results['apl'].shape[0]
    avgapl = apltotal / results['apl'].shape[0]
    avgapt = apttotal / results['apl'].shape[0]
    avgheight = heighttotal / results['apl'].shape[0]

    averages = {
        'Average Tilt': avgtilt,
        'Average s2': avgs2,
        'Average apl': avgapl,
        'Average apt': avgapt,
        'Average height': avgheight
    }
    print(averages)

    return averages

~                                                                                                                                                 
