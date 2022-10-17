from ROOT import TFile
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path : str) -> dict:
    """Loads relevant polarization data from the .root file at file_path

    Args:
        file_path (str): path where .root file is located

    Returns:
        dict: Dictionary containing numpy arrays of all necessary observables
    """
    
    file = TFile(file_path)
    ntuple = file.Get("ntuple")
    
    EVENT = []
    QDC_EIO0, QDC_EIO1, QDC_EIO2, QDC_EIO3, QDC_EIO4, QDC_EIO5 = [], [], [], [], [], []
    TTTL_OP_Beam = []
    TDC_PHOTO_DIODE, TDC_ION_MCP = [], []
    TDC_DL_X1, TDC_DL_X2 = [], []
    TDC_DL_Z1, TDC_DL_Z2 = [], []
    
    for n, entry in enumerate(ntuple):
        EVENT.append(n)
        
        QDC_EIO0.append(entry.QDC_EIO0)
        QDC_EIO1.append(entry.QDC_EIO1)
        QDC_EIO2.append(entry.QDC_EIO2)
        QDC_EIO3.append(entry.QDC_EIO3)
        QDC_EIO4.append(entry.QDC_EIO4)
        QDC_EIO5.append(entry.QDC_EIO5)
        
        TTTL_OP_Beam.append(entry.TTTL_OP_Beam)
        
        # Track TDC data. Note that TDC data is given as a vector which does not always contain a number
        # hence we need to check that there is at least one entry in each TDC vector before trying to grab
        # it otherwise the code will not work.
        # If a TDC vector does not contain an element add np.nan to the array instead this allows for the use 
        # of array operations (np.nan does not).
        if entry.TDC_PHOTO_DIODE_LE_Count > 0:
            TDC_PHOTO_DIODE.append(entry.TDC_PHOTO_DIODE_LE[0])
        else:
            TDC_PHOTO_DIODE.append(np.nan)
    
        if entry.TDC_ION_MCP_LE_Count > 0:
            TDC_ION_MCP.append(entry.TDC_ION_MCP_LE[0])
        else:
            TDC_ION_MCP.append(np.nan)
        
        if entry.TDC_DL_X1_LE_Count > 0:
            TDC_DL_X1.append(entry.TDC_DL_X1_LE[0])
        else:
            TDC_DL_X1.append(np.nan)    
        
        if entry.TDC_DL_X2_LE_Count > 0:
            TDC_DL_X2.append(entry.TDC_DL_X2_LE[0])
        else:
            TDC_DL_X2.append(np.nan)    
            
        if entry.TDC_DL_Z1_LE_Count > 0:
            TDC_DL_Z1.append(entry.TDC_DL_Z1_LE[0])
        else:
            TDC_DL_Z1.append(np.nan)
            
        if entry.TDC_DL_Z2_LE_Count > 0:
            TDC_DL_Z2.append(entry.TDC_DL_Z2_LE[0])
        else:
            TDC_DL_Z2.append(np.nan)
            
    # Define conversion factor to properly convert TDC data.
    # According to OnlineHistograms.cpp in rootana src code this value
    # comes from 'Equipment/VME/Settings/V1192_ns_conversion' and seems
    # to be a hardware specific value.
    v1192_to_ns = 0.098765625
    
    # Make dictionary containing numpy arrays of all observables also take care of scaling
    dict = {
        "EVENT"             : np.asarray(EVENT),
        "TTTL_OP_Beam"      : np.asarray(TTTL_OP_Beam) / 1e6,               # us
        "TDC_PHOTO_DIODE"   : np.asarray(TDC_PHOTO_DIODE) * v1192_to_ns,    # ns
        "TDC_ION_MCP"       : np.asarray(TDC_ION_MCP) * v1192_to_ns,        # ns
        "TDC_DL_X1"         : np.asarray(TDC_DL_X1) * v1192_to_ns,          # ns
        "TDC_DL_X2"         : np.asarray(TDC_DL_X2) * v1192_to_ns,          # ns
        "TDC_DL_Z1"         : np.asarray(TDC_DL_Z1) * v1192_to_ns,          # ns
        "TDC_DL_Z2"         : np.asarray(TDC_DL_Z2) * v1192_to_ns,          # ns
        "QDC_EIO0"          : np.asarray(QDC_EIO0),
        "QDC_EIO1"          : np.asarray(QDC_EIO1),
        "QDC_EIO2"          : np.asarray(QDC_EIO2),
        "QDC_EIO3"          : np.asarray(QDC_EIO3),
        "QDC_EIO4"          : np.asarray(QDC_EIO4),
        "QDC_EIO5"          : np.asarray(QDC_EIO5)
    }
    
    # THRESHOLD is used to convert QDC_EIO data into true (1) or false (0) logic levels
    THRESHOLD = 1000
    
    # Compute extra observables
    observables = {
        "X"     : dict["TDC_DL_X1"] - dict["TDC_DL_X2"],
        "Y"     : dict["TDC_ION_MCP"] - dict["TDC_PHOTO_DIODE"],
        "Z"     : dict["TDC_DL_Z1"] - dict["TDC_DL_Z2"],
        "BITS"  : 1  * (dict["QDC_EIO0"] > THRESHOLD) \
                + 2  * (dict["QDC_EIO1"] > THRESHOLD) \
                + 4  * (dict["QDC_EIO2"] > THRESHOLD) \
                + 8  * (dict["QDC_EIO3"] > THRESHOLD) \
                + 16 * (dict["QDC_EIO4"] > THRESHOLD) \
                + 32 * (dict["QDC_EIO5"] > THRESHOLD) \
    }
    
    dict.update(observables)
    
    return dict


def generate_histograms(data : dict, cuts : dict, show=False, save=True) -> None:
    # TODO Need to work on this a bit more
    # It is only saving the cuts at the moment as precut histos are overwritten
    
    # Generate initial histograms using RootAna bins
    if show or save:
        plt.hist(data["X"], bins=np.linspace(-100.5, 100.5, 1000), histtype="step")
        plt.xlabel("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)")
        plt.savefig("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0].png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist(data["Y"], bins=np.linspace(-10000.5, 10000.5, 10000), histtype="step")
        plt.xlabel("TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0] (ns)")
        plt.savefig("TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0].png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist(data["Z"], bins=np.linspace(-100.5, 100.5, 1000), histtype="step")
        plt.xlabel("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)")
        plt.savefig("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0].png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist(data["TTTL_OP_Beam"], bins=np.linspace(0, 4200, 420), histtype="step")
        plt.xlabel("TTTL_OP_Beam ($\mu$s)")
        plt.savefig("TTTL_OP_Beam.png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist2d(data["X"], data["Z"], bins=(np.linspace(-100.5, 100.5, 1000), np.linspace(-100.5, 100.5, 1000)))
        plt.xlabel("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)")
        plt.ylabel("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)")
        plt.savefig("DelayLineAnode-2d.png") if save else 0
        plt.show() if show else plt.close()

    # Generate Histograms with cut applied
    if show or save:
        plt.hist(data["X"][(cuts["X"][0] <= data["X"]) & (data["X"] <= cuts["X"][1])],
                bins=np.linspace(-100.5, 100.5, 1000), histtype="step",
                range=[cuts["X"][0], cuts["X"][1]])
        plt.xlabel("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)")
        plt.xlim([cuts["X"][0], cuts["X"][1]])
        plt.savefig("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0].png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist(data["Y"][(cuts["Y"][0] <= data["Y"]) & (data["Y"] <= cuts["Y"][1])],
                bins=np.linspace(-10000.5, 10000.5, 10000), histtype="step",
                range=[cuts["Y"][0], cuts["Y"][1]])
        plt.xlabel("TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0] (ns)")
        plt.xlim([cuts["Y"][0], cuts["Y"][1]])
        plt.savefig("TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0].png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist(data["Z"][(cuts["Z"][0] <= data["Z"]) & (data["Z"] <= cuts["Z"][1])], 
                bins=np.linspace(-100.5, 100.5, 1000), histtype="step",
                range=[cuts["Z"][0], cuts["Z"][1]])
        plt.xlabel("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)")
        plt.xlim([cuts["Z"][0], cuts["Z"][1]])
        plt.savefig("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0].png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        plt.hist(data["TTTL_OP_Beam"][(cuts["TTTL_OP_Beam"][0] <= data["TTTL_OP_Beam"]) 
                                    & (data["TTTL_OP_Beam"] <= cuts["TTTL_OP_Beam"][1])],
                bins=np.linspace(0, 4200, 420), histtype="step",
                range=[cuts["TTTL_OP_Beam"][0], cuts["TTTL_OP_Beam"][1]])
        plt.xlabel("TTTL_OP_Beam ($\mu$s)")
        plt.xlim([cuts["TTTL_OP_Beam"][0], cuts["TTTL_OP_Beam"][1]])
        plt.savefig("TTTL_OP_Beam.png") if save else 0
        plt.show() if show else plt.close()

    if show or save:
        X_CUT = (cuts["X"][0] <= data["X"]) & (data["X"] <= cuts["X"][1])
        Z_CUT = (cuts["Z"][0] <= data["Z"]) & (data["Z"] <= cuts["Z"][1])
        
        plt.hist2d(data["X"][X_CUT & Z_CUT], data["Z"][X_CUT & Z_CUT],
                bins=(np.linspace(-100.5, 100.5, 500), np.linspace(-100.5, 100.5, 500)),
                range=[[cuts["X"][0], cuts["X"][1]], [cuts["Z"][0], cuts["Z"][1]]])
        plt.xlabel("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)")
        plt.ylabel("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)")
        plt.xlim([cuts["X"][0], cuts["X"][1]])
        plt.ylim([cuts["Z"][0], cuts["Z"][1]])
        plt.savefig("DelayLineAnode-2d.png") if save else 0
        plt.show() if show else plt.close()
    
    return